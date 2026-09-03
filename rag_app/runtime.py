"""Khởi tạo một lần các tài nguyên nặng và phối hợp service."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import httpx
from langchain_ollama import ChatOllama

from .chat import ChatService
from .config import Settings
from .embeddings import EmbeddingService
from .errors import ConflictError, DocumentBusyError, DocumentNotFoundError
from .graph import GraphRetriever, GraphService
from .graph_extraction import LocalGraphExtractor, graph_extractor_fingerprint
from .graph_tool import GraphRetrievalTool
from .ingestion import IngestionService, IngestionWorker
from .logging import configure_logging
from .orchestration import (
    AdaptiveEvidenceGrader,
    AdaptiveQueryPlanner,
    AgenticOrchestrator,
    HybridRetrievalTool,
    LLMEvidenceGrader,
    LLMQueryPlanner,
)
from .reranking import RerankerService
from .retrieval import ParentChildRetriever
from .storage import SQLiteStorage
from .structured_tool import StructuredDecisionTool
from .vector_index import ChromaIndex


class Runtime:
    def __init__(self, settings: Settings | None = None):
        self.settings = settings or Settings.from_env()
        self.storage: SQLiteStorage
        self.embeddings: EmbeddingService
        self.index: ChromaIndex
        self.ingestion: IngestionService
        self.worker: IngestionWorker
        self.chat: ChatService
        self.reranker: RerankerService | None = None
        self.graph_service: GraphService | None = None
        self.graph_retriever: GraphRetriever | None = None
        self.logger: Any
        self.ollama_ready = False
        self.ollama_error: str | None = None
        self.review_model_ready = False
        self.graph_extractor_model_ready = False
        self.ollama_model_digests: dict[str, str] = {}
        self.graph_extractor_fingerprint: str | None = None
        self.index_fingerprint = self.settings.index_fingerprint
        self.collection_name = self.settings.collection_name
        self._started = False

    async def start(
        self,
        *,
        start_worker: bool = True,
        reset_interrupted_jobs: bool = True,
    ) -> None:
        if self._started:
            return
        self.settings.ensure_directories()
        self.logger = configure_logging(self.settings.logs_dir, self.settings.log_level)
        self.storage = SQLiteStorage(self.settings.sqlite_path)
        self.storage.initialise()
        reset_count = (
            self.storage.reset_interrupted_jobs() if reset_interrupted_jobs else 0
        )
        if reset_count:
            self.logger.info("jobs reset", extra={"event": "jobs_reset"})

        # Tải BGE-M3 đúng một lần. local_files_only bảo đảm không có network fallback.
        self.embeddings = await asyncio.to_thread(
            EmbeddingService,
            self.settings,
            self.logger,
        )
        self.index_fingerprint = self.settings.index_fingerprint_for(
            self.embeddings.revision
        )
        self.collection_name = self.settings.collection_name_for(
            self.embeddings.revision
        )
        self.index = ChromaIndex(
            self.settings.chroma_dir, self.collection_name, self.embeddings
        )

        # Reranker load failure is an observable degraded state, not an
        # ingestion/startup failure. Requests that require it fail closed.
        if self.settings.rerank_enabled:
            self.reranker = await asyncio.to_thread(
                RerankerService,
                self.settings,
                self.logger,
            )

        # Fingerprint khác dùng collection riêng. Xóa mapping/vector cũ rồi requeue.
        for document in self.storage.documents_requiring_reindex(
            self.index_fingerprint
        ):
            self.storage.requeue_document(document["id"])

        await self.check_ollama()

        answer_llm = ChatOllama(
            model=self.settings.chat_model,
            base_url=self.settings.ollama_base_url,
            reasoning=False,
            temperature=0.2,
            top_p=0.9,
            top_k=40,
            num_ctx=8192,
            num_predict=self.settings.answer_num_predict,
        )
        rewrite_llm = ChatOllama(
            model=self.settings.chat_model,
            base_url=self.settings.ollama_base_url,
            reasoning=False,
            temperature=0,
            num_ctx=4096,
            num_predict=self.settings.rewrite_num_predict,
        )
        review_num_predict = min(self.settings.answer_num_predict, 256)
        review_llm = ChatOllama(
            model=self.settings.review_model,
            base_url=self.settings.ollama_base_url,
            reasoning=False,
            temperature=0,
            num_ctx=8192,
            num_predict=review_num_predict,
        )
        planner_llm = ChatOllama(
            model=self.settings.review_model,
            base_url=self.settings.ollama_base_url,
            reasoning=False,
            temperature=0,
            num_ctx=8192,
            num_predict=min(self.settings.answer_num_predict, 256),
        )
        advanced_answer_num_predict = self.settings.advanced_answer_num_predict
        advanced_answer_llm = ChatOllama(
            model=self.settings.chat_model,
            base_url=self.settings.ollama_base_url,
            reasoning=False,
            temperature=0,
            num_ctx=8192,
            num_predict=advanced_answer_num_predict,
        )
        graph_llm = ChatOllama(
            model=self.settings.graph_extractor_model,
            base_url=self.settings.ollama_base_url,
            reasoning=False,
            temperature=0,
            num_ctx=8192,
            num_predict=min(self.settings.answer_num_predict, 1536),
        )
        retriever = ParentChildRetriever(
            self.settings,
            self.storage,
            self.index,
            self.logger,
            reranker=self.reranker,
        )
        generation_lock = asyncio.Semaphore(1)

        extractor_digest = self.ollama_model_digests.get(
            self.settings.graph_extractor_model
        )
        self.graph_extractor_fingerprint = graph_extractor_fingerprint(
            index_fingerprint=self.index_fingerprint,
            model_name=self.settings.graph_extractor_model,
            model_digest=extractor_digest,
        )
        graph_extractor = (
            LocalGraphExtractor(graph_llm)
            if self.settings.graph_build_enabled
            else None
        )
        self.graph_service = GraphService(
            self.storage,
            index_fingerprint=self.index_fingerprint,
            extractor_fingerprint=self.graph_extractor_fingerprint,
            extractor=graph_extractor,
            generation_lock=generation_lock,
            logger=self.logger,
            worker_concurrency=self.settings.graph_worker_concurrency,
        )
        if self.settings.graph_enabled:
            self.graph_retriever = GraphRetriever(
                self.storage,
                extractor_fingerprint=self.graph_extractor_fingerprint,
                max_hops=self.settings.graph_max_hops,
            )

        orchestrator: AgenticOrchestrator | None = None
        if self.settings.agentic_enabled:
            tools = [
                HybridRetrievalTool(retriever),
                StructuredDecisionTool(self.storage),
            ]
            allowed_tools = ["hybrid", "structured"]
            if self.graph_retriever is not None:
                tools.append(
                    GraphRetrievalTool(
                        self.graph_retriever,
                        reranker=self.reranker,
                        child_k=self.settings.rerank_child_k,
                        min_score=self.settings.rerank_min_score,
                    )
                )
                allowed_tools.append("graph")
            orchestrator = AgenticOrchestrator(
                tools,
                settings=self.settings,
                planner=AdaptiveQueryPlanner(
                    LLMQueryPlanner(planner_llm, generation_lock),
                    self.storage.list_ready_tickers,
                ),
                grader=AdaptiveEvidenceGrader(
                    LLMEvidenceGrader(review_llm, generation_lock),
                    min_rerank_score=self.settings.rerank_min_score,
                ),
                allowed_tools=allowed_tools,
                logger=self.logger,
            )
        self.chat = ChatService(
            self.settings,
            self.storage,
            retriever,
            answer_llm,
            generation_lock,
            rewrite_llm,
            orchestrator=orchestrator,
            review_llm=review_llm,
            advanced_llm=advanced_answer_llm,
            advanced_num_predict=advanced_answer_num_predict,
            required_reranker=self.reranker,
        )
        self.ingestion = IngestionService(
            self.settings,
            self.storage,
            self.index,
            self.logger,
            index_fingerprint=self.index_fingerprint,
            collection_name=self.collection_name,
        )
        # Resume a batch provenance upgrade that committed to SQLite before a
        # prior process could update the derived Chroma metadata.
        await asyncio.to_thread(
            self.ingestion.reconcile_vector_metadata_updates
        )
        self.worker = IngestionWorker(self.ingestion)
        self.ingestion.seed_papers_once()
        if start_worker:
            self.worker.start()
            if self.settings.graph_build_enabled:
                await self.graph_service.start()
        elif self.settings.graph_build_enabled:
            await asyncio.to_thread(self.graph_service.reconcile, apply=True)
        self._started = True

    async def stop(self) -> None:
        if not self._started:
            return
        if self.graph_service is not None:
            await self.graph_service.stop()
        await self.worker.stop()
        self._started = False

    async def check_ollama(self) -> bool:
        try:
            async with httpx.AsyncClient(timeout=3.0) as client:
                response = await client.get(f"{self.settings.ollama_base_url}/api/tags")
                response.raise_for_status()
            model_rows = [
                item
                for item in response.json().get("models", [])
                if isinstance(item, dict)
            ]
            models = {
                str(item.get("name") or item.get("model"))
                for item in model_rows
                if item.get("name") or item.get("model")
            }
            self.ollama_model_digests = {
                str(item.get("name") or item.get("model")): str(item["digest"])
                for item in model_rows
                if (item.get("name") or item.get("model")) and item.get("digest")
            }

            def model_ready(model: str) -> bool:
                return model in models or f"{model}:latest" in models

            expected = self.settings.chat_model
            self.ollama_ready = model_ready(expected)
            self.review_model_ready = model_ready(self.settings.review_model)
            self.graph_extractor_model_ready = model_ready(
                self.settings.graph_extractor_model
            )
            self.ollama_error = None if self.ollama_ready else "model_missing"
        except Exception as exc:
            self.ollama_ready = False
            self.review_model_ready = False
            self.graph_extractor_model_ready = False
            self.ollama_model_digests = {}
            self.ollama_error = type(exc).__name__
        return self.ollama_ready

    async def health(self) -> dict[str, Any]:
        sqlite_ok = await asyncio.to_thread(self.storage.healthcheck)
        fts_ok = await asyncio.to_thread(self.storage.fts_healthcheck)
        chroma_ok = await asyncio.to_thread(self.index.healthcheck)
        indexed_children = await asyncio.to_thread(self.index.count)
        fts_indexed_children = await asyncio.to_thread(self.storage.fts_count)
        full_report_payloads = await asyncio.to_thread(
            self.storage.full_report_payload_health
        )
        fts_index_ready = fts_ok and fts_indexed_children == indexed_children
        full_report_payloads_ready = (
            full_report_payloads["missing_ready_payloads"] == 0
            and full_report_payloads["pending_ready_decision_indexes"] == 0
        )
        if self.settings.rerank_enabled and self.reranker is not None:
            reranker = self.reranker.health()
            reranker_ready = bool(reranker["ready"])
        elif self.settings.rerank_enabled:
            reranker = {
                "ready": False,
                "model": self.settings.rerank_model,
                "revision": None,
                "device": None,
                "error": "not_initialised",
            }
            reranker_ready = False
        else:
            reranker = {
                "ready": False,
                "model": self.settings.rerank_model,
                "revision": None,
                "device": None,
                "error": None,
            }
            reranker_ready = True
        graph_health: dict[str, Any]
        graph_health_ok = True
        graph_service = getattr(self, "graph_service", None)
        if graph_service is None:
            graph_health = {
                "schema_version": 1,
                "extractor_fingerprint": self.graph_extractor_fingerprint,
                "eligible": 0,
                "covered": 0,
                "coverage": 1.0,
                "backlog": 0,
                "failures": 0,
                "counts": {},
            }
        else:
            try:
                graph_health = await asyncio.to_thread(graph_service.health)
            except Exception as exc:
                graph_health_ok = False
                graph_health = {
                    "schema_version": 1,
                    "extractor_fingerprint": self.graph_extractor_fingerprint,
                    "eligible": 0,
                    "covered": 0,
                    "coverage": 0.0,
                    "backlog": 0,
                    "failures": 0,
                    "counts": {},
                    "error": type(exc).__name__,
                }
        advanced_review_ready = (
            not (self.settings.agentic_enabled or self.settings.self_check_enabled)
            or self.review_model_ready
        )
        graph_model_ready = (
            not self.settings.graph_build_enabled
            or self.graph_extractor_model_ready
        )
        status = (
            "ok"
            if sqlite_ok
            and fts_index_ready
            and chroma_ok
            and self.ollama_ready
            and full_report_payloads_ready
            and reranker_ready
            and advanced_review_ready
            and graph_model_ready
            and graph_health_ok
            else "degraded"
        )
        return {
            "status": status,
            "sqlite": sqlite_ok,
            "fts": fts_ok,
            "fts_index_ready": fts_index_ready,
            "full_report_payloads_ready": full_report_payloads_ready,
            "full_report_payloads": full_report_payloads,
            "chroma": chroma_ok,
            "ollama": self.ollama_ready,
            "ollama_error": self.ollama_error,
            "chat_model": self.settings.chat_model,
            "review_model": self.settings.review_model,
            "review_model_ready": self.review_model_ready,
            "embedding_model": self.settings.embedding_model,
            "embedding_revision": self.embeddings.revision,
            "embedding_device": self.embeddings.device,
            "embedding_dimension": self.embeddings.dimension,
            "embedding_truncated_texts": self.embeddings.truncated_texts,
            "hybrid_search": self.settings.hybrid_search,
            "reranker": {"enabled": self.settings.rerank_enabled, **reranker},
            "advanced": {
                "agentic_enabled": self.settings.agentic_enabled,
                "self_check_enabled": self.settings.self_check_enabled,
                "graph_build_enabled": self.settings.graph_build_enabled,
                "graph_enabled": self.settings.graph_enabled,
                "answer_num_predict": self.settings.advanced_answer_num_predict,
            },
            "graph": {
                "build_enabled": self.settings.graph_build_enabled,
                "routing_enabled": self.settings.graph_enabled,
                "extractor_model": self.settings.graph_extractor_model,
                "extractor_model_ready": self.graph_extractor_model_ready,
                **graph_health,
            },
            "indexed_children": indexed_children,
            "fts_indexed_children": fts_indexed_children,
        }

    async def force_reindex(self) -> dict[str, list[str]]:
        """Xóa index dẫn xuất và đưa mọi PDF hợp lệ về hàng đợi.

        Caller phải khởi tạo runtime với ``start_worker=False`` để worker không
        claim một document trong lúc danh sách đang được chuẩn bị.
        """

        documents = self.storage.list_documents()
        indexing = [item["id"] for item in documents if item["status"] == "indexing"]
        if indexing:
            raise DocumentBusyError("Không thể reindex khi tài liệu đang indexing")

        scheduled: list[str] = []
        skipped: list[str] = []
        for document in documents:
            document_id = str(document["id"])
            stored_path = Path(document["stored_path"])
            source_available = (
                bool(self.storage.digest_sections(document_id))
                if document.get("source_type") == "trading_digest"
                else stored_path.is_file()
            )
            if document.get("error") == "deleting" or not source_available:
                skipped.append(document_id)
                continue

            child_ids = self.storage.child_ids_for_document(document_id)
            self.storage.requeue_document(document_id)
            scheduled.append(document_id)
            self.logger.info(
                "document reindex queued",
                extra={
                    "event": "document_reindex_queued",
                    "document_id": document_id,
                    "chunk_ids": child_ids[:20],
                },
            )

        self.logger.info(
            "force reindex queued",
            extra={"event": "force_reindex_queued", "count": len(scheduled)},
        )
        return {"scheduled": scheduled, "skipped": skipped}

    async def delete_document(self, document_id: str) -> None:
        current = self.storage.get_document(document_id)
        if not current:
            raise DocumentNotFoundError("Không tìm thấy tài liệu")
        # Retained daily digests must be rejected before acquiring the PDF
        # deletion lock; that lock deliberately changes status to `failed`.
        if current.get("source_type") == "trading_digest":
            raise ConflictError(
                "Research document được giữ vĩnh viễn và không thể xóa qua API"
            )
        document = self.storage.lock_document_for_deletion(document_id)
        if not document:
            raise DocumentNotFoundError("Không tìm thấy tài liệu")
        if document["status"] == "indexing":
            raise DocumentBusyError("Không thể xóa tài liệu đang index")
        child_ids = self.storage.child_ids_for_document(document_id)
        await asyncio.to_thread(
            self.index.delete, child_ids, document.get("collection_name")
        )
        self.storage.delete_document_rows(document_id)

        stored_path = Path(document["stored_path"]).resolve()
        uploads_dir = self.settings.uploads_dir.resolve()
        if stored_path.parent == uploads_dir and stored_path.is_file():
            stored_path.unlink()
        self.logger.info(
            "document deleted",
            extra={
                "event": "document_deleted",
                "document_id": document_id,
                "chunk_ids": child_ids[:20],
            },
        )
