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
from .ingestion import IngestionService, IngestionWorker
from .logging import configure_logging
from .retrieval import ParentChildRetriever
from .storage import SQLiteStorage
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
        self.logger: Any
        self.ollama_ready = False
        self.ollama_error: str | None = None
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

        # Fingerprint khác dùng collection riêng. Xóa mapping/vector cũ rồi requeue.
        for document in self.storage.documents_requiring_reindex(
            self.index_fingerprint
        ):
            self.storage.requeue_document(document["id"])

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
        retriever = ParentChildRetriever(
            self.settings, self.storage, self.index, self.logger
        )
        generation_lock = asyncio.Semaphore(1)
        self.chat = ChatService(
            self.settings,
            self.storage,
            retriever,
            answer_llm,
            generation_lock,
            rewrite_llm,
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
        await self.check_ollama()
        self._started = True

    async def stop(self) -> None:
        if not self._started:
            return
        await self.worker.stop()
        self._started = False

    async def check_ollama(self) -> bool:
        try:
            async with httpx.AsyncClient(timeout=3.0) as client:
                response = await client.get(f"{self.settings.ollama_base_url}/api/tags")
                response.raise_for_status()
                models = {
                    item.get("name")
                    for item in response.json().get("models", [])
                    if isinstance(item, dict)
                }
            expected = self.settings.chat_model
            self.ollama_ready = expected in models or f"{expected}:latest" in models
            self.ollama_error = None if self.ollama_ready else "model_missing"
        except Exception as exc:
            self.ollama_ready = False
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
        status = (
            "ok"
            if sqlite_ok
            and fts_index_ready
            and chroma_ok
            and self.ollama_ready
            and full_report_payloads_ready
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
            "embedding_model": self.settings.embedding_model,
            "embedding_revision": self.embeddings.revision,
            "embedding_device": self.embeddings.device,
            "embedding_dimension": self.embeddings.dimension,
            "embedding_truncated_texts": self.embeddings.truncated_texts,
            "hybrid_search": self.settings.hybrid_search,
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
