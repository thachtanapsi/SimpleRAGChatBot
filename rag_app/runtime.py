"""Khởi tạo một lần các tài nguyên nặng và phối hợp service."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import httpx
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama

from .chat import ChatService
from .config import Settings
from .errors import DocumentBusyError, DocumentNotFoundError
from .ingestion import IngestionService, IngestionWorker
from .logging import configure_logging
from .retrieval import ParentChildRetriever
from .storage import SQLiteStorage
from .vector_index import ChromaIndex


class Runtime:
    def __init__(self, settings: Settings | None = None):
        self.settings = settings or Settings.from_env()
        self.storage: SQLiteStorage
        self.index: ChromaIndex
        self.ingestion: IngestionService
        self.worker: IngestionWorker
        self.chat: ChatService
        self.logger: Any
        self.ollama_ready = False
        self.ollama_error: str | None = None
        self._started = False

    async def start(self, *, start_worker: bool = True) -> None:
        if self._started:
            return
        self.settings.ensure_directories()
        self.logger = configure_logging(self.settings.logs_dir, self.settings.log_level)
        self.storage = SQLiteStorage(self.settings.sqlite_path)
        self.storage.initialise()
        reset_count = self.storage.reset_interrupted_jobs()
        if reset_count:
            self.logger.info("jobs reset", extra={"event": "jobs_reset"})

        # Tải BGE-M3 đúng một lần. local_files_only bảo đảm không có network fallback.
        embeddings = await asyncio.to_thread(
            HuggingFaceEmbeddings,
            model_name=self.settings.embedding_model,
            model_kwargs={"local_files_only": True},
            encode_kwargs={"normalize_embeddings": True},
        )
        self.index = ChromaIndex(
            self.settings.chroma_dir, self.settings.collection_name, embeddings
        )

        # Fingerprint khác dùng collection riêng. Xóa mapping/vector cũ rồi requeue.
        for document in self.storage.documents_requiring_reindex(
            self.settings.index_fingerprint
        ):
            child_ids = self.storage.child_ids_for_document(document["id"])
            self.index.delete(child_ids, document.get("collection_name"))
            self.storage.requeue_document(document["id"])

        answer_llm = ChatOllama(
            model=self.settings.chat_model,
            base_url=self.settings.ollama_base_url,
            temperature=0.2,
            top_p=0.9,
            top_k=40,
            num_ctx=8192,
            num_predict=self.settings.answer_num_predict,
        )
        rewrite_llm = ChatOllama(
            model=self.settings.chat_model,
            base_url=self.settings.ollama_base_url,
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
            self.settings, self.storage, self.index, self.logger
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
        chroma_ok = await asyncio.to_thread(self.index.healthcheck)
        status = "ok" if sqlite_ok and chroma_ok and self.ollama_ready else "degraded"
        return {
            "status": status,
            "sqlite": sqlite_ok,
            "chroma": chroma_ok,
            "ollama": self.ollama_ready,
            "ollama_error": self.ollama_error,
            "chat_model": self.settings.chat_model,
            "embedding_model": self.settings.embedding_model,
            "indexed_children": await asyncio.to_thread(self.index.count),
        }

    async def delete_document(self, document_id: str) -> None:
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
