"""Cấu hình tập trung, chỉ nhận biến môi trường có tiền tố ``RAG_``."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path


def _env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, default))


def _env_float(name: str, default: float) -> float:
    return float(os.environ.get(name, default))


@dataclass(frozen=True, slots=True)
class Settings:
    base_dir: Path
    data_dir: Path
    papers_dir: Path
    embedding_model: str = "BAAI/bge-m3"
    chat_model: str = "gemma4:e2b"
    ollama_base_url: str = "http://127.0.0.1:11434"
    parent_chunk_size: int = 2400
    parent_chunk_overlap: int = 200
    child_chunk_size: int = 600
    child_chunk_overlap: int = 100
    child_search_k: int = 20
    parent_search_k: int = 5
    relevance_threshold: float = 0.35
    max_upload_bytes: int = 50 * 1024 * 1024
    max_pdf_pages: int = 300
    min_pdf_text_chars: int = 100
    max_question_chars: int = 4000
    history_turns: int = 6
    max_session_messages: int = 100
    answer_num_predict: int = 800
    rewrite_num_predict: int = 96
    log_level: str = "INFO"

    @classmethod
    def from_env(cls, base_dir: Path | None = None) -> "Settings":
        configured_root = os.environ.get("RAG_BASE_DIR")
        root = (
            base_dir
            or (Path(configured_root).expanduser() if configured_root else Path.cwd())
        ).resolve()

        def path_value(name: str, default: str) -> Path:
            value = Path(os.environ.get(name, default)).expanduser()
            return (root / value).resolve() if not value.is_absolute() else value.resolve()

        settings = cls(
            base_dir=root,
            data_dir=path_value("RAG_DATA_DIR", "data"),
            papers_dir=path_value("RAG_PAPERS_DIR", "papers"),
            embedding_model=os.environ.get("RAG_EMBEDDING_MODEL", "BAAI/bge-m3"),
            chat_model=os.environ.get("RAG_CHAT_MODEL", "gemma4:e2b"),
            ollama_base_url=os.environ.get(
                "RAG_OLLAMA_BASE_URL", "http://127.0.0.1:11434"
            ).rstrip("/"),
            parent_chunk_size=_env_int("RAG_PARENT_CHUNK_SIZE", 2400),
            parent_chunk_overlap=_env_int("RAG_PARENT_CHUNK_OVERLAP", 200),
            child_chunk_size=_env_int("RAG_CHILD_CHUNK_SIZE", 600),
            child_chunk_overlap=_env_int("RAG_CHILD_CHUNK_OVERLAP", 100),
            child_search_k=_env_int("RAG_CHILD_SEARCH_K", 20),
            parent_search_k=_env_int("RAG_PARENT_SEARCH_K", 5),
            relevance_threshold=_env_float("RAG_RELEVANCE_THRESHOLD", 0.35),
            max_upload_bytes=_env_int("RAG_MAX_UPLOAD_BYTES", 50 * 1024 * 1024),
            max_pdf_pages=_env_int("RAG_MAX_PDF_PAGES", 300),
            min_pdf_text_chars=_env_int("RAG_MIN_PDF_TEXT_CHARS", 100),
            max_question_chars=_env_int("RAG_MAX_QUESTION_CHARS", 4000),
            history_turns=_env_int("RAG_HISTORY_TURNS", 6),
            max_session_messages=_env_int("RAG_MAX_SESSION_MESSAGES", 100),
            answer_num_predict=_env_int("RAG_ANSWER_NUM_PREDICT", 800),
            rewrite_num_predict=_env_int("RAG_REWRITE_NUM_PREDICT", 96),
            log_level=os.environ.get("RAG_LOG_LEVEL", "INFO").upper(),
        )
        settings.validate()
        return settings

    @property
    def sqlite_path(self) -> Path:
        return self.data_dir / "rag.sqlite3"

    @property
    def chroma_dir(self) -> Path:
        return self.data_dir / "chroma"

    @property
    def uploads_dir(self) -> Path:
        return self.data_dir / "uploads"

    @property
    def logs_dir(self) -> Path:
        return self.data_dir / "logs"

    @property
    def static_dir(self) -> Path:
        return Path(__file__).resolve().parent / "static"

    @property
    def index_fingerprint(self) -> str:
        payload = {
            "embedding_model": self.embedding_model,
            "parent_chunk_size": self.parent_chunk_size,
            "parent_chunk_overlap": self.parent_chunk_overlap,
            "child_chunk_size": self.child_chunk_size,
            "child_chunk_overlap": self.child_chunk_overlap,
            "normalise_embeddings": True,
            "schema": 1,
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(encoded).hexdigest()

    @property
    def collection_name(self) -> str:
        return f"rag_children_{self.index_fingerprint[:16]}"

    def ensure_directories(self) -> None:
        for path in (self.data_dir, self.chroma_dir, self.uploads_dir, self.logs_dir):
            path.mkdir(parents=True, exist_ok=True)

    def validate(self) -> None:
        for label, size, overlap in (
            ("parent", self.parent_chunk_size, self.parent_chunk_overlap),
            ("child", self.child_chunk_size, self.child_chunk_overlap),
        ):
            if size <= 0 or overlap < 0 or overlap >= size:
                raise ValueError(f"Cấu hình chunk {label} không hợp lệ")
        if not 0 <= self.relevance_threshold <= 1:
            raise ValueError("RAG_RELEVANCE_THRESHOLD phải nằm trong [0, 1]")
        if min(
            self.child_search_k,
            self.parent_search_k,
            self.max_pdf_pages,
            self.answer_num_predict,
            self.rewrite_num_predict,
        ) <= 0:
            raise ValueError("Các giới hạn cấu hình phải lớn hơn 0")
