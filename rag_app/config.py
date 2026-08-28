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


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    normalised = value.strip().lower()
    if normalised in {"1", "true", "yes", "on"}:
        return True
    if normalised in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} phải là 0/1 hoặc true/false")


@dataclass(frozen=True, slots=True)
class Settings:
    base_dir: Path
    data_dir: Path
    papers_dir: Path
    embedding_model: str = "BAAI/bge-m3"
    embedding_revision: str | None = None
    embedding_device: str = "auto"
    embedding_batch_size: int = 32
    embedding_dimension: int = 1024
    embedding_backend: str = "sentence-transformers"
    chat_model: str = "gemma4:e2b"
    ollama_base_url: str = "http://127.0.0.1:11434"
    parent_chunk_size: int = 2400
    parent_chunk_overlap: int = 200
    child_chunk_size: int = 600
    child_chunk_overlap: int = 100
    cross_page_enabled: bool = True
    cross_page_context_chars: int = 1200
    # child_search_k được giữ làm alias tương thích với cấu hình cũ.
    child_search_k: int = 30
    dense_search_k: int = 30
    lexical_search_k: int = 30
    parent_search_k: int = 5
    relevance_threshold: float = 0.35
    rrf_k: int = 60
    dense_weight: float = 0.7
    lexical_weight: float = 0.3
    hybrid_search: bool = False
    max_upload_bytes: int = 50 * 1024 * 1024
    max_pdf_pages: int = 300
    min_pdf_text_chars: int = 100
    max_question_chars: int = 4000
    history_turns: int = 6
    max_session_messages: int = 100
    answer_num_predict: int = 2048
    rewrite_num_predict: int = 96
    log_level: str = "INFO"
    rag_service_api_key: str | None = None
    digest_max_batch_size: int = 50
    digest_writer_claim_size: int = 25

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

        legacy_child_k = _env_int("RAG_CHILD_SEARCH_K", 30)
        embedding_revision = os.environ.get("RAG_EMBEDDING_REVISION") or None
        settings = cls(
            base_dir=root,
            data_dir=path_value("RAG_DATA_DIR", "data"),
            papers_dir=path_value("RAG_PAPERS_DIR", "papers"),
            embedding_model=os.environ.get("RAG_EMBEDDING_MODEL", "BAAI/bge-m3"),
            embedding_revision=embedding_revision,
            embedding_device=os.environ.get("RAG_EMBED_DEVICE", "auto").lower(),
            embedding_batch_size=_env_int("RAG_EMBED_BATCH_SIZE", 32),
            chat_model=os.environ.get("RAG_CHAT_MODEL", "gemma4:e2b"),
            ollama_base_url=os.environ.get(
                "RAG_OLLAMA_BASE_URL", "http://127.0.0.1:11434"
            ).rstrip("/"),
            parent_chunk_size=_env_int("RAG_PARENT_CHUNK_SIZE", 2400),
            parent_chunk_overlap=_env_int("RAG_PARENT_CHUNK_OVERLAP", 200),
            child_chunk_size=_env_int("RAG_CHILD_CHUNK_SIZE", 600),
            child_chunk_overlap=_env_int("RAG_CHILD_CHUNK_OVERLAP", 100),
            cross_page_enabled=_env_bool("RAG_CROSS_PAGE_ENABLED", True),
            cross_page_context_chars=_env_int(
                "RAG_CROSS_PAGE_CONTEXT_CHARS", 1200
            ),
            child_search_k=legacy_child_k,
            dense_search_k=_env_int("RAG_DENSE_SEARCH_K", legacy_child_k),
            lexical_search_k=_env_int("RAG_LEXICAL_SEARCH_K", 30),
            parent_search_k=_env_int("RAG_PARENT_SEARCH_K", 5),
            relevance_threshold=_env_float("RAG_RELEVANCE_THRESHOLD", 0.35),
            rrf_k=_env_int("RAG_RRF_K", 60),
            dense_weight=_env_float("RAG_DENSE_WEIGHT", 0.7),
            lexical_weight=_env_float("RAG_LEXICAL_WEIGHT", 0.3),
            hybrid_search=_env_bool("RAG_HYBRID_SEARCH", False),
            max_upload_bytes=_env_int("RAG_MAX_UPLOAD_BYTES", 50 * 1024 * 1024),
            max_pdf_pages=_env_int("RAG_MAX_PDF_PAGES", 300),
            min_pdf_text_chars=_env_int("RAG_MIN_PDF_TEXT_CHARS", 100),
            max_question_chars=_env_int("RAG_MAX_QUESTION_CHARS", 4000),
            history_turns=_env_int("RAG_HISTORY_TURNS", 6),
            max_session_messages=_env_int("RAG_MAX_SESSION_MESSAGES", 100),
            answer_num_predict=_env_int("RAG_ANSWER_NUM_PREDICT", 2048),
            rewrite_num_predict=_env_int("RAG_REWRITE_NUM_PREDICT", 96),
            log_level=os.environ.get("RAG_LOG_LEVEL", "INFO").upper(),
            rag_service_api_key=os.environ.get("RAG_SERVICE_API_KEY") or None,
            digest_max_batch_size=_env_int("RAG_DIGEST_MAX_BATCH_SIZE", 50),
            digest_writer_claim_size=_env_int("RAG_DIGEST_WRITER_CLAIM_SIZE", 25),
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
        return self.index_fingerprint_for(self.embedding_revision)

    def index_fingerprint_for(self, resolved_revision: str | None) -> str:
        payload = {
            "embedding_model": self.embedding_model,
            "embedding_revision": resolved_revision or "unresolved-local-cache",
            "embedding_dimension": self.embedding_dimension,
            "embedding_backend": self.embedding_backend,
            "parent_chunk_size": self.parent_chunk_size,
            "parent_chunk_overlap": self.parent_chunk_overlap,
            "child_chunk_size": self.child_chunk_size,
            "child_chunk_overlap": self.child_chunk_overlap,
            "cross_page_enabled": self.cross_page_enabled,
            "cross_page_context_chars": self.cross_page_context_chars,
            "bridge_child_strategy": "boundary_plus_parent_windows_v1",
            "normalise_embeddings": True,
            "schema": 6,
            "digest_sections": "independent_provenance_v2",
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(encoded).hexdigest()

    @property
    def collection_name(self) -> str:
        return self.collection_name_for(self.embedding_revision)

    def collection_name_for(self, resolved_revision: str | None) -> str:
        return f"rag_children_{self.index_fingerprint_for(resolved_revision)[:16]}"

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
        if self.embedding_device not in {"auto", "cpu", "mps"}:
            raise ValueError("RAG_EMBED_DEVICE chỉ hỗ trợ auto, cpu hoặc mps")
        if self.embedding_dimension <= 0 or self.embedding_batch_size <= 0:
            raise ValueError("Cấu hình embedding phải lớn hơn 0")
        if self.cross_page_context_chars <= 0:
            raise ValueError("RAG_CROSS_PAGE_CONTEXT_CHARS phải lớn hơn 0")
        if self.dense_weight <= 0 or self.lexical_weight < 0:
            raise ValueError("Dense weight phải dương và lexical weight không được âm")
        if min(
            self.child_search_k,
            self.dense_search_k,
            self.lexical_search_k,
            self.parent_search_k,
            self.rrf_k,
            self.max_pdf_pages,
            self.answer_num_predict,
            self.rewrite_num_predict,
            self.digest_max_batch_size,
            self.digest_writer_claim_size,
        ) <= 0:
            raise ValueError("Các giới hạn cấu hình phải lớn hơn 0")
