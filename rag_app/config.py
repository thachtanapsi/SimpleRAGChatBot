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
    rerank_enabled: bool = False
    rerank_model: str = "BAAI/bge-reranker-v2-m3"
    rerank_revision: str | None = None
    rerank_device: str = "auto"
    rerank_batch_size: int = 8
    rerank_candidate_k: int = 40
    rerank_child_k: int = 12
    rerank_min_score: float = 0.5
    agentic_enabled: bool = False
    self_check_enabled: bool = False
    review_model: str = "gemma4:e2b"
    agent_timeout_seconds: int = 120
    agent_max_retrieval_rounds: int = 2
    agent_max_subqueries: int = 3
    agent_max_tool_calls: int = 4
    self_check_max_answer_attempts: int = 2
    graph_build_enabled: bool = False
    graph_enabled: bool = False
    graph_extractor_model: str = "gemma4:e2b"
    graph_max_hops: int = 2
    graph_worker_concurrency: int = 1
    max_upload_bytes: int = 50 * 1024 * 1024
    max_pdf_pages: int = 300
    min_pdf_text_chars: int = 100
    max_question_chars: int = 4000
    history_turns: int = 6
    max_session_messages: int = 100
    answer_num_predict: int = 2048
    advanced_answer_num_predict: int = 1024
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
        chat_model = os.environ.get("RAG_CHAT_MODEL", "gemma4:e2b")
        review_model = os.environ.get("RAG_REVIEW_MODEL", chat_model)
        settings = cls(
            base_dir=root,
            data_dir=path_value("RAG_DATA_DIR", "data"),
            papers_dir=path_value("RAG_PAPERS_DIR", "papers"),
            embedding_model=os.environ.get("RAG_EMBEDDING_MODEL", "BAAI/bge-m3"),
            embedding_revision=embedding_revision,
            embedding_device=os.environ.get("RAG_EMBED_DEVICE", "auto").lower(),
            embedding_batch_size=_env_int("RAG_EMBED_BATCH_SIZE", 32),
            chat_model=chat_model,
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
            rerank_enabled=_env_bool("RAG_RERANK_ENABLED", False),
            rerank_model=os.environ.get(
                "RAG_RERANK_MODEL", "BAAI/bge-reranker-v2-m3"
            ),
            rerank_revision=os.environ.get("RAG_RERANK_REVISION") or None,
            rerank_device=os.environ.get("RAG_RERANK_DEVICE", "auto").lower(),
            rerank_batch_size=_env_int("RAG_RERANK_BATCH_SIZE", 8),
            rerank_candidate_k=_env_int("RAG_RERANK_CANDIDATE_K", 40),
            rerank_child_k=_env_int("RAG_RERANK_CHILD_K", 12),
            rerank_min_score=_env_float("RAG_RERANK_MIN_SCORE", 0.5),
            agentic_enabled=_env_bool("RAG_AGENTIC_ENABLED", False),
            self_check_enabled=_env_bool("RAG_SELF_CHECK_ENABLED", False),
            review_model=review_model,
            agent_timeout_seconds=_env_int("RAG_AGENT_TIMEOUT_SECONDS", 120),
            agent_max_retrieval_rounds=_env_int(
                "RAG_AGENT_MAX_RETRIEVAL_ROUNDS", 2
            ),
            agent_max_subqueries=_env_int("RAG_AGENT_MAX_SUBQUERIES", 3),
            agent_max_tool_calls=_env_int("RAG_AGENT_MAX_TOOL_CALLS", 4),
            self_check_max_answer_attempts=_env_int(
                "RAG_SELF_CHECK_MAX_ANSWER_ATTEMPTS", 2
            ),
            graph_build_enabled=_env_bool("RAG_GRAPH_BUILD_ENABLED", False),
            graph_enabled=_env_bool("RAG_GRAPH_ENABLED", False),
            graph_extractor_model=os.environ.get(
                "RAG_GRAPH_EXTRACTOR_MODEL", review_model
            ),
            graph_max_hops=_env_int("RAG_GRAPH_MAX_HOPS", 2),
            graph_worker_concurrency=_env_int(
                "RAG_GRAPH_WORKER_CONCURRENCY", 1
            ),
            max_upload_bytes=_env_int("RAG_MAX_UPLOAD_BYTES", 50 * 1024 * 1024),
            max_pdf_pages=_env_int("RAG_MAX_PDF_PAGES", 300),
            min_pdf_text_chars=_env_int("RAG_MIN_PDF_TEXT_CHARS", 100),
            max_question_chars=_env_int("RAG_MAX_QUESTION_CHARS", 4000),
            history_turns=_env_int("RAG_HISTORY_TURNS", 6),
            max_session_messages=_env_int("RAG_MAX_SESSION_MESSAGES", 100),
            answer_num_predict=_env_int("RAG_ANSWER_NUM_PREDICT", 2048),
            advanced_answer_num_predict=_env_int(
                "RAG_ADVANCED_ANSWER_NUM_PREDICT", 1024
            ),
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
        if self.rerank_device not in {"auto", "cpu", "mps"}:
            raise ValueError("RAG_RERANK_DEVICE chỉ hỗ trợ auto, cpu hoặc mps")
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
            self.advanced_answer_num_predict,
            self.rewrite_num_predict,
            self.digest_max_batch_size,
            self.digest_writer_claim_size,
            self.rerank_batch_size,
            self.rerank_candidate_k,
            self.rerank_child_k,
            self.agent_timeout_seconds,
            self.agent_max_retrieval_rounds,
            self.agent_max_subqueries,
            self.agent_max_tool_calls,
            self.self_check_max_answer_attempts,
            self.graph_max_hops,
            self.graph_worker_concurrency,
        ) <= 0:
            raise ValueError("Các giới hạn cấu hình phải lớn hơn 0")
        if not 0 <= self.rerank_min_score <= 1:
            raise ValueError("RAG_RERANK_MIN_SCORE phải nằm trong [0, 1]")
        if not 256 <= self.advanced_answer_num_predict <= 1024:
            raise ValueError(
                "RAG_ADVANCED_ANSWER_NUM_PREDICT phải nằm trong [256, 1024]"
            )
        if not self.rerank_model.strip():
            raise ValueError("RAG_RERANK_MODEL không được để trống")
        if not self.review_model.strip() or not self.graph_extractor_model.strip():
            raise ValueError("Review/extractor model không được để trống")
        if self.agent_timeout_seconds > 120:
            raise ValueError("RAG_AGENT_TIMEOUT_SECONDS không được vượt quá 120")
        if self.agent_max_retrieval_rounds > 2:
            raise ValueError("RAG_AGENT_MAX_RETRIEVAL_ROUNDS không được vượt quá 2")
        if self.agent_max_subqueries > 3:
            raise ValueError("RAG_AGENT_MAX_SUBQUERIES không được vượt quá 3")
        if self.agent_max_tool_calls > 4:
            raise ValueError("RAG_AGENT_MAX_TOOL_CALLS không được vượt quá 4")
        if self.self_check_max_answer_attempts > 2:
            raise ValueError("RAG_SELF_CHECK_MAX_ANSWER_ATTEMPTS không được vượt quá 2")
        if self.graph_max_hops > 2:
            raise ValueError("RAG_GRAPH_MAX_HOPS không được vượt quá 2")
        if self.graph_worker_concurrency != 1:
            raise ValueError("RAG_GRAPH_WORKER_CONCURRENCY v1 phải bằng 1")
        if self.agentic_enabled and not self.hybrid_search:
            raise ValueError("RAG_AGENTIC_ENABLED yêu cầu RAG_HYBRID_SEARCH=1")
        if self.agentic_enabled and not self.rerank_enabled:
            raise ValueError("RAG_AGENTIC_ENABLED yêu cầu RAG_RERANK_ENABLED=1")
        if self.self_check_enabled and not self.agentic_enabled:
            raise ValueError("RAG_SELF_CHECK_ENABLED yêu cầu RAG_AGENTIC_ENABLED=1")
        if self.graph_enabled and not self.agentic_enabled:
            raise ValueError("RAG_GRAPH_ENABLED yêu cầu RAG_AGENTIC_ENABLED=1")
        if self.graph_enabled and not self.graph_build_enabled:
            raise ValueError("RAG_GRAPH_ENABLED yêu cầu RAG_GRAPH_BUILD_ENABLED=1")
