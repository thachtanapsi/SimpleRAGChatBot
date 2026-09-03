"""Cross-encoder reranking local, bounded và có trạng thái health rõ ràng."""

from __future__ import annotations

import math
import threading
from dataclasses import dataclass
from typing import Any, Callable, Sequence

import torch
from sentence_transformers import CrossEncoder

from .config import Settings
from .errors import ServiceUnavailableError


class RerankerUnavailableError(ServiceUnavailableError):
    """Reranker được yêu cầu nhưng model local không thể dùng an toàn."""

    error_code = "rag_reranker_unavailable"


@dataclass(frozen=True, slots=True)
class RerankCandidate:
    """Phần dữ liệu tối thiểu được gửi vào cross-encoder."""

    id: str
    text: str


@dataclass(frozen=True, slots=True)
class RerankResult:
    """Điểm sigmoid đã chuẩn hóa của một child candidate."""

    id: str
    score: float


def resolve_reranker_device(requested: str) -> str:
    """Giữ cùng chính sách device với embedding nhưng độc lập cấu hình."""

    if requested == "auto":
        return "mps" if torch.backends.mps.is_available() else "cpu"
    if requested == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("RAG_RERANK_DEVICE=mps nhưng PyTorch không dùng được MPS")
    return requested


def _resolved_revision(model: Any, configured_revision: str | None) -> str:
    try:
        commit_hash = getattr(model.model.config, "_commit_hash", None)
        if commit_hash:
            return str(commit_hash)
    except (AttributeError, TypeError):
        pass
    return configured_revision or "local-cache"


def _score_value(raw: Any) -> float:
    """Chấp nhận scalar numpy/torch và output một nhãn của CrossEncoder."""

    if hasattr(raw, "tolist"):
        raw = raw.tolist()
    if isinstance(raw, (list, tuple)):
        if len(raw) != 1:
            raise ValueError("Reranker phải trả đúng một score cho mỗi candidate")
        raw = raw[0]
    score = float(raw)
    if not math.isfinite(score):
        raise ValueError("Reranker trả score không hữu hạn")
    # activation_fn là sigmoid, clamp chỉ bảo vệ model/test double sai số nhỏ.
    return max(0.0, min(1.0, score))


class RerankerService:
    """Adapter ``CrossEncoder`` không bao giờ tải model từ network.

    Lỗi load được giữ lại thay vì làm hỏng ingestion/startup. Caller có thể đưa
    trạng thái này vào health check; một request thực sự cần rerank sẽ nhận lỗi
    có mã ổn định thay vì âm thầm quay về RRF.
    """

    def __init__(
        self,
        settings: Settings,
        logger: Any | None = None,
        *,
        model_factory: Callable[..., Any] = CrossEncoder,
    ):
        self.settings = settings
        self.logger = logger
        self.model_name = str(
            getattr(settings, "rerank_model", "BAAI/bge-reranker-v2-m3")
        )
        self.configured_revision = getattr(settings, "rerank_revision", None)
        self.batch_size = int(getattr(settings, "rerank_batch_size", 8))
        self.max_length = 512
        self.device: str | None = None
        self.revision: str | None = None
        self.model: Any | None = None
        self.load_error: str | None = None
        self._lock = threading.Lock()

        try:
            self.device = resolve_reranker_device(
                str(getattr(settings, "rerank_device", "auto"))
            )
            model_kwargs: dict[str, Any] = {
                "device": self.device,
                "local_files_only": True,
                "max_length": self.max_length,
                "activation_fn": torch.nn.Sigmoid(),
            }
            if self.configured_revision:
                model_kwargs["revision"] = self.configured_revision
            self.model = model_factory(self.model_name, **model_kwargs)
            self.revision = _resolved_revision(self.model, self.configured_revision)
        except Exception as exc:  # Model cache/device failure is an operational state.
            self.model = None
            self.load_error = type(exc).__name__
            if self.logger:
                self.logger.warning(
                    "reranker unavailable",
                    extra={
                        "event": "reranker_unavailable",
                        "reason_code": self.load_error,
                    },
                )

    @property
    def available(self) -> bool:
        return self.model is not None and self.load_error is None

    @property
    def ready(self) -> bool:
        """Alias thuận tiện cho runtime health checks."""

        return self.available

    @property
    def error(self) -> str | None:
        """Alias ổn định cho health/runtime mà không lộ nội dung exception."""

        return self.load_error

    def health(self) -> dict[str, Any]:
        return {
            "ready": self.ready,
            "model": self.model_name,
            "revision": self.revision,
            "device": self.device,
            "error": self.load_error,
        }

    def rerank(
        self,
        query: str,
        candidates: Sequence[RerankCandidate],
        *,
        top_k: int | None = None,
        min_score: float = 0.0,
    ) -> list[RerankResult]:
        if not self.available:
            raise RerankerUnavailableError(
                "Reranker local chưa sẵn sàng; không thể bảo đảm chất lượng advanced RAG"
            )
        if not candidates:
            return []
        if top_k is not None and top_k <= 0:
            raise ValueError("top_k của reranker phải lớn hơn 0")
        if not 0 <= min_score <= 1:
            raise ValueError("min_score của reranker phải nằm trong [0, 1]")
        ids = [str(item.id) for item in candidates]
        if len(set(ids)) != len(ids):
            raise ValueError("Candidate ID của reranker phải duy nhất")

        pairs = [(query, item.text) for item in candidates]
        try:
            with self._lock:
                raw_scores = self.model.predict(
                    pairs,
                    batch_size=self.batch_size,
                    show_progress_bar=False,
                    convert_to_numpy=True,
                )
            values = list(raw_scores)
            if len(values) != len(candidates):
                raise ValueError("Reranker trả sai số lượng score")
            results = [
                RerankResult(id=item.id, score=_score_value(score))
                for item, score in zip(candidates, values, strict=True)
            ]
        except RerankerUnavailableError:
            raise
        except Exception as exc:
            if self.logger:
                self.logger.warning(
                    "reranker inference failed",
                    extra={
                        "event": "reranker_inference_failed",
                        "reason_code": type(exc).__name__,
                        "candidate_count": len(candidates),
                    },
                )
            raise RerankerUnavailableError("Reranker local không thể chấm điểm") from exc

        results = [item for item in results if item.score >= min_score]
        results.sort(key=lambda item: (-item.score, item.id))
        return results if top_k is None else results[:top_k]
