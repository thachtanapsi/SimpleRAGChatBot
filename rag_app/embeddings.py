"""BGE-M3 dense embedding chạy offline với cấu hình và chẩn đoán tập trung."""

from __future__ import annotations

import math
from typing import Any, Callable, Sequence

import torch
from langchain_core.embeddings import Embeddings
from sentence_transformers import SentenceTransformer

from .config import Settings


def resolve_embedding_device(requested: str) -> str:
    """Chọn MPS khi dùng ``auto`` và backend thật sự khả dụng."""

    if requested == "auto":
        return "mps" if torch.backends.mps.is_available() else "cpu"
    if requested == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("RAG_EMBED_DEVICE=mps nhưng PyTorch không dùng được MPS")
    return requested


def _model_revision(model: Any, configured_revision: str | None) -> str:
    """Lấy commit đã resolve từ Transformers mà không gọi Hugging Face Hub."""

    try:
        first_module = model._first_module()
        config = first_module.auto_model.config
        commit_hash = getattr(config, "_commit_hash", None)
        if commit_hash:
            return str(commit_hash)
    except (AttributeError, TypeError):
        pass
    return configured_revision or "local-cache"


class EmbeddingService(Embeddings):
    """Adapter SentenceTransformers dùng chung cho ingestion và Chroma query."""

    def __init__(
        self,
        settings: Settings,
        logger: Any | None = None,
        *,
        model_factory: Callable[..., Any] = SentenceTransformer,
    ):
        self.settings = settings
        self.logger = logger
        self.device = resolve_embedding_device(settings.embedding_device)
        model_kwargs: dict[str, Any] = {
            "device": self.device,
            "local_files_only": True,
        }
        if settings.embedding_revision:
            model_kwargs["revision"] = settings.embedding_revision
        self.model = model_factory(settings.embedding_model, **model_kwargs)
        dimension_getter = getattr(self.model, "get_embedding_dimension", None)
        if dimension_getter is None:
            dimension_getter = self.model.get_sentence_embedding_dimension
        self.dimension = int(dimension_getter())
        if self.dimension != settings.embedding_dimension:
            raise RuntimeError(
                "Embedding dimension không khớp: "
                f"mong đợi {settings.embedding_dimension}, nhận {self.dimension}"
            )
        self.max_sequence_length = int(getattr(self.model, "max_seq_length", 0) or 0)
        self.revision = _model_revision(self.model, settings.embedding_revision)
        self.embedded_texts = 0
        self.truncated_texts = 0

    def _detect_truncation(self, texts: Sequence[str]) -> int:
        tokenizer = getattr(self.model, "tokenizer", None)
        if not texts or tokenizer is None or self.max_sequence_length <= 0:
            return 0
        try:
            encoded = tokenizer(
                list(texts),
                add_special_tokens=True,
                padding=False,
                truncation=False,
                return_attention_mask=False,
                verbose=False,
            )
            lengths = [len(token_ids) for token_ids in encoded["input_ids"]]
        except (KeyError, TypeError, ValueError):
            return 0
        truncated = sum(length > self.max_sequence_length for length in lengths)
        self.truncated_texts += truncated
        if truncated and self.logger:
            self.logger.warning(
                "embedding input truncated",
                extra={"event": "embedding_truncated", "count": truncated},
            )
        return truncated

    def _encode(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        self._detect_truncation(texts)
        vectors = self.model.encode(
            list(texts),
            batch_size=self.settings.embedding_batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        values = vectors.tolist() if hasattr(vectors, "tolist") else list(vectors)
        result = [[float(value) for value in vector] for vector in values]
        for vector in result:
            if len(vector) != self.dimension or not all(math.isfinite(v) for v in vector):
                raise RuntimeError("Model trả về embedding không hợp lệ")
        self.embedded_texts += len(result)
        return result

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._encode(texts)

    def embed_query(self, text: str) -> list[float]:
        vectors = self._encode([text])
        return vectors[0]
