from __future__ import annotations

import math

import pytest

from rag_app.embeddings import EmbeddingService, resolve_embedding_device


class FakeTokenizer:
    def __call__(self, texts, **_kwargs):
        return {"input_ids": [list(range(len(text.split()))) for text in texts]}


class FakeConfig:
    _commit_hash = "resolved-commit"


class FakeTransformer:
    config = FakeConfig()


class FakeFirstModule:
    auto_model = FakeTransformer()


class FakeSentenceTransformer:
    max_seq_length = 3
    tokenizer = FakeTokenizer()

    def __init__(self, model_name, **kwargs):
        self.model_name = model_name
        self.kwargs = kwargs
        self.encode_kwargs = None

    def _first_module(self):
        return FakeFirstModule()

    def get_sentence_embedding_dimension(self):
        return 1024

    def encode(self, texts, **kwargs):
        self.encode_kwargs = kwargs
        vector = [1.0] + [0.0] * 1023
        return [vector[:] for _ in texts]


def test_embedding_service_batches_normalises_and_detects_truncation(settings):
    service = EmbeddingService(
        settings,
        model_factory=FakeSentenceTransformer,
    )
    vectors = service.embed_documents(["one two", "one two three four"])

    assert len(vectors) == 2
    assert len(vectors[0]) == 1024
    assert math.isclose(math.sqrt(sum(value * value for value in vectors[0])), 1.0)
    assert service.model.encode_kwargs["batch_size"] == 32
    assert service.model.encode_kwargs["normalize_embeddings"] is True
    assert service.truncated_texts == 1
    assert service.revision == "resolved-commit"
    assert service.device == "cpu"


def test_auto_device_uses_mps_only_when_available(monkeypatch):
    monkeypatch.setattr("rag_app.embeddings.torch.backends.mps.is_available", lambda: True)
    assert resolve_embedding_device("auto") == "mps"
    monkeypatch.setattr("rag_app.embeddings.torch.backends.mps.is_available", lambda: False)
    assert resolve_embedding_device("auto") == "cpu"
    with pytest.raises(RuntimeError, match="không dùng được MPS"):
        resolve_embedding_device("mps")


def test_embedding_dimension_mismatch_fails_fast(settings):
    class WrongDimensionModel(FakeSentenceTransformer):
        def get_sentence_embedding_dimension(self):
            return 3

    with pytest.raises(RuntimeError, match="dimension không khớp"):
        EmbeddingService(settings, model_factory=WrongDimensionModel)
