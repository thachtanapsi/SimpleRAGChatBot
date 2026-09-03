from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from rag_app.errors import ServiceUnavailableError
from rag_app.reranking import (
    RerankCandidate,
    RerankerService,
    RerankerUnavailableError,
    resolve_reranker_device,
)


class FakeCrossEncoder:
    def __init__(self, model_name, **kwargs):
        self.model_name = model_name
        self.kwargs = kwargs
        self.model = SimpleNamespace(
            config=SimpleNamespace(_commit_hash="resolved-reranker-revision")
        )
        self.calls = []

    def predict(self, pairs, **kwargs):
        self.calls.append((pairs, kwargs))
        return [0.2, 0.91, 0.7]


def test_reranker_loads_offline_and_returns_sigmoid_ranked_results(settings):
    configured = replace(
        settings,
        rerank_device="cpu",
        rerank_batch_size=8,
        rerank_min_score=0.5,
    )
    service = RerankerService(configured, model_factory=FakeCrossEncoder)

    assert service.ready is True
    assert service.error is None
    assert service.revision == "resolved-reranker-revision"
    assert service.model.kwargs["local_files_only"] is True
    assert service.model.kwargs["max_length"] == 512
    assert isinstance(service.model.kwargs["activation_fn"], torch.nn.Sigmoid)

    results = service.rerank(
        "query",
        [
            RerankCandidate("a", "first"),
            RerankCandidate("b", "second"),
            RerankCandidate("c", "third"),
        ],
        top_k=2,
        min_score=0.5,
    )

    assert [(item.id, item.score) for item in results] == [("b", 0.91), ("c", 0.7)]
    pairs, options = service.model.calls[0]
    assert pairs == [("query", "first"), ("query", "second"), ("query", "third")]
    assert options["batch_size"] == 8
    assert options["show_progress_bar"] is False


def test_missing_local_reranker_is_health_degraded_and_request_fails_503(settings):
    def missing_model(*_args, **_kwargs):
        raise OSError("not cached")

    service = RerankerService(settings, model_factory=missing_model)

    assert service.available is False
    assert service.error == "OSError"
    assert service.health()["ready"] is False
    with pytest.raises(RerankerUnavailableError) as caught:
        service.rerank("query", [RerankCandidate("a", "text")])
    assert isinstance(caught.value, ServiceUnavailableError)
    assert caught.value.error_code == "rag_reranker_unavailable"
    with pytest.raises(RerankerUnavailableError):
        service.rerank("query", [])


def test_reranker_rejects_invalid_model_output_without_silent_fallback(settings):
    class InvalidCrossEncoder(FakeCrossEncoder):
        def predict(self, pairs, **kwargs):
            return [float("nan") for _ in pairs]

    service = RerankerService(
        replace(settings, rerank_device="cpu"),
        model_factory=InvalidCrossEncoder,
    )

    with pytest.raises(RerankerUnavailableError):
        service.rerank("query", [RerankCandidate("a", "text")])


def test_reranker_auto_device_is_local_cpu_when_mps_is_unavailable(monkeypatch):
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    assert resolve_reranker_device("auto") == "cpu"
