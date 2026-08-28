from __future__ import annotations

from types import SimpleNamespace

import pytest

import rag_app.runtime as runtime_module
from rag_app.runtime import Runtime


@pytest.mark.asyncio
async def test_missing_bge_cache_fails_startup(settings, monkeypatch):
    def missing_embeddings(*_args, **_kwargs):
        raise OSError("model not found in local cache")

    monkeypatch.setattr(runtime_module, "EmbeddingService", missing_embeddings)
    runtime = Runtime(settings)
    with pytest.raises(OSError, match="local cache"):
        await runtime.start()


class FakeResponse:
    def raise_for_status(self):
        pass

    def json(self):
        return {"models": [{"name": "another-model:latest"}]}


class FakeAsyncClient:
    def __init__(self, **_kwargs):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_args):
        pass

    async def get(self, _url):
        return FakeResponse()


@pytest.mark.asyncio
async def test_ollama_missing_model_is_reported_without_crash(settings, monkeypatch):
    monkeypatch.setattr(runtime_module.httpx, "AsyncClient", FakeAsyncClient)
    runtime = Runtime(settings)
    ready = await runtime.check_ollama()
    assert ready is False
    assert runtime.ollama_error == "model_missing"


class HealthStorage:
    def healthcheck(self):
        return True

    def fts_healthcheck(self):
        return True

    def fts_count(self):
        return 2

    def full_report_payload_health(self):
        return {
            "full_reports": 0,
            "missing_ready_payloads": 0,
            "pending_ready_decision_indexes": 0,
            "partial_decisions": 0,
        }


class HealthIndex:
    def healthcheck(self):
        return True

    def count(self):
        return 2


@pytest.mark.asyncio
async def test_health_reports_embedding_and_fts_state(settings):
    runtime = Runtime(settings)
    runtime.storage = HealthStorage()
    runtime.index = HealthIndex()
    runtime.embeddings = SimpleNamespace(
        revision="commit",
        device="cpu",
        dimension=1024,
        truncated_texts=0,
    )
    runtime.ollama_ready = True

    health = await runtime.health()

    assert health["status"] == "ok"
    assert health["fts_index_ready"] is True
    assert health["full_report_payloads_ready"] is True
    assert health["embedding_device"] == "cpu"
    assert health["embedding_dimension"] == 1024
