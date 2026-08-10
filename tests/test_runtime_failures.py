from __future__ import annotations

import pytest

import rag_app.runtime as runtime_module
from rag_app.runtime import Runtime


@pytest.mark.asyncio
async def test_missing_bge_cache_fails_startup(settings, monkeypatch):
    def missing_embeddings(**_kwargs):
        raise OSError("model not found in local cache")

    monkeypatch.setattr(runtime_module, "HuggingFaceEmbeddings", missing_embeddings)
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

