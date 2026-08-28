from __future__ import annotations

import pytest

from rag_app.digests import canonicalise_digest
from rag_app.errors import ConflictError, DocumentBusyError
from rag_app.runtime import Runtime
from rag_app.storage import SQLiteStorage


class FakeIndex:
    def __init__(self):
        self.deleted = []

    def delete(self, ids, collection_name=None):
        self.deleted.append((list(ids), collection_name))


class FakeLogger:
    def info(self, *_args, **_kwargs):
        pass


def create_document(storage, settings, document_id, sha):
    path = settings.uploads_dir / f"{document_id}.pdf"
    path.write_bytes(b"%PDF-test")
    storage.create_pending_document(
        document_id=document_id,
        filename="paper.pdf",
        sha256=sha,
        stored_path=path,
        size_bytes=9,
        page_count=1,
    )
    return path


@pytest.mark.asyncio
async def test_runtime_delete_removes_vector_rows_and_upload(settings):
    storage = SQLiteStorage(settings.sqlite_path)
    storage.initialise()
    path = create_document(storage, settings, "doc1", "hash1")
    storage.claim_next_pending()
    storage.finish_indexing(
        document_id="doc1",
        page_count=1,
        parents=[
            {"id": "p1", "document_id": "doc1", "page": 1, "start_index": 0, "text": "parent"}
        ],
        children=[
            {
                "id": "c1",
                "parent_id": "p1",
                "document_id": "doc1",
                "page": 1,
                "start_index": 0,
                "text": "child",
            }
        ],
        fingerprint=settings.index_fingerprint,
        collection_name=settings.collection_name,
    )
    runtime = Runtime(settings)
    runtime.storage = storage
    runtime.index = FakeIndex()
    runtime.logger = FakeLogger()

    await runtime.delete_document("doc1")

    assert runtime.index.deleted == [(["c1"], settings.collection_name)]
    assert storage.get_document("doc1") is None
    assert storage.fts_count() == 0
    assert not path.exists()


@pytest.mark.asyncio
async def test_runtime_rejects_delete_while_indexing(settings):
    storage = SQLiteStorage(settings.sqlite_path)
    storage.initialise()
    create_document(storage, settings, "doc1", "hash1")
    storage.claim_next_pending()
    runtime = Runtime(settings)
    runtime.storage = storage
    runtime.index = FakeIndex()
    runtime.logger = FakeLogger()

    with pytest.raises(DocumentBusyError):
        await runtime.delete_document("doc1")
    assert storage.get_document("doc1")["status"] == "indexing"


@pytest.mark.asyncio
async def test_runtime_rejects_digest_delete_without_mutating_retained_history(settings):
    storage = SQLiteStorage(settings.sqlite_path)
    storage.initialise()
    storage.create_digest_batch(
        batch_id="batch",
        source="tradingagents",
        analysis_date="2026-08-21",
        cutoff="2026-08-21T16:45:00+07:00",
        target_count=1,
    )
    digest = canonicalise_digest(
        {
            "ticker": "HPG",
            "analysis_date": "2026-08-21",
            "cutoff": "2026-08-21T16:45:00+07:00",
            "source_updated_at": "2026-08-21T17:00:00+07:00",
            "liquidity_rank": 1,
            "status": "complete",
            "sections": {
                "summary": "Tổng quan",
                "market": "Thị trường",
                "fundamentals": "Cơ bản",
                "news_events": "Tin tức",
                "sentiment": "Tâm lý",
                "macro": "Vĩ mô",
                "catalysts": "Hỗ trợ",
                "risks_unknowns": "Rủi ro",
                "next_session_watch": "Theo dõi",
            },
        },
        batch_source="tradingagents",
    )
    stored = storage.replace_digest_document(digest=digest, batch_id="batch")
    before = storage.get_document(stored["id"])
    runtime = Runtime(settings)
    runtime.storage = storage
    runtime.index = FakeIndex()
    runtime.logger = FakeLogger()

    with pytest.raises(ConflictError, match="giữ vĩnh viễn"):
        await runtime.delete_document(stored["id"])

    after = storage.get_document(stored["id"])
    assert after["status"] == before["status"] == "pending"
    assert storage.digest_sections(stored["id"])
    assert runtime.index.deleted == []


@pytest.mark.asyncio
async def test_force_reindex_durably_queues_vector_cleanup_and_document(settings):
    storage = SQLiteStorage(settings.sqlite_path)
    storage.initialise()
    path = create_document(storage, settings, "doc1", "hash1")
    storage.claim_next_pending()
    storage.finish_indexing(
        document_id="doc1",
        page_count=1,
        parents=[
            {
                "id": "p1",
                "document_id": "doc1",
                "page": 1,
                "page_end": 1,
                "kind": "page",
                "start_index": 0,
                "text": "parent",
            }
        ],
        children=[
            {
                "id": "c1",
                "parent_id": "p1",
                "document_id": "doc1",
                "page": 1,
                "page_end": 1,
                "kind": "page",
                "start_index": 0,
                "text": "child",
            }
        ],
        fingerprint=settings.index_fingerprint,
        collection_name=settings.collection_name,
    )
    runtime = Runtime(settings)
    runtime.storage = storage
    runtime.index = FakeIndex()
    runtime.logger = FakeLogger()

    result = await runtime.force_reindex()

    assert result == {"scheduled": ["doc1"], "skipped": []}
    assert runtime.index.deleted == []
    assert storage.get_document("doc1")["status"] == "pending"
    assert storage.child_ids_for_document("doc1") == []
    assert storage.fts_count() == 0
    assert storage.pending_vector_deletions("doc1") == [
        {"collection_name": settings.collection_name, "child_ids": ["c1"]}
    ]
    assert path.is_file()


@pytest.mark.asyncio
async def test_force_reindex_rejects_active_indexing_job(settings):
    storage = SQLiteStorage(settings.sqlite_path)
    storage.initialise()
    create_document(storage, settings, "doc1", "hash1")
    storage.claim_next_pending()
    runtime = Runtime(settings)
    runtime.storage = storage
    runtime.index = FakeIndex()
    runtime.logger = FakeLogger()

    with pytest.raises(DocumentBusyError, match="reindex"):
        await runtime.force_reindex()

    assert runtime.index.deleted == []
    assert storage.get_document("doc1")["status"] == "indexing"
