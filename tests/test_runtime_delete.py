from __future__ import annotations

import pytest

from rag_app.errors import DocumentBusyError
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
async def test_force_reindex_deletes_derived_index_and_requeues_document(settings):
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
    assert runtime.index.deleted == [(["c1"], settings.collection_name)]
    assert storage.get_document("doc1")["status"] == "pending"
    assert storage.child_ids_for_document("doc1") == []
    assert storage.fts_count() == 0
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
