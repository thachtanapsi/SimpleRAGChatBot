from __future__ import annotations

from pathlib import Path

import pytest

from rag_app.storage import SQLiteStorage


def add_document(storage: SQLiteStorage, tmp_path: Path, doc_id="doc_hash", sha="hash"):
    path = tmp_path / f"{doc_id}.pdf"
    path.write_bytes(b"%PDF-dummy")
    return storage.create_pending_document(
        document_id=doc_id,
        filename="paper.pdf",
        sha256=sha,
        stored_path=path,
        size_bytes=10,
        page_count=1,
    )


def test_duplicate_hash_and_restart_persist(settings, tmp_path):
    storage = SQLiteStorage(settings.sqlite_path)
    storage.initialise()
    add_document(storage, tmp_path)
    with pytest.raises(Exception):
        add_document(storage, tmp_path, doc_id="other", sha="hash")

    restarted = SQLiteStorage(settings.sqlite_path)
    restarted.initialise()
    assert restarted.get_document("doc_hash")["filename"] == "paper.pdf"


def test_interrupted_job_returns_to_pending(settings, tmp_path):
    storage = SQLiteStorage(settings.sqlite_path)
    storage.initialise()
    add_document(storage, tmp_path)
    assert storage.claim_next_pending()["status"] == "indexing"
    assert storage.reset_interrupted_jobs() == 1
    assert storage.get_document("doc_hash")["status"] == "pending"


def test_pending_document_locked_for_delete_cannot_be_claimed(settings, tmp_path):
    storage = SQLiteStorage(settings.sqlite_path)
    storage.initialise()
    add_document(storage, tmp_path)
    locked = storage.lock_document_for_deletion("doc_hash")
    assert locked["status"] == "pending"
    assert storage.claim_next_pending() is None


def test_parent_child_commit_and_delete_cascade(settings, tmp_path):
    storage = SQLiteStorage(settings.sqlite_path)
    storage.initialise()
    add_document(storage, tmp_path)
    storage.claim_next_pending()
    storage.finish_indexing(
        document_id="doc_hash",
        page_count=1,
        parents=[
            {"id": "p1", "document_id": "doc_hash", "page": 1, "start_index": 0, "text": "parent"}
        ],
        children=[
            {"id": "c1", "parent_id": "p1", "document_id": "doc_hash", "page": 1, "start_index": 0}
        ],
        fingerprint=settings.index_fingerprint,
        collection_name=settings.collection_name,
    )
    assert storage.child_ids_for_document("doc_hash") == ["c1"]
    storage.delete_document_rows("doc_hash")
    assert storage.child_ids_for_document("doc_hash") == []


def test_history_is_pruned_to_limit(settings):
    storage = SQLiteStorage(settings.sqlite_path)
    storage.initialise()
    for number in range(6):
        storage.add_exchange("session", f"q{number}", f"a{number}", max_messages=4)
    messages = storage.all_messages("session")
    assert len(messages) == 4
    assert messages[0]["content"] == "q4"
