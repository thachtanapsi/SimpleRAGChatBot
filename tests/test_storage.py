from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from rag_app.storage import SQLiteStorage, build_fts_query


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


def test_schema_v2_parent_child_rows_migrate_to_page_ranges(settings):
    settings.sqlite_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(settings.sqlite_path) as db:
        db.executescript(
            """
            CREATE TABLE parents (
                id TEXT PRIMARY KEY,
                document_id TEXT NOT NULL,
                page INTEGER NOT NULL,
                start_index INTEGER NOT NULL,
                text TEXT NOT NULL
            );
            CREATE TABLE children (
                id TEXT PRIMARY KEY,
                parent_id TEXT NOT NULL,
                document_id TEXT NOT NULL,
                page INTEGER NOT NULL,
                start_index INTEGER NOT NULL,
                text TEXT NOT NULL DEFAULT ''
            );
            INSERT INTO parents VALUES ('p1', 'doc1', 3, 0, 'parent');
            INSERT INTO children VALUES ('c1', 'p1', 'doc1', 3, 0, 'child');
            """
        )

    storage = SQLiteStorage(settings.sqlite_path)
    storage.initialise()

    with sqlite3.connect(settings.sqlite_path) as db:
        parent = db.execute(
            "SELECT page, page_end, kind FROM parents WHERE id='p1'"
        ).fetchone()
        child = db.execute(
            "SELECT page, page_end, kind FROM children WHERE id='c1'"
        ).fetchone()
        version = db.execute(
            "SELECT value FROM app_meta WHERE key='schema_version'"
        ).fetchone()[0]
    assert parent == (3, 3, "page")
    assert child == (3, 3, "page")
    assert version == "7"


def test_digest_provenance_columns_migrate_without_replacing_legacy_rows(settings):
    settings.sqlite_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(settings.sqlite_path) as db:
        db.executescript(
            """
            CREATE TABLE documents (
                id TEXT PRIMARY KEY,
                filename TEXT NOT NULL,
                sha256 TEXT NOT NULL UNIQUE,
                stored_path TEXT NOT NULL,
                size_bytes INTEGER NOT NULL,
                page_count INTEGER,
                status TEXT NOT NULL,
                error TEXT,
                index_fingerprint TEXT,
                collection_name TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            INSERT INTO documents(
                id, filename, sha256, stored_path, size_bytes, page_count,
                status, created_at, updated_at
            ) VALUES (
                'legacy-pdf', 'legacy.pdf', 'legacy-sha', '/old/legacy.pdf',
                10, 1, 'ready', 'old', 'old'
            );
            CREATE TABLE digest_batches (
                id TEXT PRIMARY KEY,
                source TEXT NOT NULL,
                analysis_date TEXT NOT NULL,
                cutoff TEXT NOT NULL,
                target_count INTEGER NOT NULL DEFAULT 500,
                status TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                sealed_at TEXT
            );
            INSERT INTO digest_batches(
                id, source, analysis_date, cutoff, target_count, status,
                created_at, updated_at
            ) VALUES (
                'legacy-batch', 'tradingagents_daily_research', '2026-08-20',
                '2026-08-20T16:45:00+07:00', 1, 'sealed', 'old', 'old'
            );
            """
        )

    storage = SQLiteStorage(settings.sqlite_path)
    storage.initialise()
    migrated = storage.get_digest_batch("legacy-batch")
    assert migrated["research_mode"] == "legacy_unknown"
    assert migrated["data_provenance"] == "legacy_unknown"
    assert migrated["status"] == "sealed"
    legacy_pdf = storage.get_document("legacy-pdf")
    assert legacy_pdf["source_type"] == "pdf"
    assert legacy_pdf["research_mode"] == "legacy_unknown"
    assert legacy_pdf["data_provenance"] == "legacy_unknown"
    assert legacy_pdf["content_kind"] == "pdf"
    assert legacy_pdf["report_schema"] == "pdf"
    assert "expected_digest_hash" in legacy_pdf
    with sqlite3.connect(settings.sqlite_path) as db:
        version = db.execute(
            "SELECT value FROM app_meta WHERE key='digest_schema_version'"
        ).fetchone()[0]
    assert version == "6"


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
            {
                "id": "c1",
                "parent_id": "p1",
                "document_id": "doc_hash",
                "page": 1,
                "start_index": 0,
                "text": "Phát hiện gian lận bằng FFD-2024",
            }
        ],
        fingerprint=settings.index_fingerprint,
        collection_name=settings.collection_name,
    )
    assert storage.child_ids_for_document("doc_hash") == ["c1"]
    parent = storage.parents_by_ids(["p1"])["p1"]
    assert parent["page_end"] == 1
    assert parent["kind"] == "page"
    assert storage.fts_count() == 1
    assert storage.lexical_search("phat hien", k=5, document_ids=["doc_hash"])[0][
        "child_id"
    ] == "c1"
    assert storage.lexical_search(
        'FFD-2024" OR *', k=5, document_ids=["doc_hash"]
    )[0]["child_id"] == "c1"
    assert storage.lexical_search("FFD", k=5, document_ids=["other"]) == []
    restarted = SQLiteStorage(settings.sqlite_path)
    restarted.initialise()
    assert restarted.lexical_search("gian lan", k=5, document_ids=["doc_hash"])
    restarted.delete_document_rows("doc_hash")
    assert restarted.child_ids_for_document("doc_hash") == []
    assert restarted.fts_count() == 0


def test_fts_query_is_tokenised_deduplicated_and_bounded():
    unsafe = '" OR secret* (token)'
    query = build_fts_query(unsafe)
    assert query == '"or" OR "secret" OR "token"'
    assert len(build_fts_query(" ".join(f"t{i}" for i in range(40))).split(" OR ")) == 32


def test_history_is_pruned_to_limit(settings):
    storage = SQLiteStorage(settings.sqlite_path)
    storage.initialise()
    for number in range(6):
        storage.add_exchange("session", f"q{number}", f"a{number}", max_messages=4)
    messages = storage.all_messages("session")
    assert len(messages) == 4
    assert messages[0]["content"] == "q4"
