"""SQLite lưu metadata, parent-child mapping, job và hội thoại."""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


class SQLiteStorage:
    def __init__(self, path: Path):
        self.path = path

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA busy_timeout = 30000")
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def initialise(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as db:
            db.execute("PRAGMA journal_mode = WAL")
            db.executescript(
                """
                CREATE TABLE IF NOT EXISTS documents (
                    id TEXT PRIMARY KEY,
                    filename TEXT NOT NULL,
                    sha256 TEXT NOT NULL UNIQUE,
                    stored_path TEXT NOT NULL,
                    size_bytes INTEGER NOT NULL,
                    page_count INTEGER,
                    status TEXT NOT NULL CHECK (
                        status IN ('pending', 'indexing', 'ready', 'failed')
                    ),
                    error TEXT,
                    index_fingerprint TEXT,
                    collection_name TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_documents_status
                    ON documents(status, created_at);

                CREATE TABLE IF NOT EXISTS parents (
                    id TEXT PRIMARY KEY,
                    document_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
                    page INTEGER NOT NULL,
                    start_index INTEGER NOT NULL,
                    text TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_parents_document
                    ON parents(document_id);

                CREATE TABLE IF NOT EXISTS children (
                    id TEXT PRIMARY KEY,
                    parent_id TEXT NOT NULL REFERENCES parents(id) ON DELETE CASCADE,
                    document_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
                    page INTEGER NOT NULL,
                    start_index INTEGER NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_children_document
                    ON children(document_id);
                CREATE INDEX IF NOT EXISTS idx_children_parent
                    ON children(parent_id);

                CREATE TABLE IF NOT EXISTS sessions (
                    id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
                    role TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
                    content TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_messages_session
                    ON messages(session_id, id);

                CREATE TABLE IF NOT EXISTS app_meta (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );
                """
            )

    def healthcheck(self) -> bool:
        with self.connect() as db:
            return db.execute("SELECT 1").fetchone()[0] == 1

    def reset_interrupted_jobs(self) -> int:
        with self.connect() as db:
            cursor = db.execute(
                "UPDATE documents SET status='pending', error=NULL, updated_at=? "
                "WHERE status='indexing'",
                (utc_now(),),
            )
            return cursor.rowcount

    def create_pending_document(
        self,
        *,
        document_id: str,
        filename: str,
        sha256: str,
        stored_path: Path,
        size_bytes: int,
        page_count: int,
    ) -> dict[str, Any]:
        now = utc_now()
        with self.connect() as db:
            db.execute(
                """
                INSERT INTO documents(
                    id, filename, sha256, stored_path, size_bytes, page_count,
                    status, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, 'pending', ?, ?)
                """,
                (
                    document_id,
                    filename,
                    sha256,
                    str(stored_path),
                    size_bytes,
                    page_count,
                    now,
                    now,
                ),
            )
        return self.get_document(document_id)  # type: ignore[return-value]

    def get_document_by_hash(self, sha256: str) -> dict[str, Any] | None:
        with self.connect() as db:
            row = db.execute(
                "SELECT * FROM documents WHERE sha256=?", (sha256,)
            ).fetchone()
            return dict(row) if row else None

    def get_document(self, document_id: str) -> dict[str, Any] | None:
        with self.connect() as db:
            row = db.execute(
                "SELECT * FROM documents WHERE id=?", (document_id,)
            ).fetchone()
            return dict(row) if row else None

    def list_documents(self) -> list[dict[str, Any]]:
        with self.connect() as db:
            rows = db.execute(
                "SELECT * FROM documents ORDER BY created_at DESC"
            ).fetchall()
            return [dict(row) for row in rows]

    def list_ready_documents(self) -> list[dict[str, Any]]:
        with self.connect() as db:
            rows = db.execute(
                "SELECT * FROM documents WHERE status='ready' ORDER BY created_at"
            ).fetchall()
            return [dict(row) for row in rows]

    def claim_next_pending(self) -> dict[str, Any] | None:
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute(
                "SELECT * FROM documents WHERE status='pending' "
                "ORDER BY created_at LIMIT 1"
            ).fetchone()
            if row is None:
                return None
            changed = db.execute(
                "UPDATE documents SET status='indexing', error=NULL, updated_at=? "
                "WHERE id=? AND status='pending'",
                (utc_now(), row["id"]),
            ).rowcount
            if changed != 1:
                return None
            claimed = dict(row)
            claimed["status"] = "indexing"
            return claimed

    def finish_indexing(
        self,
        *,
        document_id: str,
        page_count: int,
        parents: Sequence[dict[str, Any]],
        children: Sequence[dict[str, Any]],
        fingerprint: str,
        collection_name: str,
    ) -> None:
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            db.execute("DELETE FROM children WHERE document_id=?", (document_id,))
            db.execute("DELETE FROM parents WHERE document_id=?", (document_id,))
            db.executemany(
                "INSERT INTO parents(id, document_id, page, start_index, text) "
                "VALUES (:id, :document_id, :page, :start_index, :text)",
                parents,
            )
            db.executemany(
                "INSERT INTO children(id, parent_id, document_id, page, start_index) "
                "VALUES (:id, :parent_id, :document_id, :page, :start_index)",
                children,
            )
            changed = db.execute(
                """
                UPDATE documents
                SET status='ready', page_count=?, error=NULL,
                    index_fingerprint=?, collection_name=?, updated_at=?
                WHERE id=? AND status='indexing'
                """,
                (
                    page_count,
                    fingerprint,
                    collection_name,
                    utc_now(),
                    document_id,
                ),
            ).rowcount
            if changed != 1:
                raise RuntimeError("Job indexing không còn ở trạng thái hợp lệ")

    def mark_failed(self, document_id: str, error_code: str) -> None:
        with self.connect() as db:
            db.execute(
                "UPDATE documents SET status='failed', error=?, updated_at=? WHERE id=?",
                (error_code[:200], utc_now(), document_id),
            )

    def documents_requiring_reindex(self, fingerprint: str) -> list[dict[str, Any]]:
        with self.connect() as db:
            rows = db.execute(
                "SELECT * FROM documents WHERE status='ready' "
                "AND COALESCE(index_fingerprint, '') <> ?",
                (fingerprint,),
            ).fetchall()
            return [dict(row) for row in rows]

    def requeue_document(self, document_id: str) -> None:
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            db.execute("DELETE FROM children WHERE document_id=?", (document_id,))
            db.execute("DELETE FROM parents WHERE document_id=?", (document_id,))
            db.execute(
                """
                UPDATE documents SET status='pending', error=NULL,
                    index_fingerprint=NULL, collection_name=NULL, updated_at=?
                WHERE id=?
                """,
                (utc_now(), document_id),
            )

    def child_ids_for_document(self, document_id: str) -> list[str]:
        with self.connect() as db:
            rows = db.execute(
                "SELECT id FROM children WHERE document_id=?", (document_id,)
            ).fetchall()
            return [row["id"] for row in rows]

    def parents_by_ids(self, parent_ids: Sequence[str]) -> dict[str, dict[str, Any]]:
        if not parent_ids:
            return {}
        placeholders = ",".join("?" for _ in parent_ids)
        with self.connect() as db:
            rows = db.execute(
                f"""
                SELECT p.*, d.filename
                FROM parents p JOIN documents d ON d.id=p.document_id
                WHERE p.id IN ({placeholders}) AND d.status='ready'
                """,
                tuple(parent_ids),
            ).fetchall()
            return {row["id"]: dict(row) for row in rows}

    def delete_document_rows(self, document_id: str) -> None:
        with self.connect() as db:
            db.execute("DELETE FROM documents WHERE id=?", (document_id,))

    def lock_document_for_deletion(self, document_id: str) -> dict[str, Any] | None:
        """Ngăn worker claim một job pending trong lúc API đang xóa.

        Trạng thái ``failed`` được dùng làm khóa chuyển tiếp hợp lệ trong schema.
        Nếu xóa Chroma thất bại, client có thể retry DELETE an toàn.
        """

        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute(
                "SELECT * FROM documents WHERE id=?", (document_id,)
            ).fetchone()
            if row is None:
                return None
            document = dict(row)
            if document["status"] != "indexing":
                db.execute(
                    "UPDATE documents SET status='failed', error='deleting', updated_at=? "
                    "WHERE id=?",
                    (utc_now(), document_id),
                )
            return document

    def ready_document_ids(self, requested: Sequence[str] | None = None) -> list[str]:
        with self.connect() as db:
            if requested is None:
                rows = db.execute(
                    "SELECT id FROM documents WHERE status='ready'"
                ).fetchall()
                return [row["id"] for row in rows]
            if not requested:
                return []
            placeholders = ",".join("?" for _ in requested)
            rows = db.execute(
                f"SELECT id FROM documents WHERE status='ready' "
                f"AND id IN ({placeholders})",
                tuple(requested),
            ).fetchall()
            return [row["id"] for row in rows]

    def ensure_session(self, session_id: str) -> None:
        now = utc_now()
        with self.connect() as db:
            db.execute(
                "INSERT OR IGNORE INTO sessions(id, created_at, updated_at) VALUES (?, ?, ?)",
                (session_id, now, now),
            )

    def session_exists(self, session_id: str) -> bool:
        with self.connect() as db:
            return (
                db.execute("SELECT 1 FROM sessions WHERE id=?", (session_id,)).fetchone()
                is not None
            )

    def recent_messages(self, session_id: str, limit: int) -> list[dict[str, Any]]:
        with self.connect() as db:
            rows = db.execute(
                """
                SELECT * FROM (
                    SELECT id, role, content, created_at FROM messages
                    WHERE session_id=? ORDER BY id DESC LIMIT ?
                ) ORDER BY id
                """,
                (session_id, limit),
            ).fetchall()
            return [dict(row) for row in rows]

    def add_exchange(
        self,
        session_id: str,
        question: str,
        answer: str,
        max_messages: int,
    ) -> None:
        now = utc_now()
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            db.execute(
                "INSERT OR IGNORE INTO sessions(id, created_at, updated_at) VALUES (?, ?, ?)",
                (session_id, now, now),
            )
            db.executemany(
                "INSERT INTO messages(session_id, role, content, created_at) VALUES (?, ?, ?, ?)",
                (
                    (session_id, "user", question, now),
                    (session_id, "assistant", answer, now),
                ),
            )
            db.execute(
                "UPDATE sessions SET updated_at=? WHERE id=?", (now, session_id)
            )
            db.execute(
                """
                DELETE FROM messages
                WHERE session_id=? AND id NOT IN (
                    SELECT id FROM messages WHERE session_id=?
                    ORDER BY id DESC LIMIT ?
                )
                """,
                (session_id, session_id, max_messages),
            )

    def all_messages(self, session_id: str) -> list[dict[str, Any]] | None:
        if not self.session_exists(session_id):
            return None
        with self.connect() as db:
            rows = db.execute(
                "SELECT id, role, content, created_at FROM messages "
                "WHERE session_id=? ORDER BY id",
                (session_id,),
            ).fetchall()
            return [dict(row) for row in rows]

    def delete_session(self, session_id: str) -> bool:
        with self.connect() as db:
            return db.execute("DELETE FROM sessions WHERE id=?", (session_id,)).rowcount == 1

    def get_meta(self, key: str) -> str | None:
        with self.connect() as db:
            row = db.execute("SELECT value FROM app_meta WHERE key=?", (key,)).fetchone()
            return row["value"] if row else None

    def set_meta(self, key: str, value: str) -> None:
        with self.connect() as db:
            db.execute(
                "INSERT INTO app_meta(key, value) VALUES (?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                (key, value),
            )
