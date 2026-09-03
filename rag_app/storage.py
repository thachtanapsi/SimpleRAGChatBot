"""SQLite lưu metadata, parent-child mapping, job và hội thoại."""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import unicodedata
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

from .digests import digest_content_sha256, digest_document_sha256
from .errors import ConflictError, DocumentBusyError, PromotionConflictError
from .full_reports import (
    DECISION_STRUCTURED_SECTION,
    FULL_REPORT_V1,
    FULL_SECTION_ORDER,
)


FTS_MAX_QUERY_TOKENS = 32
FTS_TOKEN_RE = re.compile(r"\w+", flags=re.UNICODE)
LEGACY_PROVENANCE = "legacy_unknown"
VALID_PROVENANCE_PAIRS = {
    ("eod", "eod_cutoff"),
    ("live", "request_cutoff"),
    ("historical", "historical_replay"),
}


def normalise_fts_text(text: str) -> str:
    """Bỏ dấu cho lexical index; xử lý riêng đ/Đ mà Unicode NFD không tách."""

    folded = text.casefold().replace("đ", "d")
    return "".join(
        character
        for character in unicodedata.normalize("NFD", folded)
        if unicodedata.category(character) != "Mn"
    )


def build_fts_query(text: str) -> str:
    """Tạo MATCH expression chỉ từ token, không nhận cú pháp FTS của người dùng."""

    unique_tokens: list[str] = []
    seen: set[str] = set()
    for token in FTS_TOKEN_RE.findall(normalise_fts_text(text)):
        if token not in seen:
            unique_tokens.append(token)
            seen.add(token)
        if len(unique_tokens) == FTS_MAX_QUERY_TOKENS:
            break
    return " OR ".join(f'"{token}"' for token in unique_tokens)


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _stable_storage_id(prefix: str, *parts: object) -> str:
    encoded = json.dumps(
        parts, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return f"{prefix}_{hashlib.sha256(encoded).hexdigest()}"


def _document_filter_sql(
    filters: dict[str, Any] | None, *, alias: str = "d"
) -> tuple[list[str], list[Any]]:
    """Build a parameterised filter shared by catalog and retrieval queries."""

    if not filters:
        return [], []
    clauses: list[str] = []
    params: list[Any] = []
    for field, column in (
        ("source_types", "source_type"),
        ("tickers", "ticker"),
        ("batch_ids", "batch_id"),
        ("research_modes", "research_mode"),
        ("data_provenances", "data_provenance"),
        ("content_kinds", "content_kind"),
    ):
        values = list(dict.fromkeys(filters.get(field) or []))
        if values:
            placeholders = ",".join("?" for _ in values)
            clauses.append(f"{alias}.{column} IN ({placeholders})")
            params.extend(values)
    if filters.get("date_from"):
        clauses.append(f"{alias}.analysis_date >= ?")
        params.append(filters["date_from"])
    if filters.get("date_to"):
        clauses.append(f"{alias}.analysis_date <= ?")
        params.append(filters["date_to"])
    if filters.get("liquidity_rank_min") is not None:
        clauses.append(f"{alias}.liquidity_rank >= ?")
        params.append(int(filters["liquidity_rank_min"]))
    if filters.get("liquidity_rank_max") is not None:
        clauses.append(f"{alias}.liquidity_rank <= ?")
        params.append(int(filters["liquidity_rank_max"]))
    sections = list(dict.fromkeys(filters.get("digest_sections") or []))
    if sections:
        section_clauses: list[str] = []
        stored_sections = [
            section for section in sections if section != DECISION_STRUCTURED_SECTION
        ]
        if stored_sections:
            placeholders = ",".join("?" for _ in stored_sections)
            section_clauses.append(
                "EXISTS (SELECT 1 FROM digest_sections ds "
                f"WHERE ds.document_id={alias}.id "
                f"AND ds.section IN ({placeholders}) AND ds.text <> '')"
            )
            params.extend(stored_sections)
        if DECISION_STRUCTURED_SECTION in sections:
            section_clauses.append(
                "EXISTS (SELECT 1 FROM full_report_payloads frp "
                f"WHERE frp.document_id={alias}.id AND frp.decision_json <> '')"
            )
        clauses.append(f"({' OR '.join(section_clauses)})")
    stages = list(dict.fromkeys(filters.get("report_stages") or []))
    if stages:
        placeholders = ",".join("?" for _ in stages)
        clauses.append(
            "EXISTS (SELECT 1 FROM digest_sections rs "
            f"WHERE rs.document_id={alias}.id AND rs.report_stage IN ({placeholders}) "
            "AND rs.text <> '')"
        )
        params.extend(stages)
    return clauses, params


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
                    source_type TEXT NOT NULL DEFAULT 'pdf',
                    source_key TEXT,
                    content_sha256 TEXT,
                    ticker TEXT,
                    analysis_date TEXT,
                    analysis_cutoff TEXT,
                    source_run_id TEXT,
                    source_updated_at TEXT,
                    batch_id TEXT,
                    liquidity_rank INTEGER,
                    prompt_fingerprint TEXT,
                    model_fingerprint TEXT,
                    evidence_fingerprint TEXT,
                    digest_status TEXT,
                    research_mode TEXT NOT NULL DEFAULT 'legacy_unknown',
                    data_provenance TEXT NOT NULL DEFAULT 'legacy_unknown',
                    content_kind TEXT NOT NULL DEFAULT 'legacy_unknown',
                    report_schema TEXT NOT NULL DEFAULT 'legacy_unknown',
                    identity_hash TEXT,
                    digest_hash TEXT,
                    parent_identity_hash TEXT,
                    expected_digest_hash TEXT,
                    full_report_hash TEXT,
                    pipeline_version TEXT,
                    execution_generation INTEGER,
                    contains_decision_content INTEGER NOT NULL DEFAULT 0,
                    stage_status_json TEXT,
                    stage_fingerprints_json TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_documents_status
                    ON documents(status, created_at);

                CREATE TABLE IF NOT EXISTS parents (
                    id TEXT PRIMARY KEY,
                    document_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
                    page INTEGER NOT NULL,
                    page_end INTEGER NOT NULL,
                    kind TEXT NOT NULL CHECK (kind IN ('page', 'bridge')),
                    digest_section TEXT,
                    report_stage TEXT,
                    section_status TEXT,
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
                    page_end INTEGER NOT NULL,
                    kind TEXT NOT NULL CHECK (kind IN ('page', 'bridge')),
                    digest_section TEXT,
                    report_stage TEXT,
                    section_status TEXT,
                    start_index INTEGER NOT NULL,
                    text TEXT NOT NULL DEFAULT ''
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

                CREATE TABLE IF NOT EXISTS digest_batches (
                    id TEXT PRIMARY KEY,
                    source TEXT NOT NULL,
                    analysis_date TEXT NOT NULL,
                    cutoff TEXT NOT NULL,
                    target_count INTEGER NOT NULL DEFAULT 500,
                    research_mode TEXT NOT NULL DEFAULT 'legacy_unknown',
                    data_provenance TEXT NOT NULL DEFAULT 'legacy_unknown',
                    status TEXT NOT NULL CHECK (status IN ('open', 'sealed')),
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    sealed_at TEXT
                );

                CREATE TABLE IF NOT EXISTS digest_sections (
                    document_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
                    section TEXT NOT NULL,
                    position INTEGER NOT NULL,
                    report_stage TEXT NOT NULL DEFAULT 'digest',
                    section_status TEXT NOT NULL DEFAULT 'complete',
                    text TEXT NOT NULL,
                    PRIMARY KEY(document_id, section)
                );

                CREATE TABLE IF NOT EXISTS full_report_payloads (
                    document_id TEXT PRIMARY KEY
                        REFERENCES documents(id) ON DELETE CASCADE,
                    canonical_json TEXT NOT NULL,
                    canonical_sha256 TEXT NOT NULL,
                    decision_json TEXT NOT NULL,
                    decision_sha256 TEXT NOT NULL,
                    decision_status TEXT NOT NULL CHECK (
                        decision_status IN ('complete', 'partial')
                    ),
                    projection_version INTEGER NOT NULL,
                    decision_index_version INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_full_report_payload_status
                    ON full_report_payloads(decision_status, projection_version);

                CREATE TABLE IF NOT EXISTS pending_vector_deletions (
                    document_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
                    collection_name TEXT NOT NULL,
                    child_id TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    PRIMARY KEY(document_id, collection_name, child_id)
                );

                CREATE INDEX IF NOT EXISTS idx_pending_vector_deletions_document
                    ON pending_vector_deletions(document_id);

                CREATE TABLE IF NOT EXISTS pending_vector_metadata_updates (
                    document_id TEXT PRIMARY KEY REFERENCES documents(id) ON DELETE CASCADE,
                    research_mode TEXT NOT NULL,
                    data_provenance TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS digest_claims (
                    id TEXT PRIMARY KEY,
                    document_id TEXT NOT NULL
                        REFERENCES documents(id) ON DELETE CASCADE,
                    section TEXT NOT NULL,
                    position INTEGER NOT NULL,
                    text TEXT NOT NULL,
                    confidence_json TEXT,
                    evidence_ids_json TEXT NOT NULL DEFAULT '[]',
                    created_at TEXT NOT NULL,
                    UNIQUE(document_id, section, position)
                );

                CREATE INDEX IF NOT EXISTS idx_digest_claims_document
                    ON digest_claims(document_id, section, position);

                CREATE TABLE IF NOT EXISTS graph_builds (
                    id TEXT PRIMARY KEY,
                    document_id TEXT NOT NULL
                        REFERENCES documents(id) ON DELETE CASCADE,
                    content_sha256 TEXT NOT NULL,
                    index_fingerprint TEXT NOT NULL,
                    extractor_fingerprint TEXT NOT NULL,
                    status TEXT NOT NULL CHECK (
                        status IN ('pending', 'building', 'ready', 'failed', 'stale')
                    ),
                    active INTEGER NOT NULL DEFAULT 0 CHECK (active IN (0, 1)),
                    retry_count INTEGER NOT NULL DEFAULT 0,
                    error_code TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    started_at TEXT,
                    completed_at TEXT,
                    UNIQUE(
                        document_id, content_sha256,
                        index_fingerprint, extractor_fingerprint
                    )
                );

                CREATE INDEX IF NOT EXISTS idx_graph_builds_status
                    ON graph_builds(status, created_at);
                CREATE INDEX IF NOT EXISTS idx_graph_builds_document
                    ON graph_builds(document_id, status, active);
                CREATE UNIQUE INDEX IF NOT EXISTS idx_graph_builds_one_active
                    ON graph_builds(document_id) WHERE active=1;

                CREATE TABLE IF NOT EXISTS graph_entities (
                    id TEXT PRIMARY KEY,
                    entity_type TEXT NOT NULL,
                    canonical_key TEXT NOT NULL,
                    canonical_name TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    UNIQUE(entity_type, canonical_key)
                );

                CREATE INDEX IF NOT EXISTS idx_graph_entities_key
                    ON graph_entities(canonical_key, entity_type);

                CREATE TABLE IF NOT EXISTS graph_mentions (
                    id TEXT PRIMARY KEY,
                    build_id TEXT NOT NULL
                        REFERENCES graph_builds(id) ON DELETE CASCADE,
                    entity_id TEXT NOT NULL
                        REFERENCES graph_entities(id) ON DELETE CASCADE,
                    parent_id TEXT NOT NULL
                        REFERENCES parents(id) ON DELETE CASCADE,
                    surface TEXT NOT NULL,
                    alias_key TEXT NOT NULL,
                    start_offset INTEGER NOT NULL CHECK (start_offset >= 0),
                    end_offset INTEGER NOT NULL CHECK (end_offset >= start_offset),
                    created_at TEXT NOT NULL,
                    UNIQUE(build_id, parent_id, entity_id, start_offset, end_offset)
                );

                CREATE INDEX IF NOT EXISTS idx_graph_mentions_entity
                    ON graph_mentions(entity_id, alias_key);
                CREATE INDEX IF NOT EXISTS idx_graph_mentions_build
                    ON graph_mentions(build_id, parent_id);

                CREATE TABLE IF NOT EXISTS graph_assertions (
                    id TEXT PRIMARY KEY,
                    build_id TEXT NOT NULL
                        REFERENCES graph_builds(id) ON DELETE CASCADE,
                    document_id TEXT NOT NULL
                        REFERENCES documents(id) ON DELETE CASCADE,
                    subject_entity_id TEXT NOT NULL
                        REFERENCES graph_entities(id) ON DELETE CASCADE,
                    predicate TEXT NOT NULL,
                    object_entity_id TEXT
                        REFERENCES graph_entities(id) ON DELETE CASCADE,
                    object_literal TEXT,
                    object_type TEXT,
                    modality TEXT NOT NULL DEFAULT 'asserted',
                    valid_at TEXT,
                    confidence REAL NOT NULL DEFAULT 1.0
                        CHECK (confidence >= 0.0 AND confidence <= 1.0),
                    created_at TEXT NOT NULL,
                    CHECK (
                        (object_entity_id IS NOT NULL AND object_literal IS NULL)
                        OR
                        (object_entity_id IS NULL AND object_literal IS NOT NULL)
                    )
                );

                CREATE INDEX IF NOT EXISTS idx_graph_assertions_subject
                    ON graph_assertions(subject_entity_id, predicate, build_id);
                CREATE INDEX IF NOT EXISTS idx_graph_assertions_object
                    ON graph_assertions(object_entity_id, predicate, build_id)
                    WHERE object_entity_id IS NOT NULL;
                CREATE INDEX IF NOT EXISTS idx_graph_assertions_document
                    ON graph_assertions(document_id, build_id);

                CREATE TABLE IF NOT EXISTS graph_evidence (
                    id TEXT PRIMARY KEY,
                    build_id TEXT NOT NULL
                        REFERENCES graph_builds(id) ON DELETE CASCADE,
                    assertion_id TEXT NOT NULL
                        REFERENCES graph_assertions(id) ON DELETE CASCADE,
                    parent_id TEXT NOT NULL
                        REFERENCES parents(id) ON DELETE CASCADE,
                    child_id TEXT REFERENCES children(id) ON DELETE SET NULL,
                    quote TEXT NOT NULL,
                    quote_sha256 TEXT NOT NULL,
                    start_offset INTEGER NOT NULL CHECK (start_offset >= 0),
                    end_offset INTEGER NOT NULL CHECK (end_offset >= start_offset),
                    created_at TEXT NOT NULL,
                    UNIQUE(assertion_id, parent_id, start_offset, end_offset)
                );

                CREATE INDEX IF NOT EXISTS idx_graph_evidence_assertion
                    ON graph_evidence(assertion_id, parent_id);
                CREATE INDEX IF NOT EXISTS idx_graph_evidence_parent
                    ON graph_evidence(parent_id, build_id);
                """
            )
            document_columns = {
                row["name"] for row in db.execute("PRAGMA table_info(documents)")
            }
            document_migrations = {
                "source_type": "TEXT NOT NULL DEFAULT 'pdf'",
                "source_key": "TEXT",
                "content_sha256": "TEXT",
                "ticker": "TEXT",
                "analysis_date": "TEXT",
                "analysis_cutoff": "TEXT",
                "source_run_id": "TEXT",
                "source_updated_at": "TEXT",
                "batch_id": "TEXT",
                "liquidity_rank": "INTEGER",
                "prompt_fingerprint": "TEXT",
                "model_fingerprint": "TEXT",
                "evidence_fingerprint": "TEXT",
                "digest_status": "TEXT",
                "research_mode": "TEXT NOT NULL DEFAULT 'legacy_unknown'",
                "data_provenance": "TEXT NOT NULL DEFAULT 'legacy_unknown'",
                "content_kind": "TEXT NOT NULL DEFAULT 'legacy_unknown'",
                "report_schema": "TEXT NOT NULL DEFAULT 'legacy_unknown'",
                "identity_hash": "TEXT",
                "digest_hash": "TEXT",
                "parent_identity_hash": "TEXT",
                "expected_digest_hash": "TEXT",
                "full_report_hash": "TEXT",
                "pipeline_version": "TEXT",
                "execution_generation": "INTEGER",
                "contains_decision_content": "INTEGER NOT NULL DEFAULT 0",
                "stage_status_json": "TEXT",
                "stage_fingerprints_json": "TEXT",
            }
            for column, declaration in document_migrations.items():
                if column not in document_columns:
                    db.execute(
                        f"ALTER TABLE documents ADD COLUMN {column} {declaration}"
                    )
            db.execute(
                """
                UPDATE documents
                SET content_kind=CASE
                        WHEN source_type='pdf' THEN 'pdf'
                        ELSE 'daily_digest_v1'
                    END,
                    report_schema=CASE
                        WHEN source_type='pdf' THEN 'pdf'
                        ELSE 'daily_digest_v1'
                    END
                WHERE content_kind='legacy_unknown'
                   OR report_schema='legacy_unknown'
                """
            )
            batch_columns = {
                row["name"] for row in db.execute("PRAGMA table_info(digest_batches)")
            }
            batch_migrations = {
                "research_mode": "TEXT NOT NULL DEFAULT 'legacy_unknown'",
                "data_provenance": "TEXT NOT NULL DEFAULT 'legacy_unknown'",
            }
            for column, declaration in batch_migrations.items():
                if column not in batch_columns:
                    db.execute(
                        f"ALTER TABLE digest_batches ADD COLUMN {column} {declaration}"
                    )
            db.execute(
                "CREATE UNIQUE INDEX IF NOT EXISTS idx_documents_source_key "
                "ON documents(source_key) WHERE source_key IS NOT NULL"
            )
            db.execute(
                "CREATE INDEX IF NOT EXISTS idx_documents_digest_lookup "
                "ON documents(source_type, ticker, analysis_date, batch_id, status)"
            )
            db.execute(
                "CREATE UNIQUE INDEX IF NOT EXISTS idx_documents_batch_ticker "
                "ON documents(batch_id, ticker) "
                "WHERE source_type='trading_digest'"
            )
            db.execute(
                "CREATE UNIQUE INDEX IF NOT EXISTS idx_documents_batch_rank "
                "ON documents(batch_id, liquidity_rank) "
                "WHERE source_type='trading_digest' AND liquidity_rank IS NOT NULL"
            )
            db.execute(
                "CREATE INDEX IF NOT EXISTS idx_digest_batches_date "
                "ON digest_batches(analysis_date, cutoff)"
            )
            db.execute(
                "CREATE UNIQUE INDEX IF NOT EXISTS idx_digest_batches_identity "
                "ON digest_batches(source, analysis_date, cutoff)"
            )
            digest_section_columns = {
                row["name"] for row in db.execute("PRAGMA table_info(digest_sections)")
            }
            if "report_stage" not in digest_section_columns:
                db.execute(
                    "ALTER TABLE digest_sections ADD COLUMN "
                    "report_stage TEXT NOT NULL DEFAULT 'digest'"
                )
            if "section_status" not in digest_section_columns:
                db.execute(
                    "ALTER TABLE digest_sections ADD COLUMN "
                    "section_status TEXT NOT NULL DEFAULT 'complete'"
                )
            parent_columns = {
                row["name"] for row in db.execute("PRAGMA table_info(parents)")
            }
            if "page_end" not in parent_columns:
                db.execute("ALTER TABLE parents ADD COLUMN page_end INTEGER")
            if "kind" not in parent_columns:
                db.execute(
                    "ALTER TABLE parents ADD COLUMN kind TEXT NOT NULL DEFAULT 'page'"
                )
            if "digest_section" not in parent_columns:
                db.execute("ALTER TABLE parents ADD COLUMN digest_section TEXT")
            if "report_stage" not in parent_columns:
                db.execute("ALTER TABLE parents ADD COLUMN report_stage TEXT")
            if "section_status" not in parent_columns:
                db.execute("ALTER TABLE parents ADD COLUMN section_status TEXT")
            db.execute("UPDATE parents SET page_end=page WHERE page_end IS NULL")
            db.execute(
                "UPDATE parents SET report_stage='digest' "
                "WHERE digest_section IS NOT NULL AND report_stage IS NULL"
            )
            db.execute(
                "UPDATE parents SET section_status='complete' "
                "WHERE digest_section IS NOT NULL AND section_status IS NULL"
            )

            child_columns = {
                row["name"] for row in db.execute("PRAGMA table_info(children)")
            }
            if "text" not in child_columns:
                db.execute("ALTER TABLE children ADD COLUMN text TEXT NOT NULL DEFAULT ''")
            if "page_end" not in child_columns:
                db.execute("ALTER TABLE children ADD COLUMN page_end INTEGER")
            if "kind" not in child_columns:
                db.execute(
                    "ALTER TABLE children ADD COLUMN kind TEXT NOT NULL DEFAULT 'page'"
                )
            if "digest_section" not in child_columns:
                db.execute("ALTER TABLE children ADD COLUMN digest_section TEXT")
            if "report_stage" not in child_columns:
                db.execute("ALTER TABLE children ADD COLUMN report_stage TEXT")
            if "section_status" not in child_columns:
                db.execute("ALTER TABLE children ADD COLUMN section_status TEXT")
            db.execute("UPDATE children SET page_end=page WHERE page_end IS NULL")
            db.execute(
                "UPDATE children SET report_stage='digest' "
                "WHERE digest_section IS NOT NULL AND report_stage IS NULL"
            )
            db.execute(
                "UPDATE children SET section_status='complete' "
                "WHERE digest_section IS NOT NULL AND section_status IS NULL"
            )
            try:
                db.execute(
                    """
                    CREATE VIRTUAL TABLE IF NOT EXISTS child_fts USING fts5(
                        child_id UNINDEXED,
                        parent_id UNINDEXED,
                        document_id UNINDEXED,
                        text,
                        tokenize='unicode61 remove_diacritics 2'
                    )
                    """
                )
            except sqlite3.OperationalError as exc:
                raise RuntimeError("SQLite hiện tại không hỗ trợ FTS5") from exc
            try:
                db.execute(
                    """
                    CREATE VIRTUAL TABLE IF NOT EXISTS graph_entity_fts USING fts5(
                        entity_id UNINDEXED,
                        canonical_key,
                        canonical_name,
                        aliases,
                        tokenize='unicode61 remove_diacritics 2'
                    )
                    """
                )
            except sqlite3.OperationalError as exc:
                raise RuntimeError("SQLite hiện tại không hỗ trợ FTS5") from exc
            db.execute(
                "INSERT INTO app_meta(key, value) VALUES ('schema_version', '8') "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value"
            )
            db.execute(
                "INSERT INTO app_meta(key, value) VALUES ('digest_schema_version', '6') "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value"
            )
            db.execute(
                "INSERT INTO app_meta(key, value) VALUES ('graph_schema_version', '1') "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value"
            )

    @staticmethod
    def _stale_graph_builds(db: sqlite3.Connection, document_id: str) -> int:
        """Deactivate derived graph state in the caller's document transaction."""

        return db.execute(
            """
            UPDATE graph_builds
            SET status='stale', active=0, updated_at=?
            WHERE document_id=? AND status <> 'stale'
            """,
            (utc_now(), document_id),
        ).rowcount

    def healthcheck(self) -> bool:
        with self.connect() as db:
            return db.execute("SELECT 1").fetchone()[0] == 1

    def fts_healthcheck(self) -> bool:
        try:
            with self.connect() as db:
                db.execute("SELECT count(*) FROM child_fts").fetchone()
            return True
        except sqlite3.Error:
            return False

    def fts_count(self) -> int:
        with self.connect() as db:
            return int(db.execute("SELECT count(*) FROM child_fts").fetchone()[0])

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
                    status, content_kind, report_schema, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, 'pending', 'pdf', 'pdf', ?, ?)
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

    def list_ready_tickers(self) -> list[str]:
        """Return canonical tickers backed by ready documents in stable order."""

        with self.connect() as db:
            rows = db.execute(
                "SELECT ticker FROM documents "
                "WHERE status='ready' AND ticker IS NOT NULL"
            ).fetchall()
        return sorted(
            {
                canonical
                for row in rows
                if (canonical := str(row["ticker"]).strip().upper())
            }
        )

    def list_documents_page(
        self,
        *,
        page: int,
        page_size: int,
        filters: dict[str, Any] | None = None,
        status: str | None = None,
        ticker_query: str | None = None,
    ) -> dict[str, Any]:
        clauses, params = _document_filter_sql(filters, alias="d")
        if status:
            clauses.append("d.status=?")
            params.append(status)
        if ticker_query:
            clauses.append("(d.ticker LIKE ? OR d.filename LIKE ?)")
            pattern = f"%{ticker_query.strip().upper()}%"
            params.extend((pattern, pattern))
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        offset = (page - 1) * page_size
        with self.connect() as db:
            total = int(
                db.execute(
                    f"SELECT count(*) FROM documents d {where}", tuple(params)
                ).fetchone()[0]
            )
            rows = db.execute(
                f"SELECT d.* FROM documents d {where} "
                "ORDER BY COALESCE(d.analysis_date, d.created_at) DESC, "
                "COALESCE(d.liquidity_rank, 2147483647), d.filename "
                "LIMIT ? OFFSET ?",
                (*params, page_size, offset),
            ).fetchall()
        return {
            "items": [dict(row) for row in rows],
            "page": page,
            "page_size": page_size,
            "total": total,
        }

    def get_document_by_source_key(self, source_key: str) -> dict[str, Any] | None:
        with self.connect() as db:
            row = db.execute(
                "SELECT * FROM documents WHERE source_key=?", (source_key,)
            ).fetchone()
            return dict(row) if row else None

    def get_digest_batch_member(
        self, batch_id: str, *, ticker: str | None = None, liquidity_rank: int | None = None
    ) -> dict[str, Any] | None:
        if ticker is None and liquidity_rank is None:
            raise ValueError("ticker hoặc liquidity_rank là bắt buộc")
        column, value = (
            ("ticker", ticker) if ticker is not None else ("liquidity_rank", liquidity_rank)
        )
        with self.connect() as db:
            row = db.execute(
                f"SELECT * FROM documents WHERE source_type='trading_digest' "
                f"AND batch_id=? AND {column}=?",
                (batch_id, value),
            ).fetchone()
            return dict(row) if row else None

    def create_digest_batch(
        self,
        *,
        batch_id: str,
        source: str,
        analysis_date: str,
        cutoff: str,
        target_count: int,
        research_mode: str = "legacy_unknown",
        data_provenance: str = "legacy_unknown",
    ) -> tuple[str, dict[str, Any]]:
        now = utc_now()
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            existing = db.execute(
                "SELECT * FROM digest_batches WHERE id=?", (batch_id,)
            ).fetchone()
            if existing:
                batch = dict(existing)
                same_identity = (
                    batch["source"] == source
                    and batch["analysis_date"] == analysis_date
                    and batch["cutoff"] == cutoff
                    and int(batch["target_count"]) == target_count
                )
                current_provenance = (
                    str(batch["research_mode"]),
                    str(batch["data_provenance"]),
                )
                requested_provenance = (research_mode, data_provenance)
                if same_identity and current_provenance == requested_provenance:
                    return "unchanged", batch
                # Batches created by pre-provenance clients can be labelled once
                # when the immutable identity still matches.  Once labelled, a
                # retry may neither remove nor change that provenance pair.
                if (
                    same_identity
                    and current_provenance
                    == (LEGACY_PROVENANCE, LEGACY_PROVENANCE)
                    and requested_provenance in VALID_PROVENANCE_PAIRS
                ):
                    legacy_documents = db.execute(
                        """
                        SELECT * FROM documents
                        WHERE source_type='trading_digest' AND batch_id=?
                          AND research_mode=? AND data_provenance=?
                        """,
                        (batch_id, LEGACY_PROVENANCE, LEGACY_PROVENANCE),
                    ).fetchall()
                    db.execute(
                        "UPDATE digest_batches SET research_mode=?, "
                        "data_provenance=?, updated_at=? WHERE id=?",
                        (research_mode, data_provenance, now, batch_id),
                    )
                    for document in legacy_documents:
                        section_rows = db.execute(
                            "SELECT section, text FROM digest_sections "
                            "WHERE document_id=? ORDER BY position",
                            (document["id"],),
                        ).fetchall()
                        sections = {
                            str(section["section"]): str(section["text"])
                            for section in section_rows
                        }
                        content_sha256 = digest_content_sha256(
                            source=source,
                            ticker=str(document["ticker"]),
                            analysis_date=str(document["analysis_date"]),
                            cutoff=str(document["analysis_cutoff"]),
                            liquidity_rank=document["liquidity_rank"],
                            prompt_fingerprint=document["prompt_fingerprint"],
                            model_fingerprint=document["model_fingerprint"],
                            evidence_fingerprint=document["evidence_fingerprint"],
                            digest_status=str(document["digest_status"]),
                            research_mode=research_mode,
                            data_provenance=data_provenance,
                            sections=sections,
                        )
                        sha256 = digest_document_sha256(
                            str(document["source_key"]), content_sha256
                        )
                        db.execute(
                            """
                            UPDATE documents
                            SET content_sha256=?, sha256=?, research_mode=?,
                                data_provenance=?, updated_at=?
                            WHERE id=?
                            """,
                            (
                                content_sha256,
                                sha256,
                                research_mode,
                                data_provenance,
                                now,
                                document["id"],
                            ),
                        )
                        # A document-level outbox also covers a worker that was
                        # already indexing with its pre-upgrade snapshot.  Once
                        # its children become ready, metadata is reconciled in
                        # place without recomputing embeddings.
                        db.execute(
                            """
                            INSERT INTO pending_vector_metadata_updates(
                                document_id, research_mode, data_provenance,
                                created_at, updated_at
                            ) VALUES (?, ?, ?, ?, ?)
                            ON CONFLICT(document_id) DO UPDATE SET
                                research_mode=excluded.research_mode,
                                data_provenance=excluded.data_provenance,
                                updated_at=excluded.updated_at
                            """,
                            (
                                document["id"],
                                research_mode,
                                data_provenance,
                                now,
                                now,
                            ),
                        )
                    upgraded = db.execute(
                        "SELECT * FROM digest_batches WHERE id=?", (batch_id,)
                    ).fetchone()
                    assert upgraded is not None
                    return "upgraded", dict(upgraded)
                return "conflict", batch
            identity_owner = db.execute(
                "SELECT * FROM digest_batches WHERE source=? AND analysis_date=? AND cutoff=?",
                (source, analysis_date, cutoff),
            ).fetchone()
            if identity_owner:
                return "conflict", dict(identity_owner)
            db.execute(
                """
                INSERT INTO digest_batches(
                    id, source, analysis_date, cutoff, target_count, status,
                    research_mode, data_provenance, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, 'open', ?, ?, ?, ?)
                """,
                (
                    batch_id,
                    source,
                    analysis_date,
                    cutoff,
                    target_count,
                    research_mode,
                    data_provenance,
                    now,
                    now,
                ),
            )
        return "created", self.get_digest_batch(batch_id)  # type: ignore[return-value]

    def get_digest_batch(self, batch_id: str) -> dict[str, Any] | None:
        with self.connect() as db:
            row = db.execute(
                "SELECT * FROM digest_batches WHERE id=?", (batch_id,)
            ).fetchone()
            return dict(row) if row else None

    def latest_digest_batch(self) -> dict[str, Any] | None:
        with self.connect() as db:
            row = db.execute(
                "SELECT * FROM digest_batches "
                "ORDER BY analysis_date DESC, cutoff DESC, created_at DESC LIMIT 1"
            ).fetchone()
            return dict(row) if row else None

    def seal_digest_batch(self, batch_id: str) -> tuple[str, dict[str, Any] | None]:
        now = utc_now()
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            batch = db.execute(
                "SELECT * FROM digest_batches WHERE id=?", (batch_id,)
            ).fetchone()
            if not batch:
                return "missing", None
            counts = db.execute(
                """
                SELECT count(*) AS accepted,
                       count(DISTINCT ticker) AS unique_tickers,
                       count(liquidity_rank) AS ranked,
                       count(DISTINCT liquidity_rank) AS unique_ranks,
                       min(liquidity_rank) AS min_rank,
                       max(liquidity_rank) AS max_rank,
                       sum(liquidity_rank) AS rank_sum
                FROM documents
                WHERE source_type='trading_digest' AND batch_id=?
                """,
                (batch_id,),
            ).fetchone()
            target = int(batch["target_count"])
            if not (
                int(counts["accepted"]) == target
                and int(counts["unique_tickers"]) == target
                and int(counts["ranked"]) == target
                and int(counts["unique_ranks"]) == target
                and int(counts["min_rank"] or 0) == 1
                and int(counts["max_rank"] or 0) == target
                and int(counts["rank_sum"] or 0) == target * (target + 1) // 2
            ):
                return "incomplete", dict(batch)
            changed = db.execute(
                "UPDATE digest_batches SET status='sealed', sealed_at=COALESCE(sealed_at, ?), "
                "updated_at=? WHERE id=?",
                (now, now, batch_id),
            ).rowcount
            if changed != 1:
                return "missing", None
        return "sealed", self.get_digest_batch(batch_id)

    def digest_batch_status(self, batch_id: str) -> dict[str, Any] | None:
        batch = self.get_digest_batch(batch_id)
        if not batch:
            return None
        with self.connect() as db:
            rows = db.execute(
                "SELECT status, count(*) AS count FROM documents "
                "WHERE source_type='trading_digest' AND batch_id=? GROUP BY status",
                (batch_id,),
            ).fetchall()
            partial = int(
                db.execute(
                    "SELECT count(*) FROM documents WHERE source_type='trading_digest' "
                    "AND batch_id=? AND digest_status='partial'",
                    (batch_id,),
                ).fetchone()[0]
            )
        counts = {item: 0 for item in ("pending", "indexing", "ready", "failed")}
        counts.update({str(row["status"]): int(row["count"]) for row in rows})
        accepted = sum(counts.values())
        return {
            **batch,
            "accepted_count": accepted,
            "counts": counts,
            "partial_count": partial,
            "rag_ready": counts["ready"],
            "complete": accepted == int(batch["target_count"])
            and counts["ready"] == accepted,
        }

    def digest_batch_items(
        self,
        batch_id: str,
        *,
        page: int,
        page_size: int,
        status: str | None = None,
        ticker: str | None = None,
    ) -> dict[str, Any]:
        clauses = ["source_type='trading_digest'", "batch_id=?"]
        params: list[Any] = [batch_id]
        if status:
            clauses.append("status=?")
            params.append(status)
        if ticker:
            clauses.append("ticker=?")
            params.append(ticker.strip().upper())
        where = " AND ".join(clauses)
        offset = (page - 1) * page_size
        with self.connect() as db:
            total = int(
                db.execute(
                    f"SELECT count(*) FROM documents WHERE {where}", tuple(params)
                ).fetchone()[0]
            )
            rows = db.execute(
                f"""
                SELECT id AS document_id, ticker, liquidity_rank, status, error,
                       digest_status, research_mode, data_provenance,
                       source_updated_at, updated_at
                FROM documents
                WHERE {where}
                ORDER BY liquidity_rank, ticker
                LIMIT ? OFFSET ?
                """,
                (*params, page_size, offset),
            ).fetchall()
        return {
            "items": [dict(row) for row in rows],
            "items_page": page,
            "items_page_size": page_size,
            "items_total": total,
        }

    @staticmethod
    def _upsert_full_report_payload(
        db: sqlite3.Connection,
        *,
        document_id: str,
        digest: Any,
        now: str,
    ) -> bool:
        """Persist the canonical report and projection in the caller transaction."""

        if getattr(digest, "content_kind", None) != FULL_REPORT_V1:
            return False
        existing = db.execute(
            "SELECT decision_sha256, projection_version, decision_index_version "
            "FROM full_report_payloads WHERE document_id=?",
            (document_id,),
        ).fetchone()
        needs_reindex = existing is None or (
            str(existing["decision_sha256"]) != digest.decision_sha256
            or int(existing["projection_version"]) != digest.projection_version
            or int(existing["decision_index_version"]) != digest.projection_version
        )
        db.execute(
            """
            INSERT INTO full_report_payloads(
                document_id, canonical_json, canonical_sha256,
                decision_json, decision_sha256, decision_status,
                projection_version, decision_index_version,
                created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, 0, ?, ?)
            ON CONFLICT(document_id) DO UPDATE SET
                canonical_json=excluded.canonical_json,
                canonical_sha256=excluded.canonical_sha256,
                decision_json=excluded.decision_json,
                decision_sha256=excluded.decision_sha256,
                decision_status=excluded.decision_status,
                projection_version=excluded.projection_version,
                decision_index_version=CASE
                    WHEN full_report_payloads.decision_sha256=excluded.decision_sha256
                     AND full_report_payloads.projection_version=excluded.projection_version
                    THEN full_report_payloads.decision_index_version
                    ELSE 0
                END,
                updated_at=excluded.updated_at
            """,
            (
                document_id,
                digest.canonical_json,
                digest.canonical_sha256,
                digest.decision_json,
                digest.decision_sha256,
                digest.decision_status,
                digest.projection_version,
                now,
                now,
            ),
        )
        return needs_reindex

    def replace_digest_document(
        self,
        *,
        digest: Any,
        batch_id: str,
        require_daily_parent: bool = False,
        expected_existing_content_sha256: str | None = None,
        expected_existing_source_updated_at: str | None = None,
    ) -> dict[str, Any]:
        """Insert/update one digest and durably queue superseded vector deletion.

        SQLite is the source of truth.  Recording old child ids in the same
        transaction that makes the document pending closes the crash window in
        which Chroma could be deleted while SQLite still advertised a ready
        document.  The ingestion worker drains this small outbox before adding
        vectors for the replacement.
        """

        now = utc_now()
        values = {
            "id": digest.document_id,
            "filename": digest.filename,
            "sha256": digest.sha256,
            "size_bytes": digest.size_bytes,
            "source_key": digest.source_key,
            "content_sha256": digest.content_sha256,
            "ticker": digest.ticker,
            "analysis_date": digest.analysis_date,
            "analysis_cutoff": digest.cutoff,
            "source_run_id": digest.source_run_id,
            "source_updated_at": digest.source_updated_at,
            "batch_id": batch_id,
            "liquidity_rank": digest.liquidity_rank,
            "prompt_fingerprint": digest.prompt_fingerprint,
            "model_fingerprint": digest.model_fingerprint,
            "evidence_fingerprint": digest.evidence_fingerprint,
            "digest_status": digest.digest_status,
            "research_mode": digest.research_mode,
            "data_provenance": digest.data_provenance,
            "content_kind": digest.content_kind,
            "report_schema": digest.report_schema,
            "identity_hash": digest.identity_hash,
            "digest_hash": digest.digest_hash,
            "parent_identity_hash": digest.parent_identity_hash,
            "expected_digest_hash": digest.expected_digest_hash,
            "full_report_hash": digest.full_report_hash,
            "pipeline_version": digest.pipeline_version,
            "execution_generation": digest.execution_generation,
            "contains_decision_content": int(digest.contains_decision_content),
            "stage_status_json": json.dumps(
                digest.stage_status,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ),
            "stage_fingerprints_json": json.dumps(
                digest.stage_fingerprints,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ),
            "now": now,
        }
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            batch = db.execute(
                "SELECT * FROM digest_batches WHERE id=?", (batch_id,)
            ).fetchone()
            if batch is None:
                raise ConflictError("Digest batch không tồn tại")
            existing = db.execute(
                """
                SELECT id, collection_name, status, batch_id, content_kind,
                       content_sha256, identity_hash, digest_hash, parent_identity_hash,
                       expected_digest_hash, full_report_hash,
                       execution_generation, ticker, analysis_date,
                       analysis_cutoff, liquidity_rank, research_mode,
                       data_provenance, source_updated_at
                FROM documents WHERE source_key=?
                """,
                (digest.source_key,),
            ).fetchone()
            promotion_action = "promoted"
            if require_daily_parent:
                if batch["status"] != "sealed":
                    raise PromotionConflictError("parent_batch_not_sealed")
                promotion_action = self._assert_promotion_parent(
                    existing, digest=digest, batch_id=batch_id
                )
                if promotion_action == "unchanged":
                    document_id = str(existing["id"])
                    self._upsert_full_report_payload(
                        db,
                        document_id=document_id,
                        digest=digest,
                        now=now,
                    )
                    current = db.execute(
                        "SELECT * FROM documents WHERE id=?", (document_id,)
                    ).fetchone()
                    if current is None:  # pragma: no cover - protected by BEGIN IMMEDIATE.
                        raise PromotionConflictError("parent_digest_not_found")
                    result = dict(current)
                    result["_promotion_action"] = "unchanged"
                    return result
            else:
                self._assert_daily_upsert_parent(
                    batch,
                    existing,
                    digest=digest,
                    batch_id=batch_id,
                    expected_content_sha256=expected_existing_content_sha256,
                    expected_source_updated_at=expected_existing_source_updated_at,
                )
            if existing:
                if existing["status"] == "indexing":
                    raise DocumentBusyError(
                        "Digest đang indexing; hãy retry sau"
                    )
                document_id = str(existing["id"])
                old_collection = str(existing["collection_name"] or "")
                old_child_ids = db.execute(
                    "SELECT id FROM children WHERE document_id=?", (document_id,)
                ).fetchall()
                db.executemany(
                    """
                    INSERT OR IGNORE INTO pending_vector_deletions(
                        document_id, collection_name, child_id, created_at
                    ) VALUES (?, ?, ?, ?)
                    """,
                    [
                        (document_id, old_collection, str(row["id"]), now)
                        for row in old_child_ids
                    ],
                )
                db.execute("DELETE FROM child_fts WHERE document_id=?", (document_id,))
                db.execute("DELETE FROM children WHERE document_id=?", (document_id,))
                db.execute("DELETE FROM parents WHERE document_id=?", (document_id,))
                db.execute("DELETE FROM digest_sections WHERE document_id=?", (document_id,))
                db.execute("DELETE FROM digest_claims WHERE document_id=?", (document_id,))
                self._stale_graph_builds(db, document_id)
                db.execute(
                    """
                    UPDATE documents SET filename=:filename, sha256=:sha256, stored_path='',
                        size_bytes=:size_bytes, page_count=0, status='pending', error=NULL,
                        index_fingerprint=NULL, collection_name=NULL,
                        source_type='trading_digest', content_sha256=:content_sha256,
                        ticker=:ticker, analysis_date=:analysis_date,
                        analysis_cutoff=:analysis_cutoff, source_run_id=:source_run_id,
                        source_updated_at=:source_updated_at, batch_id=:batch_id,
                        liquidity_rank=:liquidity_rank,
                        prompt_fingerprint=:prompt_fingerprint,
                        model_fingerprint=:model_fingerprint,
                        evidence_fingerprint=:evidence_fingerprint,
                        digest_status=:digest_status, research_mode=:research_mode,
                        data_provenance=:data_provenance,
                        content_kind=:content_kind, report_schema=:report_schema,
                        identity_hash=:identity_hash, digest_hash=:digest_hash,
                        parent_identity_hash=:parent_identity_hash,
                        expected_digest_hash=:expected_digest_hash,
                        full_report_hash=:full_report_hash,
                        pipeline_version=:pipeline_version,
                        execution_generation=:execution_generation,
                        contains_decision_content=:contains_decision_content,
                        stage_status_json=:stage_status_json,
                        stage_fingerprints_json=:stage_fingerprints_json,
                        updated_at=:now
                    WHERE id=:document_id
                    """,
                    {**values, "document_id": document_id},
                )
            else:
                db.execute(
                    """
                    INSERT INTO documents(
                        id, filename, sha256, stored_path, size_bytes, page_count,
                        status, source_type, source_key, content_sha256, ticker,
                        analysis_date, analysis_cutoff, source_run_id, source_updated_at, batch_id,
                        liquidity_rank, prompt_fingerprint, model_fingerprint,
                        evidence_fingerprint, digest_status, research_mode,
                        data_provenance, content_kind, report_schema, identity_hash,
                        digest_hash, parent_identity_hash, full_report_hash,
                        expected_digest_hash,
                        pipeline_version, execution_generation,
                        contains_decision_content, stage_status_json,
                        stage_fingerprints_json, created_at, updated_at
                    ) VALUES (
                        :id, :filename, :sha256, '', :size_bytes, 0,
                        'pending', 'trading_digest', :source_key, :content_sha256,
                        :ticker, :analysis_date, :analysis_cutoff, :source_run_id,
                        :source_updated_at, :batch_id, :liquidity_rank,
                        :prompt_fingerprint, :model_fingerprint,
                        :evidence_fingerprint, :digest_status, :research_mode,
                        :data_provenance, :content_kind, :report_schema,
                        :identity_hash, :digest_hash, :parent_identity_hash,
                        :full_report_hash, :expected_digest_hash,
                        :pipeline_version, :execution_generation,
                        :contains_decision_content, :stage_status_json,
                        :stage_fingerprints_json, :now, :now
                    )
                    """,
                    values,
                )
                document_id = digest.document_id
            db.executemany(
                "INSERT INTO digest_sections(document_id, section, position, "
                "report_stage, section_status, text) VALUES (?, ?, ?, ?, ?, ?)",
                [
                    (
                        document_id,
                        section,
                        position,
                        digest.section_stages[section],
                        digest.section_statuses[section],
                        digest.sections[section],
                    )
                    for position, section in enumerate(digest.sections)
                ],
            )
            claim_rows: list[tuple[Any, ...]] = []
            for section, claims in (getattr(digest, "claims", {}) or {}).items():
                for position, claim in enumerate(claims):
                    claim_text = str(claim["text"])
                    evidence_ids = list(claim.get("evidence_ids") or [])
                    claim_rows.append(
                        (
                            _stable_storage_id(
                                "digest_claim",
                                document_id,
                                section,
                                position,
                                claim_text,
                                claim.get("confidence"),
                                evidence_ids,
                            ),
                            document_id,
                            section,
                            position,
                            claim_text,
                            json.dumps(
                                claim.get("confidence"),
                                ensure_ascii=False,
                                separators=(",", ":"),
                            )
                            if claim.get("confidence") is not None
                            else None,
                            json.dumps(
                                evidence_ids,
                                ensure_ascii=False,
                                separators=(",", ":"),
                            ),
                            now,
                        )
                    )
            db.executemany(
                """
                INSERT INTO digest_claims(
                    id, document_id, section, position, text,
                    confidence_json, evidence_ids_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                claim_rows,
            )
            self._upsert_full_report_payload(
                db,
                document_id=document_id,
                digest=digest,
                now=now,
            )
        result = self.get_document(document_id)
        if require_daily_parent and result is not None:
            result["_promotion_action"] = promotion_action
        return result  # type: ignore[return-value]

    @staticmethod
    def _assert_daily_upsert_parent(
        batch: sqlite3.Row,
        existing: sqlite3.Row | None,
        *,
        digest: Any,
        batch_id: str,
        expected_content_sha256: str | None,
        expected_source_updated_at: str | None,
    ) -> None:
        """Keep an open-batch daily write from racing seal/promotion."""

        if digest.content_kind != "daily_digest_v1":
            raise ConflictError("Daily upsert chỉ nhận daily_digest_v1")
        if batch["status"] != "open":
            raise ConflictError("Digest batch đã seal; không thể thay đổi nội dung")
        if (
            batch["source"] != digest.source
            or batch["analysis_date"] != digest.analysis_date
            or batch["cutoff"] != digest.cutoff
            or batch["research_mode"] != digest.research_mode
            or batch["data_provenance"] != digest.data_provenance
        ):
            raise ConflictError("Digest không khớp identity của batch")
        if existing is None:
            if (
                expected_content_sha256 is not None
                or expected_source_updated_at is not None
            ):
                raise ConflictError("Daily digest đã thay đổi đồng thời")
            return
        if str(existing["batch_id"] or "") != batch_id:
            raise ConflictError("Digest identity đã thuộc batch khác")
        if existing["content_kind"] != "daily_digest_v1":
            raise ConflictError("Không thể ghi đè Full Analysis bằng daily digest")
        if (
            expected_content_sha256 is None
            or expected_source_updated_at is None
            or existing["content_sha256"] != expected_content_sha256
            or existing["source_updated_at"] != expected_source_updated_at
        ):
            raise ConflictError("Daily digest đã thay đổi đồng thời")

    @staticmethod
    def _assert_promotion_parent(
        existing: sqlite3.Row | None, *, digest: Any, batch_id: str
    ) -> str:
        """Enforce the one-way daily-to-full transition under BEGIN IMMEDIATE."""

        if existing is None or str(existing["batch_id"] or "") != batch_id:
            raise PromotionConflictError("parent_digest_not_found")
        if existing["content_kind"] == "full_report_v1":
            exact = (
                existing["full_report_hash"] == digest.full_report_hash
                and existing["identity_hash"] == digest.identity_hash
                and existing["parent_identity_hash"] == digest.parent_identity_hash
                and existing["expected_digest_hash"] == digest.expected_digest_hash
                and int(existing["execution_generation"] or 0)
                == digest.execution_generation
            )
            if exact:
                return "unchanged"
            raise PromotionConflictError("full_report_already_promoted")
        if existing["content_kind"] != "daily_digest_v1":
            raise PromotionConflictError("parent_content_kind_invalid")
        if existing["identity_hash"] != digest.parent_identity_hash:
            raise PromotionConflictError("parent_identity_hash_mismatch")
        if existing["digest_hash"] != digest.expected_digest_hash:
            raise PromotionConflictError("expected_digest_hash_mismatch")
        if (
            existing["ticker"] != digest.ticker
            or existing["analysis_date"] != digest.analysis_date
            or existing["analysis_cutoff"] != digest.cutoff
            or int(existing["liquidity_rank"] or 0) != digest.liquidity_rank
            or existing["research_mode"] != digest.research_mode
            or existing["data_provenance"] != digest.data_provenance
        ):
            raise PromotionConflictError("parent_identity_mismatch")
        if digest.source_updated_at <= str(existing["source_updated_at"]):
            raise PromotionConflictError("source_updated_at_not_newer")
        return "promoted"

    def pending_vector_deletions(
        self, document_id: str
    ) -> list[dict[str, Any]]:
        """Return durable deletion groups without exposing vector text/metadata."""

        with self.connect() as db:
            rows = db.execute(
                """
                SELECT collection_name, child_id
                FROM pending_vector_deletions
                WHERE document_id=?
                ORDER BY collection_name, child_id
                """,
                (document_id,),
            ).fetchall()
        grouped: dict[str, list[str]] = {}
        for row in rows:
            grouped.setdefault(str(row["collection_name"]), []).append(
                str(row["child_id"])
            )
        return [
            {"collection_name": collection, "child_ids": child_ids}
            for collection, child_ids in grouped.items()
        ]

    def finish_vector_deletion(
        self, document_id: str, collection_name: str, child_ids: Sequence[str]
    ) -> None:
        if not child_ids:
            return
        placeholders = ",".join("?" for _ in child_ids)
        with self.connect() as db:
            db.execute(
                f"DELETE FROM pending_vector_deletions WHERE document_id=? "
                f"AND collection_name=? AND child_id IN ({placeholders})",
                (document_id, collection_name, *child_ids),
            )

    def pending_vector_metadata_updates(
        self,
        document_id: str | None = None,
        *,
        batch_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """Return crash-safe provenance updates together with vector location."""

        clauses = []
        params: list[Any] = []
        if document_id is not None:
            clauses.append("u.document_id=?")
            params.append(document_id)
        if batch_id is not None:
            clauses.append("d.batch_id=?")
            params.append(batch_id)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        with self.connect() as db:
            rows = db.execute(
                f"""
                SELECT u.document_id, u.research_mode, u.data_provenance,
                       d.status, d.collection_name
                FROM pending_vector_metadata_updates u
                JOIN documents d ON d.id=u.document_id
                {where}
                ORDER BY u.created_at, u.document_id
                """,
                tuple(params),
            ).fetchall()
            return [dict(row) for row in rows]

    def finish_vector_metadata_update(
        self,
        document_id: str,
        research_mode: str,
        data_provenance: str,
    ) -> None:
        """Ack only the exact update that was applied to Chroma."""

        with self.connect() as db:
            db.execute(
                """
                DELETE FROM pending_vector_metadata_updates
                WHERE document_id=? AND research_mode=? AND data_provenance=?
                """,
                (document_id, research_mode, data_provenance),
            )

    def advance_digest_source(
        self,
        document_id: str,
        *,
        batch_id: str,
        expected_content_sha256: str,
        expected_source_updated_at: str,
        allow_sealed: bool,
        source_updated_at: str,
        source_run_id: str | None,
        identity_hash: str | None = None,
        digest_hash: str | None = None,
    ) -> dict[str, Any]:
        """Advance a daily watermark and fill immutable v2 hashes if absent.

        Identity/digest hashes are metadata for promotion preconditions.  They
        do not change the indexed daily text, so adding them to an older exact
        replay must not trigger embedding work.
        """

        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            batch = db.execute(
                "SELECT status FROM digest_batches WHERE id=?", (batch_id,)
            ).fetchone()
            expected_batch_status = "sealed" if allow_sealed else "open"
            if batch is None or batch["status"] != expected_batch_status:
                raise ConflictError("Trạng thái digest batch đã thay đổi")
            document = db.execute(
                "SELECT * FROM documents WHERE id=?", (document_id,)
            ).fetchone()
            if (
                document is None
                or str(document["batch_id"] or "") != batch_id
                or document["content_kind"] != "daily_digest_v1"
                or document["content_sha256"] != expected_content_sha256
                or document["source_updated_at"] != expected_source_updated_at
            ):
                raise ConflictError(
                    "Daily digest đã thay đổi hoặc đã được promote"
                )
            if (
                document["identity_hash"]
                and identity_hash
                and document["identity_hash"] != identity_hash
            ) or (
                document["digest_hash"]
                and digest_hash
                and document["digest_hash"] != digest_hash
            ):
                raise ConflictError("Hash daily digest không khớp")
            if source_updated_at < expected_source_updated_at:
                raise ConflictError("source_updated_at cũ hơn daily digest hiện tại")
            db.execute(
                """
                UPDATE documents
                SET source_updated_at=?, source_run_id=COALESCE(?, source_run_id),
                    identity_hash=COALESCE(identity_hash, ?),
                    digest_hash=COALESCE(digest_hash, ?),
                    updated_at=?
                WHERE id=? AND source_updated_at=?
                """,
                (
                    source_updated_at,
                    source_run_id,
                    identity_hash,
                    digest_hash,
                    utc_now(),
                    document_id,
                    expected_source_updated_at,
                ),
            )
            current = db.execute(
                "SELECT * FROM documents WHERE id=?", (document_id,)
            ).fetchone()
            assert current is not None
            return dict(current)

    def requeue_daily_document(
        self,
        document_id: str,
        *,
        batch_id: str,
        expected_content_sha256: str,
        expected_source_updated_at: str,
        allow_sealed: bool,
    ) -> dict[str, Any]:
        """Retry a daily row without ever mutating a promoted Full Analysis."""

        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            batch = db.execute(
                "SELECT status FROM digest_batches WHERE id=?", (batch_id,)
            ).fetchone()
            expected_batch_status = "sealed" if allow_sealed else "open"
            if batch is None or batch["status"] != expected_batch_status:
                raise ConflictError("Trạng thái digest batch đã thay đổi")
            document = db.execute(
                "SELECT * FROM documents WHERE id=?", (document_id,)
            ).fetchone()
            if (
                document is None
                or str(document["batch_id"] or "") != batch_id
                or document["content_kind"] != "daily_digest_v1"
                or document["content_sha256"] != expected_content_sha256
                or document["source_updated_at"] != expected_source_updated_at
            ):
                raise ConflictError(
                    "Daily digest đã thay đổi hoặc đã được promote"
                )
            if document["status"] == "failed":
                collection_name = str(document["collection_name"] or "")
                child_ids = db.execute(
                    "SELECT id FROM children WHERE document_id=?", (document_id,)
                ).fetchall()
                now = utc_now()
                db.executemany(
                    """
                    INSERT OR IGNORE INTO pending_vector_deletions(
                        document_id, collection_name, child_id, created_at
                    ) VALUES (?, ?, ?, ?)
                    """,
                    [
                        (document_id, collection_name, str(row["id"]), now)
                        for row in child_ids
                    ],
                )
                db.execute(
                    "DELETE FROM child_fts WHERE document_id=?", (document_id,)
                )
                db.execute(
                    "DELETE FROM children WHERE document_id=?", (document_id,)
                )
                db.execute(
                    "DELETE FROM parents WHERE document_id=?", (document_id,)
                )
                self._stale_graph_builds(db, document_id)
                db.execute(
                    """
                    UPDATE documents SET status='pending', error=NULL,
                        index_fingerprint=NULL, collection_name=NULL, updated_at=?
                    WHERE id=? AND status='failed'
                    """,
                    (now, document_id),
                )
            current = db.execute(
                "SELECT * FROM documents WHERE id=?", (document_id,)
            ).fetchone()
            assert current is not None
            return dict(current)

    def requeue_full_report(
        self,
        document_id: str,
        *,
        batch_id: str,
        digest: Any,
    ) -> dict[str, Any]:
        """Retry one exact Full Analysis generation without resetting a worker."""

        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            document = db.execute(
                "SELECT * FROM documents WHERE id=?", (document_id,)
            ).fetchone()
            if document is None:
                raise PromotionConflictError("parent_digest_not_found")
            if self._assert_promotion_parent(
                document, digest=digest, batch_id=batch_id
            ) != "unchanged":  # pragma: no cover - exact full is the only path.
                raise PromotionConflictError("full_report_already_promoted")
            if document["status"] == "failed":
                collection_name = str(document["collection_name"] or "")
                child_ids = db.execute(
                    "SELECT id FROM children WHERE document_id=?", (document_id,)
                ).fetchall()
                now = utc_now()
                db.executemany(
                    """
                    INSERT OR IGNORE INTO pending_vector_deletions(
                        document_id, collection_name, child_id, created_at
                    ) VALUES (?, ?, ?, ?)
                    """,
                    [
                        (document_id, collection_name, str(row["id"]), now)
                        for row in child_ids
                    ],
                )
                db.execute(
                    "DELETE FROM child_fts WHERE document_id=?", (document_id,)
                )
                db.execute(
                    "DELETE FROM children WHERE document_id=?", (document_id,)
                )
                db.execute(
                    "DELETE FROM parents WHERE document_id=?", (document_id,)
                )
                self._stale_graph_builds(db, document_id)
                db.execute(
                    """
                    UPDATE documents SET status='pending', error=NULL,
                        index_fingerprint=NULL, collection_name=NULL, updated_at=?
                    WHERE id=? AND status='failed'
                    """,
                    (now, document_id),
                )
            current = db.execute(
                "SELECT * FROM documents WHERE id=?", (document_id,)
            ).fetchone()
            assert current is not None
            return dict(current)

    def digest_sections(self, document_id: str) -> list[dict[str, Any]]:
        with self.connect() as db:
            rows = db.execute(
                "SELECT section, position, report_stage, section_status, text "
                "FROM digest_sections "
                "WHERE document_id=? ORDER BY position",
                (document_id,),
            ).fetchall()
            return [dict(row) for row in rows]

    def digest_claims(self, document_id: str) -> list[dict[str, Any]]:
        """Return typed daily claims without exposing producer-only fields."""

        with self.connect() as db:
            rows = db.execute(
                """
                SELECT id, document_id, section, position, text,
                       confidence_json, evidence_ids_json, created_at
                FROM digest_claims
                WHERE document_id=?
                ORDER BY section, position
                """,
                (document_id,),
            ).fetchall()
        claims: list[dict[str, Any]] = []
        for row in rows:
            item = dict(row)
            confidence_json = item.pop("confidence_json")
            item["confidence"] = (
                json.loads(confidence_json) if confidence_json is not None else None
            )
            item["evidence_ids"] = json.loads(item.pop("evidence_ids_json"))
            claims.append(item)
        return claims

    def get_full_report_payload(self, document_id: str) -> dict[str, Any] | None:
        with self.connect() as db:
            row = db.execute(
                "SELECT * FROM full_report_payloads WHERE document_id=?",
                (document_id,),
            ).fetchone()
            return dict(row) if row else None

    def full_report_backfill_candidates(self) -> list[dict[str, Any]]:
        with self.connect() as db:
            rows = db.execute(
                """
                SELECT d.*, frp.canonical_sha256, frp.decision_sha256,
                       frp.canonical_json AS stored_canonical_json,
                       frp.decision_json AS stored_decision_json,
                       frp.decision_status AS stored_decision_status,
                       frp.projection_version, frp.decision_index_version
                FROM documents d
                LEFT JOIN full_report_payloads frp ON frp.document_id=d.id
                WHERE d.source_type='trading_digest' AND d.content_kind=?
                ORDER BY d.analysis_date, d.liquidity_rank, d.ticker
                """,
                (FULL_REPORT_V1,),
            ).fetchall()
            return [dict(row) for row in rows]

    def reconstruct_full_report_wire(self, document_id: str) -> dict[str, Any]:
        """Rebuild the frozen envelope from canonical relational columns only."""

        with self.connect() as db:
            document = db.execute(
                """
                SELECT d.*, b.source AS batch_source
                FROM documents d
                JOIN digest_batches b ON b.id=d.batch_id
                WHERE d.id=? AND d.source_type='trading_digest'
                  AND d.content_kind=?
                """,
                (document_id, FULL_REPORT_V1),
            ).fetchone()
            if document is None:
                raise ConflictError("Full Analysis document không tồn tại")
            rows = db.execute(
                """
                SELECT section, section_status, text
                FROM digest_sections
                WHERE document_id=?
                ORDER BY position
                """,
                (document_id,),
            ).fetchall()
        sections = {
            str(row["section"]): {
                "status": str(row["section_status"]),
                "text": str(row["text"]),
            }
            for row in rows
        }
        return {
            "schema_version": 2,
            "content_kind": document["content_kind"],
            "report_schema": document["report_schema"],
            "source": document["batch_source"],
            "ticker": document["ticker"],
            "analysis_date": document["analysis_date"],
            "cutoff": document["analysis_cutoff"],
            "source_updated_at": document["source_updated_at"],
            "liquidity_rank": document["liquidity_rank"],
            "research_mode": document["research_mode"],
            "data_provenance": document["data_provenance"],
            "identity_hash": document["identity_hash"],
            "parent_identity_hash": document["parent_identity_hash"],
            "expected_digest_hash": document["expected_digest_hash"],
            "full_report_hash": document["full_report_hash"],
            "pipeline_version": document["pipeline_version"],
            "prompt_fingerprint": document["prompt_fingerprint"],
            "model_fingerprint": document["model_fingerprint"],
            "evidence_fingerprint": document["evidence_fingerprint"],
            "execution_generation": document["execution_generation"],
            "contains_decision_content": bool(
                document["contains_decision_content"]
            ),
            "status": document["digest_status"],
            "stage_status": json.loads(document["stage_status_json"]),
            "stage_fingerprints": json.loads(
                document["stage_fingerprints_json"]
            ),
            "sections": sections,
        }

    def store_full_report_payload(self, document_id: str, digest: Any) -> bool:
        """Store a reconstructed payload and report whether indexing is required."""

        now = utc_now()
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            document = db.execute(
                """
                SELECT id, source_key, content_kind, full_report_hash,
                       identity_hash, ticker, analysis_date, analysis_cutoff
                FROM documents WHERE id=?
                """,
                (document_id,),
            ).fetchone()
            if document is None or document["content_kind"] != FULL_REPORT_V1:
                raise ConflictError("Full Analysis document không tồn tại")
            if (
                digest.document_id != document_id
                or digest.source_key != document["source_key"]
                or digest.full_report_hash != document["full_report_hash"]
                or digest.identity_hash != document["identity_hash"]
                or digest.ticker != document["ticker"]
                or digest.analysis_date != document["analysis_date"]
                or digest.cutoff != document["analysis_cutoff"]
            ):
                raise ConflictError("Full Analysis identity/hash không khớp")
            return self._upsert_full_report_payload(
                db,
                document_id=document_id,
                digest=digest,
                now=now,
            )

    def structured_decision_section(
        self, document_id: str
    ) -> dict[str, Any] | None:
        payload = self.get_full_report_payload(document_id)
        if payload is None:
            return None
        return {
            "section": DECISION_STRUCTURED_SECTION,
            "position": len(FULL_SECTION_ORDER),
            "report_stage": "risk",
            "section_status": payload["decision_status"],
            "text": payload["decision_json"],
        }

    def full_report_payload_health(self) -> dict[str, int]:
        with self.connect() as db:
            row = db.execute(
                """
                SELECT
                    count(*) AS full_reports,
                    coalesce(sum(CASE
                        WHEN d.status='ready' AND frp.document_id IS NULL THEN 1
                        ELSE 0 END), 0) AS missing_ready_payloads,
                    coalesce(sum(CASE
                        WHEN d.status='ready' AND frp.document_id IS NOT NULL
                         AND frp.decision_index_version <> frp.projection_version
                        THEN 1 ELSE 0 END), 0) AS pending_ready_decision_indexes,
                    coalesce(sum(CASE
                        WHEN frp.decision_status='partial' THEN 1 ELSE 0 END), 0)
                        AS partial_decisions
                FROM documents d
                LEFT JOIN full_report_payloads frp ON frp.document_id=d.id
                WHERE d.source_type='trading_digest' AND d.content_kind=?
                """,
                (FULL_REPORT_V1,),
            ).fetchone()
            assert row is not None
            return {key: int(row[key]) for key in row.keys()}

    def claim_next_pending(self) -> dict[str, Any] | None:
        claimed = self.claim_pending(1)
        return claimed[0] if claimed else None

    def claim_pending(self, limit: int) -> list[dict[str, Any]]:
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            rows = db.execute(
                "SELECT * FROM documents WHERE status='pending' "
                "ORDER BY created_at LIMIT ?",
                (limit,),
            ).fetchall()
            if not rows:
                return []
            claimed: list[dict[str, Any]] = []
            now = utc_now()
            for row in rows:
                changed = db.execute(
                    "UPDATE documents SET status='indexing', error=NULL, updated_at=? "
                    "WHERE id=? AND status='pending'",
                    (now, row["id"]),
                ).rowcount
                if changed == 1:
                    item = dict(row)
                    item["status"] = "indexing"
                    claimed.append(item)
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
        normalised_parents = [
            {
                **parent,
                "page_end": parent.get("page_end", parent["page"]),
                "kind": parent.get("kind", "page"),
                "digest_section": parent.get("digest_section"),
                "report_stage": parent.get("report_stage"),
                "section_status": parent.get("section_status"),
            }
            for parent in parents
        ]
        normalised_children = [
            {
                **child,
                "page_end": child.get("page_end", child["page"]),
                "kind": child.get("kind", "page"),
                "digest_section": child.get("digest_section"),
                "report_stage": child.get("report_stage"),
                "section_status": child.get("section_status"),
            }
            for child in children
        ]
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            db.execute("DELETE FROM child_fts WHERE document_id=?", (document_id,))
            db.execute("DELETE FROM children WHERE document_id=?", (document_id,))
            db.execute("DELETE FROM parents WHERE document_id=?", (document_id,))
            db.executemany(
                "INSERT INTO parents(id, document_id, page, page_end, kind, "
                "digest_section, report_stage, section_status, start_index, text) "
                "VALUES (:id, :document_id, :page, :page_end, :kind, "
                ":digest_section, :report_stage, :section_status, :start_index, :text)",
                normalised_parents,
            )
            db.executemany(
                "INSERT INTO children(id, parent_id, document_id, page, page_end, "
                "kind, digest_section, report_stage, section_status, start_index, text) "
                "VALUES (:id, :parent_id, :document_id, :page, :page_end, :kind, "
                ":digest_section, :report_stage, :section_status, :start_index, :text)",
                normalised_children,
            )
            db.executemany(
                "INSERT INTO child_fts(child_id, parent_id, document_id, text) "
                "VALUES (:id, :parent_id, :document_id, :fts_text)",
                [
                    {**child, "fts_text": normalise_fts_text(str(child["text"]))}
                    for child in normalised_children
                ],
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
            db.execute(
                """
                UPDATE full_report_payloads
                SET decision_index_version=projection_version, updated_at=?
                WHERE document_id=?
                """,
                (utc_now(), document_id),
            )

    def mark_failed(self, document_id: str, error_code: str) -> None:
        with self.connect() as db:
            db.execute("DELETE FROM child_fts WHERE document_id=?", (document_id,))
            db.execute("DELETE FROM children WHERE document_id=?", (document_id,))
            db.execute("DELETE FROM parents WHERE document_id=?", (document_id,))
            self._stale_graph_builds(db, document_id)
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
            document = db.execute(
                "SELECT collection_name FROM documents WHERE id=?", (document_id,)
            ).fetchone()
            if document is None:
                return
            collection_name = str(document["collection_name"] or "")
            child_ids = db.execute(
                "SELECT id FROM children WHERE document_id=?", (document_id,)
            ).fetchall()
            now = utc_now()
            db.executemany(
                """
                INSERT OR IGNORE INTO pending_vector_deletions(
                    document_id, collection_name, child_id, created_at
                ) VALUES (?, ?, ?, ?)
                """,
                [
                    (document_id, collection_name, str(row["id"]), now)
                    for row in child_ids
                ],
            )
            db.execute("DELETE FROM child_fts WHERE document_id=?", (document_id,))
            db.execute("DELETE FROM children WHERE document_id=?", (document_id,))
            db.execute("DELETE FROM parents WHERE document_id=?", (document_id,))
            self._stale_graph_builds(db, document_id)
            db.execute(
                """
                UPDATE documents SET status='pending', error=NULL,
                    index_fingerprint=NULL, collection_name=NULL, updated_at=?
                WHERE id=?
                """,
                (now, document_id),
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
                SELECT p.*, d.filename, d.source_type, d.ticker,
                       d.analysis_date, d.analysis_cutoff, d.batch_id,
                       d.liquidity_rank, d.digest_status, d.research_mode,
                       d.data_provenance, d.content_kind, d.report_schema,
                       d.identity_hash, d.digest_hash, d.parent_identity_hash,
                       d.expected_digest_hash, d.full_report_hash, d.pipeline_version,
                       d.execution_generation, d.contains_decision_content
                FROM parents p JOIN documents d ON d.id=p.document_id
                WHERE p.id IN ({placeholders}) AND d.status='ready'
                """,
                tuple(parent_ids),
            ).fetchall()
            return {row["id"]: dict(row) for row in rows}

    def lexical_search(
        self,
        query: str,
        *,
        k: int,
        document_ids: Sequence[str] | None = None,
        filters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        match_query = build_fts_query(query)
        if not match_query or (document_ids is not None and not document_ids):
            return []
        clauses = ["child_fts MATCH ?", "d.status='ready'"]
        params: list[Any] = [match_query]
        if document_ids is not None:
            placeholders = ",".join("?" for _ in document_ids)
            clauses.append(f"f.document_id IN ({placeholders})")
            params.extend(document_ids)
        filter_clauses, filter_params = _document_filter_sql(filters, alias="d")
        clauses.extend(filter_clauses)
        params.extend(filter_params)
        digest_sections = list(dict.fromkeys((filters or {}).get("digest_sections") or []))
        if digest_sections:
            placeholders = ",".join("?" for _ in digest_sections)
            clauses.append(f"c.digest_section IN ({placeholders})")
            params.extend(digest_sections)
        report_stages = list(
            dict.fromkeys((filters or {}).get("report_stages") or [])
        )
        if report_stages:
            placeholders = ",".join("?" for _ in report_stages)
            clauses.append(f"c.report_stage IN ({placeholders})")
            params.extend(report_stages)
        with self.connect() as db:
            rows = db.execute(
                f"""
                SELECT f.child_id, f.parent_id, f.document_id,
                       c.text AS text, bm25(child_fts) AS lexical_score
                FROM child_fts f
                JOIN children c ON c.id=f.child_id
                JOIN documents d ON d.id=f.document_id
                WHERE {' AND '.join(clauses)}
                ORDER BY lexical_score ASC
                LIMIT ?
                """,
                (*params, k),
            ).fetchall()
            return [dict(row) for row in rows]

    def delete_document_rows(self, document_id: str) -> None:
        with self.connect() as db:
            db.execute("DELETE FROM child_fts WHERE document_id=?", (document_id,))
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
                self._stale_graph_builds(db, document_id)
                db.execute(
                    "UPDATE documents SET status='failed', error='deleting', updated_at=? "
                    "WHERE id=?",
                    (utc_now(), document_id),
                )
            return document

    def ready_document_ids(
        self,
        requested: Sequence[str] | None = None,
        filters: dict[str, Any] | None = None,
    ) -> list[str]:
        clauses = ["d.status='ready'"]
        params: list[Any] = []
        filter_clauses, filter_params = _document_filter_sql(filters, alias="d")
        clauses.extend(filter_clauses)
        params.extend(filter_params)
        with self.connect() as db:
            if requested is not None and not requested:
                return []
            if requested is not None:
                placeholders = ",".join("?" for _ in requested)
                clauses.append(f"d.id IN ({placeholders})")
                params.extend(requested)
            rows = db.execute(
                f"SELECT d.id FROM documents d WHERE {' AND '.join(clauses)}",
                tuple(params),
            ).fetchall()
            return [row["id"] for row in rows]

    def latest_ready_document_ids(self, allowed_ids: Sequence[str]) -> list[str]:
        """Narrow an existing allow-list to current normalized research versions.

        The input is already the caller's immutable scope.  PDFs, legacy rows and
        tied daily/full projections are retained; only older normalized research
        snapshots for the same ticker are removed.
        """

        ordered_ids = list(dict.fromkeys(str(item) for item in allowed_ids))
        if not ordered_ids:
            return []
        rows: dict[str, dict[str, Any]] = {}
        # Stay below SQLite's conservative bind-variable limit for large corpora.
        for start in range(0, len(ordered_ids), 400):
            batch = ordered_ids[start : start + 400]
            placeholders = ",".join("?" for _ in batch)
            with self.connect() as db:
                selected = db.execute(
                    f"""
                    SELECT id, source_type, content_kind, ticker,
                           analysis_date, analysis_cutoff
                    FROM documents
                    WHERE status='ready' AND id IN ({placeholders})
                    """,
                    tuple(batch),
                ).fetchall()
            rows.update((str(row["id"]), dict(row)) for row in selected)

        newest_by_ticker: dict[str, tuple[str, str]] = {}
        for row in rows.values():
            ticker = str(row.get("ticker") or "").strip().upper()
            if (
                row.get("source_type") != "trading_digest"
                or row.get("content_kind") not in {"daily_digest_v1", "full_report_v1"}
                or not ticker
            ):
                continue
            version = (
                str(row.get("analysis_date") or ""),
                str(row.get("analysis_cutoff") or ""),
            )
            newest_by_ticker[ticker] = max(
                version, newest_by_ticker.get(ticker, version)
            )

        retained: list[str] = []
        for document_id in ordered_ids:
            row = rows.get(document_id)
            if row is None:
                continue
            ticker = str(row.get("ticker") or "").strip().upper()
            normalized = (
                row.get("source_type") == "trading_digest"
                and row.get("content_kind") in {"daily_digest_v1", "full_report_v1"}
                and bool(ticker)
            )
            version = (
                str(row.get("analysis_date") or ""),
                str(row.get("analysis_cutoff") or ""),
            )
            if not normalized or version == newest_by_ticker[ticker]:
                retained.append(document_id)
        return retained

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
