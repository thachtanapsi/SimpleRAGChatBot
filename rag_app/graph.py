"""Evidence-first GraphRAG over the canonical SQLite document store.

The graph is deliberately derived data: every assertion belongs to a versioned
build and every usable edge has an exact quote in a current parent chunk.  No
graph identifier is a citation; retrieval always returns the underlying parent
record so the normal citation pipeline remains the source of truth.
"""

from __future__ import annotations

import asyncio
import hashlib
import inspect
import json
import re
import sqlite3
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Awaitable, Callable, Mapping, Protocol, Sequence

from .digests import DIGEST_CONTENT_KIND, TICKER_RE
from .full_reports import DECISION_STRUCTURED_SECTION, FULL_REPORT_V1
from .storage import (
    FTS_TOKEN_RE,
    SQLiteStorage,
    _document_filter_sql,
    build_fts_query,
    normalise_fts_text,
    utc_now,
)


GRAPH_SCHEMA_VERSION = 1
GRAPH_CONTENT_KINDS = frozenset({DIGEST_CONTENT_KIND, FULL_REPORT_V1})
GRAPH_ENTITY_TYPES = frozenset(
    {
        "security",
        "organization",
        "person",
        "sector",
        "market",
        "geography",
        "currency",
        "product",
        "metric",
        "event",
        "risk",
        "catalyst",
        "concept",
    }
)
GRAPH_PREDICATES = frozenset(
    {
        "AFFECTED_BY",
        "EXPOSED_TO",
        "OPERATES_IN",
        "OWNS",
        "PARTNERS_WITH",
        "COMPETES_WITH",
        "SUPPLIES_TO",
        "CUSTOMER_OF",
        "MEMBER_OF",
        "HAS_RISK",
        "HAS_CATALYST",
        "REPORTS_METRIC",
        "HAS_SENTIMENT",
        "RECOMMENDS",
        "TRADING_ACTION",
        "TARGET_PRICE",
    }
)
DEFAULT_MAX_RETRIES = 3


def _stable_id(prefix: str, *parts: object) -> str:
    encoded = json.dumps(
        parts, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return f"{prefix}_{hashlib.sha256(encoded).hexdigest()}"


def _canonical_key(value: object) -> str:
    return " ".join(normalise_fts_text(str(value)).split())


def graph_build_id(
    document_id: str,
    content_sha256: str,
    index_fingerprint: str,
    extractor_fingerprint: str,
) -> str:
    return _stable_id(
        "graph_build",
        document_id,
        content_sha256,
        index_fingerprint,
        extractor_fingerprint,
    )


@dataclass(frozen=True, slots=True)
class GraphEntityInput:
    entity_type: str
    canonical_key: str
    canonical_name: str

    @classmethod
    def from_value(cls, value: "GraphEntityInput | Mapping[str, Any]") -> "GraphEntityInput":
        if isinstance(value, cls):
            entity = value
        elif isinstance(value, Mapping):
            entity = cls(
                entity_type=str(value.get("entity_type") or value.get("type") or ""),
                canonical_key=str(value.get("canonical_key") or value.get("key") or ""),
                canonical_name=str(value.get("canonical_name") or value.get("name") or ""),
            )
        else:
            raise ValueError("graph entity must be an object")
        entity_type = entity.entity_type.strip().casefold()
        canonical_key = _canonical_key(entity.canonical_key)
        canonical_name = entity.canonical_name.strip()
        if entity_type not in GRAPH_ENTITY_TYPES:
            raise ValueError("graph entity type is not allowed")
        if not canonical_key or not canonical_name:
            raise ValueError("graph entity identity is empty")
        if len(canonical_key) > 160 or len(canonical_name) > 240:
            raise ValueError("graph entity identity is too long")
        if entity_type == "security" and not TICKER_RE.fullmatch(
            canonical_name.upper()
        ):
            raise ValueError("security entity must use a valid ticker")
        if entity_type == "security" and canonical_key != _canonical_key(
            canonical_name.upper()
        ):
            raise ValueError("security entity key must equal its ticker")
        return cls(entity_type, canonical_key, canonical_name)

    @property
    def id(self) -> str:
        return _stable_id("graph_entity", self.entity_type, self.canonical_key)


@dataclass(frozen=True, slots=True)
class GraphAssertionInput:
    predicate: str
    quote: str
    subject: GraphEntityInput | Mapping[str, Any] | None = None
    object_entity: GraphEntityInput | Mapping[str, Any] | None = None
    object_literal: Any | None = None
    object_type: str | None = None
    modality: str = "asserted"
    valid_at: str | None = None
    confidence: float = 1.0

    @classmethod
    def from_value(
        cls,
        value: "GraphAssertionInput | Mapping[str, Any]",
        *,
        default_subject: GraphEntityInput,
    ) -> "GraphAssertionInput":
        if isinstance(value, cls):
            raw = value
        elif isinstance(value, Mapping):
            raw = cls(
                predicate=str(value.get("predicate") or ""),
                quote=str(value.get("quote") or ""),
                subject=value.get("subject"),
                object_entity=value.get("object_entity") or value.get("object"),
                object_literal=value.get("object_literal"),
                object_type=(
                    str(value["object_type"]) if value.get("object_type") else None
                ),
                modality=str(value.get("modality") or "asserted"),
                valid_at=(str(value["valid_at"]) if value.get("valid_at") else None),
                confidence=float(value.get("confidence", 1.0)),
            )
        else:
            raise ValueError("graph assertion must be an object")
        predicate = raw.predicate.strip().upper()
        quote = raw.quote.strip()
        subject = GraphEntityInput.from_value(raw.subject or default_subject)
        object_entity = (
            GraphEntityInput.from_value(raw.object_entity)
            if raw.object_entity is not None
            else None
        )
        has_literal = raw.object_literal is not None
        if predicate not in GRAPH_PREDICATES:
            raise ValueError("graph predicate is not allowed")
        if not quote:
            raise ValueError("graph evidence quote is empty")
        if len(quote) > 2400:
            raise ValueError("graph evidence quote is too long")
        if (object_entity is None) == (not has_literal):
            raise ValueError("graph assertion requires exactly one object")
        confidence = float(raw.confidence)
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("graph assertion confidence is outside 0..1")
        modality = raw.modality.strip() or "asserted"
        if len(modality) > 64:
            raise ValueError("graph assertion modality is too long")
        object_type = raw.object_type.strip() if raw.object_type else None
        if object_type is not None and len(object_type) > 64:
            raise ValueError("graph object type is too long")
        if raw.valid_at is not None and len(raw.valid_at) > 64:
            raise ValueError("graph assertion valid_at is too long")
        return cls(
            predicate=predicate,
            quote=quote,
            subject=subject,
            object_entity=object_entity,
            object_literal=raw.object_literal,
            object_type=object_type,
            modality=modality,
            valid_at=raw.valid_at,
            confidence=confidence,
        )


@dataclass(frozen=True, slots=True)
class GraphExtractionInput:
    document: Mapping[str, Any]
    parent: Mapping[str, Any]
    claims: tuple[Mapping[str, Any], ...]
    default_subject: GraphEntityInput


GraphExtractionResult = Sequence[GraphAssertionInput | Mapping[str, Any]]


class GraphExtractor(Protocol):
    def __call__(
        self, value: GraphExtractionInput
    ) -> GraphExtractionResult | Awaitable[GraphExtractionResult]: ...


@dataclass(slots=True)
class _PreparedGraph:
    entities: dict[str, GraphEntityInput] = field(default_factory=dict)
    mentions: dict[str, dict[str, Any]] = field(default_factory=dict)
    assertions: dict[str, dict[str, Any]] = field(default_factory=dict)
    evidence: dict[str, dict[str, Any]] = field(default_factory=dict)


class GraphService:
    """Version, build, and maintain evidence graphs for normalized research."""

    def __init__(
        self,
        storage: SQLiteStorage,
        *,
        index_fingerprint: str,
        extractor_fingerprint: str,
        extractor: GraphExtractor | None = None,
        generation_lock: asyncio.Semaphore | None = None,
        logger: Any | None = None,
        worker_concurrency: int = 1,
        max_retries: int = DEFAULT_MAX_RETRIES,
    ):
        if not index_fingerprint or not extractor_fingerprint:
            raise ValueError("graph fingerprints are required")
        self.storage = storage
        self.index_fingerprint = index_fingerprint
        self.extractor_fingerprint = extractor_fingerprint
        self.extractor = extractor
        self.generation_lock = generation_lock
        self.logger = logger
        self.worker_concurrency = max(1, int(worker_concurrency))
        self.max_retries = max(1, int(max_retries))
        self._wake_event = asyncio.Event()
        self._stop_event = asyncio.Event()
        self._tasks: list[asyncio.Task[None]] = []

    def _eligible_documents(self, document_id: str | None = None) -> list[dict[str, Any]]:
        clauses = [
            "status='ready'",
            "source_type='trading_digest'",
            "content_kind IN (?, ?)",
            "content_sha256 IS NOT NULL",
            "index_fingerprint=?",
        ]
        params: list[Any] = [
            DIGEST_CONTENT_KIND,
            FULL_REPORT_V1,
            self.index_fingerprint,
        ]
        if document_id is not None:
            clauses.append("id=?")
            params.append(document_id)
        with self.storage.connect() as db:
            rows = db.execute(
                f"SELECT * FROM documents WHERE {' AND '.join(clauses)} "
                "ORDER BY analysis_date, liquidity_rank, ticker, id",
                tuple(params),
            ).fetchall()
        return [dict(row) for row in rows]

    def reconcile(
        self, document_id: str | None = None, *, apply: bool = True
    ) -> dict[str, Any]:
        """Find current normalized documents and enqueue missing graph builds."""

        documents = self._eligible_documents(document_id)
        items: list[dict[str, Any]] = []
        now = utc_now()
        with self.storage.connect() as db:
            if apply:
                db.execute("BEGIN IMMEDIATE")
            for document in documents:
                build_id = graph_build_id(
                    str(document["id"]),
                    str(document["content_sha256"]),
                    self.index_fingerprint,
                    self.extractor_fingerprint,
                )
                existing = db.execute(
                    "SELECT * FROM graph_builds WHERE id=?", (build_id,)
                ).fetchone()
                if existing is None:
                    action = "enqueue"
                elif existing["status"] == "failed" and int(existing["retry_count"]) < self.max_retries:
                    action = "retry"
                elif existing["status"] == "stale":
                    action = "enqueue"
                elif existing["status"] == "ready" and int(existing["active"]) == 1:
                    action = "current"
                else:
                    action = str(existing["status"])
                items.append(
                    {
                        "document_id": document["id"],
                        "build_id": build_id,
                        "action": action,
                    }
                )
                if not apply:
                    continue
                db.execute(
                    """
                    UPDATE graph_builds
                    SET status='stale', active=0, updated_at=?
                    WHERE document_id=? AND id<>? AND status<>'stale'
                    """,
                    (now, document["id"], build_id),
                )
                if existing is None:
                    db.execute(
                        """
                        INSERT INTO graph_builds(
                            id, document_id, content_sha256, index_fingerprint,
                            extractor_fingerprint, status, active, retry_count,
                            created_at, updated_at
                        ) VALUES (?, ?, ?, ?, ?, 'pending', 0, 0, ?, ?)
                        """,
                        (
                            build_id,
                            document["id"],
                            document["content_sha256"],
                            self.index_fingerprint,
                            self.extractor_fingerprint,
                            now,
                            now,
                        ),
                    )
                elif action in {"retry", "enqueue"}:
                    db.execute(
                        """
                        UPDATE graph_builds
                        SET status='pending', active=0, error_code=NULL,
                            started_at=NULL, completed_at=NULL, updated_at=?
                        WHERE id=?
                        """,
                        (now, build_id),
                    )
        if apply and any(item["action"] in {"enqueue", "retry"} for item in items):
            self.wake()
        return {
            "apply": apply,
            "eligible": len(documents),
            "enqueued": sum(
                item["action"] in {"enqueue", "retry"} for item in items
            ),
            "items": items,
        }

    def reindex(
        self, document_id: str | None = None, *, apply: bool = False
    ) -> dict[str, Any]:
        """Dry-run or reset current graph builds without touching source data."""

        plan = self.reconcile(document_id, apply=False)
        for item in plan["items"]:
            item["action"] = "rebuild"
        if not apply:
            return plan
        now = utc_now()
        with self.storage.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            for item in plan["items"]:
                document = db.execute(
                    "SELECT * FROM documents WHERE id=?", (item["document_id"],)
                ).fetchone()
                if document is None:
                    continue
                build_id = graph_build_id(
                    str(document["id"]),
                    str(document["content_sha256"]),
                    self.index_fingerprint,
                    self.extractor_fingerprint,
                )
                db.execute(
                    "UPDATE graph_builds SET status='stale', active=0, updated_at=? "
                    "WHERE document_id=?",
                    (now, document["id"]),
                )
                db.execute("DELETE FROM graph_evidence WHERE build_id=?", (build_id,))
                db.execute("DELETE FROM graph_assertions WHERE build_id=?", (build_id,))
                db.execute("DELETE FROM graph_mentions WHERE build_id=?", (build_id,))
                db.execute(
                    """
                    INSERT INTO graph_builds(
                        id, document_id, content_sha256, index_fingerprint,
                        extractor_fingerprint, status, active, retry_count,
                        created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, 'pending', 0, 0, ?, ?)
                    ON CONFLICT(id) DO UPDATE SET
                        status='pending', active=0, retry_count=0,
                        error_code=NULL, started_at=NULL, completed_at=NULL,
                        updated_at=excluded.updated_at
                    """,
                    (
                        build_id,
                        document["id"],
                        document["content_sha256"],
                        self.index_fingerprint,
                        self.extractor_fingerprint,
                        now,
                        now,
                    ),
                )
                item["build_id"] = build_id
        plan["apply"] = True
        plan["enqueued"] = len(plan["items"])
        self.wake()
        return plan

    def _claim_build(self, document_id: str | None = None) -> dict[str, Any] | None:
        clauses = [
            "gb.status='pending'",
            "d.status='ready'",
            "d.source_type='trading_digest'",
            "d.content_kind IN (?, ?)",
            "gb.content_sha256=d.content_sha256",
            "gb.index_fingerprint=d.index_fingerprint",
            "gb.index_fingerprint=?",
            "gb.extractor_fingerprint=?",
        ]
        params: list[Any] = [
            DIGEST_CONTENT_KIND,
            FULL_REPORT_V1,
            self.index_fingerprint,
            self.extractor_fingerprint,
        ]
        if document_id is not None:
            clauses.append("d.id=?")
            params.append(document_id)
        with self.storage.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            db.execute(
                """
                UPDATE graph_builds
                SET status='stale', active=0, updated_at=?
                WHERE status='pending' AND NOT EXISTS (
                    SELECT 1 FROM documents d
                    WHERE d.id=graph_builds.document_id
                      AND d.status='ready'
                      AND d.content_sha256=graph_builds.content_sha256
                      AND d.index_fingerprint=graph_builds.index_fingerprint
                )
                """,
                (utc_now(),),
            )
            row = db.execute(
                f"""
                SELECT gb.*, d.ticker, d.analysis_date, d.analysis_cutoff,
                       d.content_kind, d.data_provenance, d.research_mode
                FROM graph_builds gb
                JOIN documents d ON d.id=gb.document_id
                WHERE {' AND '.join(clauses)}
                ORDER BY gb.created_at, gb.id
                LIMIT 1
                """,
                tuple(params),
            ).fetchone()
            if row is None:
                return None
            changed = db.execute(
                """
                UPDATE graph_builds
                SET status='building', active=0, error_code=NULL,
                    started_at=?, updated_at=?
                WHERE id=? AND status='pending'
                """,
                (utc_now(), utc_now(), row["id"]),
            ).rowcount
            return dict(row) if changed == 1 else None

    def claim_next_build(
        self, document_id: str | None = None
    ) -> dict[str, Any] | None:
        """Atomically claim one current pending build for an external worker."""

        return self._claim_build(document_id)

    def _load_build_source(
        self, build: Mapping[str, Any]
    ) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, tuple[dict[str, Any], ...]]]:
        with self.storage.connect() as db:
            document = db.execute(
                """
                SELECT d.* FROM documents d JOIN graph_builds gb ON gb.document_id=d.id
                WHERE gb.id=? AND gb.status='building' AND d.status='ready'
                  AND d.source_type='trading_digest'
                  AND d.content_kind IN (?, ?)
                  AND gb.content_sha256=d.content_sha256
                  AND gb.index_fingerprint=d.index_fingerprint
                  AND gb.extractor_fingerprint=?
                """,
                (
                    build["id"],
                    DIGEST_CONTENT_KIND,
                    FULL_REPORT_V1,
                    self.extractor_fingerprint,
                ),
            ).fetchone()
            if document is None:
                raise RuntimeError("graph_build_source_changed")
            parents = db.execute(
                """
                SELECT p.* FROM parents p
                WHERE p.document_id=? AND p.kind='page'
                  AND p.digest_section IS NOT NULL
                  AND p.section_status='complete'
                ORDER BY p.digest_section, p.start_index, p.id
                """,
                (document["id"],),
            ).fetchall()
        claims_by_section: dict[str, list[dict[str, Any]]] = {}
        for claim in self.storage.digest_claims(str(document["id"])):
            claims_by_section.setdefault(str(claim["section"]), []).append(claim)
        return (
            dict(document),
            [dict(row) for row in parents],
            {key: tuple(value) for key, value in claims_by_section.items()},
        )

    @staticmethod
    def _child_for_quote(
        db: sqlite3.Connection,
        parent_id: str,
        quote: str,
    ) -> str | None:
        row = db.execute(
            """
            SELECT id FROM children
            WHERE parent_id=? AND instr(text, ?) > 0
            ORDER BY start_index, id LIMIT 1
            """,
            (parent_id, quote),
        ).fetchone()
        return str(row["id"]) if row else None

    @staticmethod
    def _add_mention(
        prepared: _PreparedGraph,
        *,
        build_id: str,
        entity: GraphEntityInput,
        parent: Mapping[str, Any],
        surface: str,
        start_offset: int,
    ) -> None:
        end_offset = start_offset + len(surface)
        mention_id = _stable_id(
            "graph_mention",
            build_id,
            parent["id"],
            entity.id,
            start_offset,
            end_offset,
            surface,
        )
        prepared.entities[entity.id] = entity
        prepared.mentions[mention_id] = {
            "id": mention_id,
            "build_id": build_id,
            "entity_id": entity.id,
            "parent_id": parent["id"],
            "surface": surface,
            "alias_key": _canonical_key(surface),
            "start_offset": start_offset,
            "end_offset": end_offset,
        }

    def _add_assertion(
        self,
        prepared: _PreparedGraph,
        *,
        build: Mapping[str, Any],
        document: Mapping[str, Any],
        parent: Mapping[str, Any],
        value: GraphAssertionInput | Mapping[str, Any],
        default_subject: GraphEntityInput,
    ) -> None:
        assertion = GraphAssertionInput.from_value(
            value, default_subject=default_subject
        )
        parent_text = str(parent["text"])
        start_offset = parent_text.find(assertion.quote)
        if start_offset < 0:
            raise ValueError("graph evidence quote is not an exact parent substring")
        end_offset = start_offset + len(assertion.quote)
        subject = GraphEntityInput.from_value(assertion.subject or default_subject)
        object_entity = (
            GraphEntityInput.from_value(assertion.object_entity)
            if assertion.object_entity is not None
            else None
        )
        literal = None
        if object_entity is None:
            if isinstance(assertion.object_literal, str):
                literal = assertion.object_literal.strip()
            else:
                literal = json.dumps(
                    assertion.object_literal,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
            if not literal:
                raise ValueError("graph object literal is empty")
            if len(literal) > 4000:
                raise ValueError("graph object literal is too long")
        prepared.entities[subject.id] = subject
        if object_entity is not None:
            prepared.entities[object_entity.id] = object_entity
        assertion_id = _stable_id(
            "graph_assertion",
            build["id"],
            parent["id"],
            subject.id,
            assertion.predicate,
            object_entity.id if object_entity else None,
            literal,
            assertion.object_type,
            assertion.modality,
            assertion.valid_at or document.get("analysis_cutoff"),
        )
        prepared.assertions[assertion_id] = {
            "id": assertion_id,
            "build_id": build["id"],
            "document_id": document["id"],
            "subject_entity_id": subject.id,
            "predicate": assertion.predicate,
            "object_entity_id": object_entity.id if object_entity else None,
            "object_literal": literal,
            "object_type": assertion.object_type,
            "modality": assertion.modality,
            "valid_at": assertion.valid_at or document.get("analysis_cutoff"),
            "confidence": assertion.confidence,
        }
        evidence_id = _stable_id(
            "graph_evidence",
            assertion_id,
            parent["id"],
            start_offset,
            end_offset,
            assertion.quote,
        )
        prepared.evidence[evidence_id] = {
            "id": evidence_id,
            "build_id": build["id"],
            "assertion_id": assertion_id,
            "parent_id": parent["id"],
            "child_id": None,
            "quote": assertion.quote,
            "quote_sha256": hashlib.sha256(assertion.quote.encode("utf-8")).hexdigest(),
            "start_offset": start_offset,
            "end_offset": end_offset,
        }
        quote_folded = parent_text.casefold()
        for entity in (subject, object_entity):
            if entity is None:
                continue
            needle = entity.canonical_name.casefold()
            relative = quote_folded.find(needle, start_offset, end_offset)
            if relative >= 0:
                surface = parent_text[relative : relative + len(entity.canonical_name)]
                self._add_mention(
                    prepared,
                    build_id=str(build["id"]),
                    entity=entity,
                    parent=parent,
                    surface=surface,
                    start_offset=relative,
                )

    @staticmethod
    def _decision_fragment(field_name: str, value: Any) -> str:
        return (
            json.dumps(field_name, ensure_ascii=False)
            + ":"
            + json.dumps(
                value,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
        )

    def _add_decision_assertions(
        self,
        prepared: _PreparedGraph,
        *,
        build: Mapping[str, Any],
        document: Mapping[str, Any],
        parents: Sequence[Mapping[str, Any]],
        default_subject: GraphEntityInput,
    ) -> None:
        if document.get("content_kind") != FULL_REPORT_V1:
            return
        payload = self.storage.get_full_report_payload(str(document["id"]))
        if payload is None or payload.get("decision_status") != "complete":
            return
        try:
            decision = json.loads(str(payload["decision_json"]))
        except (TypeError, ValueError):
            return
        decision_parents = [
            parent
            for parent in parents
            if parent.get("digest_section") == DECISION_STRUCTURED_SECTION
        ]
        if not decision_parents:
            return
        portfolio = decision.get("portfolio") or {}
        trading = decision.get("trading") or {}
        specs: list[tuple[str, Any, str, Any, str | None]] = []
        if portfolio.get("rating") is not None:
            specs.append(
                ("rating", portfolio["rating"], "RECOMMENDS", portfolio["rating"], "rating")
            )
        if trading.get("action") is not None:
            specs.append(
                ("action", trading["action"], "TRADING_ACTION", trading["action"], "action")
            )
        if portfolio.get("price_target") is not None:
            target = {
                "value": portfolio["price_target"],
                "currency": portfolio.get("price_target_currency"),
            }
            specs.append(
                ("price_target", portfolio["price_target"], "TARGET_PRICE", target, "price")
            )
        for field_name, raw_value, predicate, literal, object_type in specs:
            quote = self._decision_fragment(field_name, raw_value)
            parent = next(
                (item for item in decision_parents if quote in str(item["text"])),
                None,
            )
            if parent is None:
                continue
            self._add_assertion(
                prepared,
                build=build,
                document=document,
                parent=parent,
                value=GraphAssertionInput(
                    subject=default_subject,
                    predicate=predicate,
                    object_literal=literal,
                    object_type=object_type,
                    quote=quote,
                    confidence=1.0,
                ),
                default_subject=default_subject,
            )

    def _base_prepared(
        self,
        *,
        build: Mapping[str, Any],
        document: Mapping[str, Any],
        parents: Sequence[Mapping[str, Any]],
    ) -> tuple[_PreparedGraph, GraphEntityInput]:
        prepared = _PreparedGraph()
        ticker = str(document.get("ticker") or "").upper()
        default_subject = GraphEntityInput.from_value(
            {
                "entity_type": "security",
                "canonical_key": ticker,
                "canonical_name": ticker,
            }
        )
        prepared.entities[default_subject.id] = default_subject
        for parent in parents:
            text = str(parent["text"])
            for match in re.finditer(re.escape(ticker), text, flags=re.IGNORECASE):
                self._add_mention(
                    prepared,
                    build_id=str(build["id"]),
                    entity=default_subject,
                    parent=parent,
                    surface=match.group(0),
                    start_offset=match.start(),
                )
        self._add_decision_assertions(
            prepared,
            build=build,
            document=document,
            parents=parents,
            default_subject=default_subject,
        )
        return prepared, default_subject

    def _apply_extractor_result(
        self,
        prepared: _PreparedGraph,
        result: GraphExtractionResult,
        *,
        build: Mapping[str, Any],
        document: Mapping[str, Any],
        parent: Mapping[str, Any],
        default_subject: GraphEntityInput,
    ) -> None:
        if isinstance(result, (str, bytes, bytearray)) or not isinstance(
            result, Sequence
        ):
            raise ValueError("graph extractor must return an assertion list")
        for value in result:
            self._add_assertion(
                prepared,
                build=build,
                document=document,
                parent=parent,
                value=value,
                default_subject=default_subject,
            )

    def _write_prepared(
        self,
        *,
        build: Mapping[str, Any],
        prepared: _PreparedGraph,
    ) -> dict[str, Any]:
        now = utc_now()
        with self.storage.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            current = db.execute(
                """
                SELECT gb.*, d.status AS document_status,
                       d.content_sha256 AS document_content_sha256,
                       d.index_fingerprint AS document_index_fingerprint
                FROM graph_builds gb JOIN documents d ON d.id=gb.document_id
                WHERE gb.id=?
                """,
                (build["id"],),
            ).fetchone()
            if (
                current is None
                or current["status"] != "building"
                or current["document_status"] != "ready"
                or current["content_sha256"] != current["document_content_sha256"]
                or current["index_fingerprint"] != current["document_index_fingerprint"]
                or current["extractor_fingerprint"] != self.extractor_fingerprint
            ):
                raise RuntimeError("graph_build_source_changed")
            db.execute("DELETE FROM graph_evidence WHERE build_id=?", (build["id"],))
            db.execute("DELETE FROM graph_assertions WHERE build_id=?", (build["id"],))
            db.execute("DELETE FROM graph_mentions WHERE build_id=?", (build["id"],))
            for entity in prepared.entities.values():
                db.execute(
                    """
                    INSERT INTO graph_entities(
                        id, entity_type, canonical_key, canonical_name,
                        created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    ON CONFLICT(id) DO UPDATE SET
                        canonical_name=excluded.canonical_name,
                        updated_at=excluded.updated_at
                    """,
                    (
                        entity.id,
                        entity.entity_type,
                        entity.canonical_key,
                        entity.canonical_name,
                        now,
                        now,
                    ),
                )
            db.executemany(
                """
                INSERT INTO graph_mentions(
                    id, build_id, entity_id, parent_id, surface, alias_key,
                    start_offset, end_offset, created_at
                ) VALUES (
                    :id, :build_id, :entity_id, :parent_id, :surface, :alias_key,
                    :start_offset, :end_offset, :created_at
                )
                """,
                [dict(item, created_at=now) for item in prepared.mentions.values()],
            )
            for item in prepared.evidence.values():
                parent = db.execute(
                    "SELECT text FROM parents WHERE id=? AND document_id=?",
                    (item["parent_id"], current["document_id"]),
                ).fetchone()
                if (
                    parent is None
                    or str(parent["text"])[item["start_offset"] : item["end_offset"]]
                    != item["quote"]
                ):
                    raise ValueError("graph evidence changed before commit")
                item["child_id"] = self._child_for_quote(
                    db, str(item["parent_id"]), str(item["quote"])
                )
            db.executemany(
                """
                INSERT INTO graph_assertions(
                    id, build_id, document_id, subject_entity_id, predicate,
                    object_entity_id, object_literal, object_type, modality,
                    valid_at, confidence, created_at
                ) VALUES (
                    :id, :build_id, :document_id, :subject_entity_id, :predicate,
                    :object_entity_id, :object_literal, :object_type, :modality,
                    :valid_at, :confidence, :created_at
                )
                """,
                [dict(item, created_at=now) for item in prepared.assertions.values()],
            )
            db.executemany(
                """
                INSERT INTO graph_evidence(
                    id, build_id, assertion_id, parent_id, child_id, quote,
                    quote_sha256, start_offset, end_offset, created_at
                ) VALUES (
                    :id, :build_id, :assertion_id, :parent_id, :child_id, :quote,
                    :quote_sha256, :start_offset, :end_offset, :created_at
                )
                """,
                [dict(item, created_at=now) for item in prepared.evidence.values()],
            )
            db.execute(
                """
                UPDATE graph_builds
                SET status='stale', active=0, updated_at=?
                WHERE document_id=? AND id<>? AND status<>'stale'
                """,
                (now, current["document_id"], build["id"]),
            )
            db.execute(
                """
                UPDATE graph_builds
                SET status='ready', active=1, error_code=NULL,
                    completed_at=?, updated_at=?
                WHERE id=? AND status='building'
                """,
                (now, now, build["id"]),
            )
            for entity_id, entity in prepared.entities.items():
                alias_rows = db.execute(
                    """
                    SELECT DISTINCT gm.surface
                    FROM graph_mentions gm
                    JOIN graph_builds gb ON gb.id=gm.build_id
                    JOIN documents d ON d.id=gb.document_id
                    WHERE gm.entity_id=? AND gb.active=1 AND gb.status='ready'
                      AND d.status='ready'
                      AND gb.content_sha256=d.content_sha256
                      AND gb.index_fingerprint=d.index_fingerprint
                    ORDER BY gm.surface
                    """,
                    (entity_id,),
                ).fetchall()
                aliases = " ".join(str(row["surface"]) for row in alias_rows)
                db.execute("DELETE FROM graph_entity_fts WHERE entity_id=?", (entity_id,))
                db.execute(
                    """
                    INSERT INTO graph_entity_fts(
                        entity_id, canonical_key, canonical_name, aliases
                    ) VALUES (?, ?, ?, ?)
                    """,
                    (
                        entity_id,
                        normalise_fts_text(entity.canonical_key),
                        normalise_fts_text(entity.canonical_name),
                        normalise_fts_text(aliases),
                    ),
                )
        return {
            "build_id": build["id"],
            "document_id": build["document_id"],
            "entities": len(prepared.entities),
            "mentions": len(prepared.mentions),
            "assertions": len(prepared.assertions),
            "evidence": len(prepared.evidence),
            "status": "ready",
        }

    def _fail_build(self, build_id: str, reason: str) -> None:
        with self.storage.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            db.execute("DELETE FROM graph_evidence WHERE build_id=?", (build_id,))
            db.execute("DELETE FROM graph_assertions WHERE build_id=?", (build_id,))
            db.execute("DELETE FROM graph_mentions WHERE build_id=?", (build_id,))
            db.execute(
                """
                UPDATE graph_builds
                SET status='failed', active=0, retry_count=retry_count+1,
                    error_code=?, completed_at=?, updated_at=?
                WHERE id=? AND status='building'
                """,
                (reason[:120], utc_now(), utc_now(), build_id),
            )

    def _build_claim_sync(self, build: Mapping[str, Any]) -> dict[str, Any]:
        document, parents, claims = self._load_build_source(build)
        prepared, default_subject = self._base_prepared(
            build=build, document=document, parents=parents
        )
        if self.extractor is not None:
            for parent in parents:
                if parent.get("digest_section") == DECISION_STRUCTURED_SECTION:
                    continue
                result = self.extractor(
                    GraphExtractionInput(
                        document=document,
                        parent=parent,
                        claims=claims.get(str(parent["digest_section"]), ()),
                        default_subject=default_subject,
                    )
                )
                if inspect.isawaitable(result):
                    raise RuntimeError("async_graph_extractor_requires_async_build")
                self._apply_extractor_result(
                    prepared,
                    result,
                    build=build,
                    document=document,
                    parent=parent,
                    default_subject=default_subject,
                )
        return self._write_prepared(build=build, prepared=prepared)

    async def _call_extractor_async(
        self, value: GraphExtractionInput
    ) -> GraphExtractionResult:
        assert self.extractor is not None

        async def invoke() -> GraphExtractionResult:
            result = self.extractor(value)
            return await result if inspect.isawaitable(result) else result

        if self.generation_lock is None:
            return await invoke()
        async with self.generation_lock:
            return await invoke()

    async def _build_claim_async(self, build: Mapping[str, Any]) -> dict[str, Any]:
        document, parents, claims = self._load_build_source(build)
        prepared, default_subject = self._base_prepared(
            build=build, document=document, parents=parents
        )
        if self.extractor is not None:
            for parent in parents:
                if parent.get("digest_section") == DECISION_STRUCTURED_SECTION:
                    continue
                result = await self._call_extractor_async(
                    GraphExtractionInput(
                        document=document,
                        parent=parent,
                        claims=claims.get(str(parent["digest_section"]), ()),
                        default_subject=default_subject,
                    )
                )
                self._apply_extractor_result(
                    prepared,
                    result,
                    build=build,
                    document=document,
                    parent=parent,
                    default_subject=default_subject,
                )
                # The shared model lock is intentionally released after every
                # parent batch so interactive chat can acquire it between work.
                await asyncio.sleep(0)
        return self._write_prepared(build=build, prepared=prepared)

    def build_document(self, document_id: str) -> dict[str, Any]:
        self.reconcile(document_id, apply=True)
        build = self._claim_build(document_id)
        if build is None:
            with self.storage.connect() as db:
                row = db.execute(
                    """
                    SELECT * FROM graph_builds
                    WHERE document_id=? AND status='ready' AND active=1
                    ORDER BY completed_at DESC LIMIT 1
                    """,
                    (document_id,),
                ).fetchone()
            if row is None:
                raise RuntimeError("graph_build_not_claimable")
            return {"build_id": row["id"], "document_id": document_id, "status": "ready"}
        return self.build_claimed(build)

    def build_claimed(self, build: Mapping[str, Any]) -> dict[str, Any]:
        """Build a record returned by :meth:`claim_next_build`."""

        try:
            return self._build_claim_sync(build)
        except Exception as exc:
            self._fail_build(str(build["id"]), type(exc).__name__)
            raise

    async def build_document_async(self, document_id: str) -> dict[str, Any]:
        self.reconcile(document_id, apply=True)
        build = self._claim_build(document_id)
        if build is None:
            with self.storage.connect() as db:
                row = db.execute(
                    "SELECT * FROM graph_builds WHERE document_id=? "
                    "AND status='ready' AND active=1 LIMIT 1",
                    (document_id,),
                ).fetchone()
            if row is None:
                raise RuntimeError("graph_build_not_claimable")
            return {"build_id": row["id"], "document_id": document_id, "status": "ready"}
        return await self.build_claimed_async(build)

    async def build_claimed_async(
        self, build: Mapping[str, Any]
    ) -> dict[str, Any]:
        """Async extractor variant for an externally claimed build."""

        try:
            return await self._build_claim_async(build)
        except Exception as exc:
            self._fail_build(str(build["id"]), type(exc).__name__)
            raise

    def health(self) -> dict[str, Any]:
        with self.storage.connect() as db:
            row = db.execute(
                """
                SELECT
                    count(*) AS eligible,
                    coalesce(sum(CASE WHEN EXISTS (
                        SELECT 1 FROM graph_builds gb
                        WHERE gb.document_id=d.id AND gb.status='ready'
                          AND gb.active=1
                          AND gb.content_sha256=d.content_sha256
                          AND gb.index_fingerprint=d.index_fingerprint
                          AND gb.extractor_fingerprint=?
                    ) THEN 1 ELSE 0 END), 0) AS covered
                FROM documents d
                WHERE d.status='ready' AND d.source_type='trading_digest'
                  AND d.content_kind IN (?, ?)
                  AND d.index_fingerprint=?
                """,
                (
                    self.extractor_fingerprint,
                    DIGEST_CONTENT_KIND,
                    FULL_REPORT_V1,
                    self.index_fingerprint,
                ),
            ).fetchone()
            states = db.execute(
                """
                SELECT status, count(*) AS total FROM graph_builds
                WHERE extractor_fingerprint=? GROUP BY status
                """,
                (self.extractor_fingerprint,),
            ).fetchall()
        counts = {str(item["status"]): int(item["total"]) for item in states}
        eligible = int(row["eligible"] if row else 0)
        covered = int(row["covered"] if row else 0)
        return {
            "schema_version": GRAPH_SCHEMA_VERSION,
            "extractor_fingerprint": self.extractor_fingerprint,
            "eligible": eligible,
            "covered": covered,
            "coverage": (covered / eligible) if eligible else 1.0,
            "backlog": counts.get("pending", 0) + counts.get("building", 0),
            "failures": counts.get("failed", 0),
            "counts": counts,
        }

    def reset_interrupted_builds(self) -> int:
        """Return builds owned by a stopped process to the local work queue."""

        with self.storage.connect() as db:
            return db.execute(
                """
                UPDATE graph_builds
                SET status='pending', active=0, error_code='interrupted',
                    started_at=NULL, updated_at=?
                WHERE status='building' AND extractor_fingerprint=?
                """,
                (utc_now(), self.extractor_fingerprint),
            ).rowcount

    def wake(self) -> None:
        self._wake_event.set()

    async def _worker(self) -> None:
        while not self._stop_event.is_set():
            build = self._claim_build()
            if build is None:
                # Ingestion and promotion are independent of the graph worker.
                # Reconcile while idle so documents that became ready after
                # startup are discovered without coupling graph failures to the
                # primary indexing transaction.
                self.reconcile(apply=True)
                build = self._claim_build()
            if build is None:
                self._wake_event.clear()
                try:
                    await asyncio.wait_for(self._wake_event.wait(), timeout=1.0)
                except TimeoutError:
                    pass
                continue
            try:
                await self._build_claim_async(build)
            except asyncio.CancelledError:
                self._fail_build(str(build["id"]), "cancelled")
                raise
            except Exception as exc:
                self._fail_build(str(build["id"]), type(exc).__name__)
                if self.logger is not None:
                    self.logger.error(
                        "graph build failed",
                        extra={
                            "event": "graph_build_failed",
                            "reason_code": type(exc).__name__,
                        },
                    )

    async def start(self) -> None:
        if self._tasks:
            return
        self._stop_event.clear()
        self.reset_interrupted_builds()
        self.reconcile(apply=True)
        self._tasks = [
            asyncio.create_task(self._worker(), name=f"graph-worker-{index}")
            for index in range(self.worker_concurrency)
        ]
        self.wake()

    async def stop(self) -> None:
        self._stop_event.set()
        self.wake()
        tasks, self._tasks = self._tasks, []
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)


class GraphRetriever:
    """Bounded 1–2 hop traversal that returns raw parent evidence records."""

    def __init__(
        self,
        storage: SQLiteStorage,
        *,
        extractor_fingerprint: str | None = None,
        max_hops: int = 2,
        max_entities: int = 100,
        max_edges_per_seed: int = 25,
        max_candidates: int = 20,
    ):
        self.storage = storage
        self.extractor_fingerprint = extractor_fingerprint
        self.max_hops = min(2, max(1, int(max_hops)))
        self.max_entities = min(100, max(1, int(max_entities)))
        self.max_edges_per_seed = min(25, max(1, int(max_edges_per_seed)))
        self.max_candidates = min(20, max(1, int(max_candidates)))

    @staticmethod
    def _active_build_sql(*, build_alias: str = "gb", document_alias: str = "d") -> str:
        return (
            f"{build_alias}.active=1 AND {build_alias}.status='ready' "
            f"AND {document_alias}.status='ready' "
            f"AND {document_alias}.source_type='trading_digest' "
            f"AND {document_alias}.content_kind IN ('{DIGEST_CONTENT_KIND}', '{FULL_REPORT_V1}') "
            f"AND {build_alias}.content_sha256={document_alias}.content_sha256 "
            f"AND {build_alias}.index_fingerprint={document_alias}.index_fingerprint"
        )

    @staticmethod
    def _query_alias_keys(query: str) -> list[str]:
        tokens = FTS_TOKEN_RE.findall(normalise_fts_text(query))[:16]
        values = {_canonical_key(query), *tokens}
        for width in range(2, min(4, len(tokens)) + 1):
            for start in range(0, len(tokens) - width + 1):
                values.add(" ".join(tokens[start : start + width]))
        values.discard("")
        return sorted(values, key=lambda value: (-len(value), value))[:64]

    def _resolve_entities(self, query: str) -> list[str]:
        keys = self._query_alias_keys(query)
        match_query = build_fts_query(query)
        candidates: list[str] = []
        current_name_or_alias_matches: set[str] = set()
        with self.storage.connect() as db:
            if keys:
                placeholders = ",".join("?" for _ in keys)
                direct_fingerprint_clause = ""
                direct_params: list[Any] = list(keys)
                if self.extractor_fingerprint is not None:
                    direct_fingerprint_clause = " AND gb.extractor_fingerprint=?"
                    direct_params.append(self.extractor_fingerprint)
                rows = db.execute(
                    f"""
                    SELECT e.id, e.canonical_key, e.entity_type
                    FROM graph_entities e
                    WHERE e.canonical_key IN ({placeholders})
                      AND EXISTS (
                          SELECT 1 FROM graph_assertions ga
                          JOIN graph_builds gb ON gb.id=ga.build_id
                          JOIN documents d ON d.id=gb.document_id
                          WHERE (
                              ga.subject_entity_id=e.id
                              OR ga.object_entity_id=e.id
                          )
                            AND {self._active_build_sql()}
                            {direct_fingerprint_clause}
                      )
                    """,
                    tuple(direct_params),
                ).fetchall()
                direct_by_key: dict[str, list[sqlite3.Row]] = {}
                for row in rows:
                    direct_by_key.setdefault(
                        str(row["canonical_key"]), []
                    ).append(row)
                direct: list[str] = []
                for key in keys:
                    key_rows = direct_by_key.get(key, [])
                    ticker_rows = [
                        row
                        for row in key_rows
                        if row["entity_type"] == "security"
                    ]
                    selected = ticker_rows or key_rows
                    direct.extend(str(row["id"]) for row in selected)
                candidates.extend(direct)
                current_name_or_alias_matches.update(direct)
                alias_keys = [key for key in keys if key not in direct_by_key]
                aliases: Sequence[sqlite3.Row] = ()
                if alias_keys:
                    alias_placeholders = ",".join("?" for _ in alias_keys)
                    alias_fingerprint_clause = ""
                    alias_params: list[Any] = list(alias_keys)
                    if self.extractor_fingerprint is not None:
                        alias_fingerprint_clause = " AND gb.extractor_fingerprint=?"
                        alias_params.append(self.extractor_fingerprint)
                    aliases = db.execute(
                        f"""
                        SELECT gm.alias_key, min(gm.entity_id) AS entity_id,
                               count(DISTINCT gm.entity_id) AS entity_count
                        FROM graph_mentions gm
                        JOIN graph_builds gb ON gb.id=gm.build_id
                        JOIN documents d ON d.id=gb.document_id
                        WHERE gm.alias_key IN ({alias_placeholders})
                          AND {self._active_build_sql()}
                          {alias_fingerprint_clause}
                        GROUP BY gm.alias_key HAVING entity_count=1
                        """,
                        tuple(alias_params),
                    ).fetchall()
                alias_ids = [str(row["entity_id"]) for row in aliases]
                candidates.extend(alias_ids)
                current_name_or_alias_matches.update(alias_ids)
            if match_query:
                try:
                    rows = db.execute(
                        """
                        SELECT entity_id, bm25(graph_entity_fts) AS rank
                        FROM graph_entity_fts
                        WHERE graph_entity_fts MATCH ?
                        ORDER BY rank LIMIT 20
                        """,
                        (match_query,),
                    ).fetchall()
                    # FTS is a candidate generator only.  Requiring an exact
                    # current canonical/alias match prevents an alias from a
                    # stale build seeding traversal for an otherwise-live
                    # global entity.
                    candidates.extend(
                        str(row["entity_id"])
                        for row in rows
                        if str(row["entity_id"]) in current_name_or_alias_matches
                    )
                except sqlite3.OperationalError:
                    pass
            unique = list(dict.fromkeys(candidates))[: self.max_entities]
            if not unique:
                return []
            placeholders = ",".join("?" for _ in unique)
            fingerprint_clause = ""
            params: list[Any] = list(unique)
            if self.extractor_fingerprint is not None:
                fingerprint_clause = " AND gb.extractor_fingerprint=?"
                params.append(self.extractor_fingerprint)
            active = db.execute(
                f"""
                SELECT e.id FROM graph_entities e
                WHERE e.id IN ({placeholders}) AND EXISTS (
                    SELECT 1 FROM graph_assertions ga
                    JOIN graph_builds gb ON gb.id=ga.build_id
                    JOIN documents d ON d.id=gb.document_id
                    WHERE (ga.subject_entity_id=e.id OR ga.object_entity_id=e.id)
                      AND {self._active_build_sql()}
                      {fingerprint_clause}
                )
                """,
                tuple(params),
            ).fetchall()
        active_ids = {str(row["id"]) for row in active}
        return [item for item in unique if item in active_ids]

    def _edge_rows(
        self,
        entity_ids: Sequence[str],
        *,
        document_ids: Sequence[str] | None,
        filters: Mapping[str, Any] | None,
    ) -> list[dict[str, Any]]:
        if not entity_ids or (document_ids is not None and not document_ids):
            return []
        placeholders = ",".join("?" for _ in entity_ids)
        clauses = [
            f"(ga.subject_entity_id IN ({placeholders}) "
            f"OR ga.object_entity_id IN ({placeholders}))",
            self._active_build_sql(),
            "p.section_status='complete'",
        ]
        params: list[Any] = [*entity_ids, *entity_ids]
        if self.extractor_fingerprint is not None:
            clauses.append("gb.extractor_fingerprint=?")
            params.append(self.extractor_fingerprint)
        if document_ids is not None:
            doc_placeholders = ",".join("?" for _ in document_ids)
            clauses.append(f"d.id IN ({doc_placeholders})")
            params.extend(document_ids)
        document_clauses, document_params = _document_filter_sql(
            dict(filters or {}), alias="d"
        )
        clauses.extend(document_clauses)
        params.extend(document_params)
        digest_sections = list(dict.fromkeys((filters or {}).get("digest_sections") or []))
        if digest_sections:
            section_placeholders = ",".join("?" for _ in digest_sections)
            clauses.append(f"p.digest_section IN ({section_placeholders})")
            params.extend(digest_sections)
        report_stages = list(dict.fromkeys((filters or {}).get("report_stages") or []))
        if report_stages:
            stage_placeholders = ",".join("?" for _ in report_stages)
            clauses.append(f"p.report_stage IN ({stage_placeholders})")
            params.extend(report_stages)
        if (filters or {}).get("latest_only"):
            latest_scope = ""
            if document_ids is not None:
                latest_scope = f"AND newer.id IN ({doc_placeholders})"
                params.extend(document_ids)
            clauses.append(
                f"""
                NOT EXISTS (
                    SELECT 1 FROM documents newer
                    WHERE newer.status='ready'
                      AND newer.source_type='trading_digest'
                      AND newer.content_kind IN ('daily_digest_v1', 'full_report_v1')
                      AND newer.ticker=d.ticker
                      AND COALESCE(newer.analysis_cutoff, newer.analysis_date)
                          > COALESCE(d.analysis_cutoff, d.analysis_date)
                      {latest_scope}
                )
                """
            )
        limit = min(
            2500,
            max(1, len(entity_ids)) * self.max_edges_per_seed * 2,
        )
        with self.storage.connect() as db:
            rows = db.execute(
                f"""
                SELECT ga.id AS assertion_id, ga.subject_entity_id,
                       ga.predicate, ga.object_entity_id, ga.object_literal,
                       ga.object_type, ga.modality, ga.valid_at, ga.confidence,
                       ge.quote AS evidence_quote,
                       p.id AS parent_id, p.document_id, p.page, p.page_end,
                       p.kind, p.digest_section, p.report_stage,
                       p.section_status, p.start_index, p.text,
                       d.filename, d.source_type, d.ticker, d.analysis_date,
                       d.analysis_cutoff, d.batch_id, d.liquidity_rank,
                       d.digest_status, d.research_mode, d.data_provenance,
                       d.content_kind, d.report_schema, d.identity_hash,
                       d.digest_hash, d.parent_identity_hash,
                       d.expected_digest_hash, d.full_report_hash,
                       d.pipeline_version, d.execution_generation,
                       d.contains_decision_content,
                       subject.canonical_name AS subject_name,
                       object.canonical_name AS object_name
                FROM graph_assertions ga
                JOIN graph_builds gb ON gb.id=ga.build_id
                JOIN documents d ON d.id=ga.document_id
                JOIN graph_evidence ge ON ge.assertion_id=ga.id
                JOIN parents p ON p.id=ge.parent_id AND p.document_id=d.id
                JOIN graph_entities subject ON subject.id=ga.subject_entity_id
                LEFT JOIN graph_entities object ON object.id=ga.object_entity_id
                WHERE {' AND '.join(clauses)}
                ORDER BY ga.confidence DESC,
                         COALESCE(d.analysis_cutoff, d.analysis_date) DESC,
                         ga.id, p.id
                LIMIT ?
                """,
                (*params, limit),
            ).fetchall()
        return [dict(row) for row in rows]

    def search(
        self,
        query: str,
        *,
        k: int = 20,
        document_ids: Sequence[str] | None = None,
        filters: Mapping[str, Any] | None = None,
        max_hops: int | None = None,
    ) -> list[dict[str, Any]]:
        seeds = self._resolve_entities(query)
        if not seeds:
            return []
        hop_limit = self.max_hops if max_hops is None else min(2, max(1, int(max_hops)))
        query_terms = set(FTS_TOKEN_RE.findall(normalise_fts_text(query)))
        frontier = list(seeds)
        visited_entities = set(seeds)
        visited_assertions: set[str] = set()
        parent_candidates: dict[str, dict[str, Any]] = {}
        for hop in range(1, hop_limit + 1):
            rows = self._edge_rows(
                frontier, document_ids=document_ids, filters=filters
            )
            per_seed_count = {entity_id: 0 for entity_id in frontier}
            next_frontier: list[str] = []
            for row in rows:
                assertion_id = str(row["assertion_id"])
                if assertion_id in visited_assertions:
                    continue
                touched = [
                    entity_id
                    for entity_id in frontier
                    if entity_id
                    in {row["subject_entity_id"], row["object_entity_id"]}
                ]
                if not touched or all(
                    per_seed_count[item] >= self.max_edges_per_seed for item in touched
                ):
                    continue
                for item in touched:
                    per_seed_count[item] += 1
                visited_assertions.add(assertion_id)
                for entity_id in (
                    row["subject_entity_id"],
                    row["object_entity_id"],
                ):
                    if entity_id and entity_id not in visited_entities:
                        if len(visited_entities) >= self.max_entities:
                            break
                        visited_entities.add(str(entity_id))
                        next_frontier.append(str(entity_id))
                searchable = normalise_fts_text(
                    " ".join(
                        str(row.get(field) or "")
                        for field in (
                            "subject_name",
                            "predicate",
                            "object_name",
                            "object_literal",
                            "evidence_quote",
                        )
                    )
                )
                overlap = len(query_terms & set(FTS_TOKEN_RE.findall(searchable)))
                score = float(row["confidence"]) * (1.0 / hop) + 0.05 * overlap
                parent_id = str(row["parent_id"])
                candidate = parent_candidates.get(parent_id)
                graph_path = {
                    "hop": hop,
                    "predicate": row["predicate"],
                    "subject": row["subject_name"],
                    "object": row["object_name"] or row["object_literal"],
                    "evidence_quote": row["evidence_quote"],
                }
                if candidate is None:
                    candidate = {
                        key: value
                        for key, value in row.items()
                        if key
                        not in {
                            "assertion_id",
                            "subject_entity_id",
                            "predicate",
                            "object_entity_id",
                            "object_literal",
                            "object_type",
                            "modality",
                            "valid_at",
                            "confidence",
                            "evidence_quote",
                            "subject_name",
                            "object_name",
                        }
                    }
                    candidate.update(
                        {
                            "score": 0.0,
                            "graph_score": score,
                            "retrieval_channel": "graph",
                            "graph_paths": [graph_path],
                        }
                    )
                    parent_candidates[parent_id] = candidate
                else:
                    candidate["graph_score"] = max(candidate["graph_score"], score)
                    candidate["graph_paths"].append(graph_path)
            frontier = list(dict.fromkeys(next_frontier))
            if not frontier:
                break
        ordered = sorted(
            parent_candidates.values(),
            key=lambda item: (
                str(item.get("analysis_cutoff") or item.get("analysis_date") or ""),
                str(item["parent_id"]),
            ),
            reverse=True,
        )
        ordered.sort(key=lambda item: float(item["graph_score"]), reverse=True)
        return ordered[: min(max(1, int(k)), self.max_candidates)]
