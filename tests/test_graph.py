from __future__ import annotations

import asyncio
import json
import sqlite3

import pytest

from rag_app.graph import (
    GRAPH_ENTITY_TYPES,
    GRAPH_PREDICATES,
    GraphAssertionInput,
    GraphEntityInput,
    GraphRetriever,
    GraphService,
)
from rag_app.storage import SQLiteStorage


INDEX_FINGERPRINT = "index-v1"
EXTRACTOR_FINGERPRINT = "extractor-v1"


def ready_digest(
    storage: SQLiteStorage,
    *,
    document_id: str,
    ticker: str,
    text: str,
    analysis_date: str = "2026-08-21",
    content_sha256: str | None = None,
    content_kind: str = "daily_digest_v1",
    section: str = "risks_unknowns",
    decision: dict | None = None,
) -> str:
    content_hash = content_sha256 or f"content-{document_id}"
    parent_id = f"parent-{document_id}"
    child_id = f"child-{document_id}"
    now = "2026-08-21T10:00:00+00:00"
    with storage.connect() as db:
        db.execute(
            """
            INSERT INTO documents(
                id, filename, sha256, stored_path, size_bytes, page_count,
                status, index_fingerprint, collection_name, source_type,
                source_key, content_sha256, ticker, analysis_date,
                analysis_cutoff, content_kind, report_schema,
                research_mode, data_provenance, created_at, updated_at
            ) VALUES (
                ?, ?, ?, '', ?, 0, 'ready', ?, 'test', 'trading_digest',
                ?, ?, ?, ?, ?, ?, ?, 'eod', 'eod_cutoff', ?, ?
            )
            """,
            (
                document_id,
                f"{ticker}.digest",
                f"sha-{document_id}",
                len(text.encode()),
                INDEX_FINGERPRINT,
                f"source-{document_id}",
                content_hash,
                ticker,
                analysis_date,
                f"{analysis_date}T16:45:00+07:00",
                content_kind,
                content_kind,
                now,
                now,
            ),
        )
        db.execute(
            """
            INSERT INTO digest_sections(
                document_id, section, position, report_stage,
                section_status, text
            ) VALUES (?, ?, 0, 'risk', 'complete', ?)
            """,
            (document_id, section, text),
        )
        db.execute(
            """
            INSERT INTO parents(
                id, document_id, page, page_end, kind, digest_section,
                report_stage, section_status, start_index, text
            ) VALUES (?, ?, 0, 0, 'page', ?, 'risk', 'complete', 0, ?)
            """,
            (parent_id, document_id, section, text),
        )
        db.execute(
            """
            INSERT INTO children(
                id, parent_id, document_id, page, page_end, kind,
                digest_section, report_stage, section_status, start_index, text
            ) VALUES (?, ?, ?, 0, 0, 'page', ?, 'risk', 'complete', 0, ?)
            """,
            (child_id, parent_id, document_id, section, text),
        )
        if decision is not None:
            decision_json = json.dumps(
                decision,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            db.execute(
                """
                INSERT INTO full_report_payloads(
                    document_id, canonical_json, canonical_sha256,
                    decision_json, decision_sha256, decision_status,
                    projection_version, decision_index_version,
                    created_at, updated_at
                ) VALUES (?, '{}', 'canonical', ?, 'decision', 'complete', 2, 2, ?, ?)
                """,
                (document_id, decision_json, now, now),
            )
    return parent_id


@pytest.fixture
def storage(settings) -> SQLiteStorage:
    value = SQLiteStorage(settings.sqlite_path)
    value.initialise()
    return value


def relation_extractor(value):
    text = value.parent["text"]
    if "JPY" not in text:
        return []
    return [
        GraphAssertionInput(
            predicate="EXPOSED_TO",
            quote=text,
            object_entity=GraphEntityInput(
                entity_type="currency",
                canonical_key="JPY",
                canonical_name="JPY",
            ),
        )
    ]


def service(storage, extractor=relation_extractor) -> GraphService:
    return GraphService(
        storage,
        index_fingerprint=INDEX_FINGERPRINT,
        extractor_fingerprint=EXTRACTOR_FINGERPRINT,
        extractor=extractor,
    )


def test_schema_v8_graph_migration_is_additive_and_idempotent(storage, settings):
    storage.initialise()
    with sqlite3.connect(settings.sqlite_path) as db:
        version = db.execute(
            "SELECT value FROM app_meta WHERE key='schema_version'"
        ).fetchone()[0]
        graph_version = db.execute(
            "SELECT value FROM app_meta WHERE key='graph_schema_version'"
        ).fetchone()[0]
        tables = {
            row[0]
            for row in db.execute(
                "SELECT name FROM sqlite_master WHERE type IN ('table', 'view')"
            )
        }
    assert version == "8"
    assert graph_version == "1"
    assert {
        "graph_builds",
        "graph_entities",
        "graph_mentions",
        "graph_assertions",
        "graph_evidence",
        "digest_claims",
        "graph_entity_fts",
    } <= tables


def test_schema_7_metadata_migrates_to_graph_v1_idempotently(settings):
    settings.sqlite_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(settings.sqlite_path) as db:
        db.execute(
            "CREATE TABLE app_meta(key TEXT PRIMARY KEY, value TEXT NOT NULL)"
        )
        db.execute(
            "INSERT INTO app_meta(key, value) VALUES ('schema_version', '7')"
        )

    migrated = SQLiteStorage(settings.sqlite_path)
    migrated.initialise()
    migrated.initialise()
    with sqlite3.connect(settings.sqlite_path) as db:
        assert db.execute(
            "SELECT value FROM app_meta WHERE key='schema_version'"
        ).fetchone()[0] == "8"
        assert db.execute(
            "SELECT value FROM app_meta WHERE key='graph_schema_version'"
        ).fetchone()[0] == "1"
        assert db.execute(
            "SELECT count(*) FROM graph_builds"
        ).fetchone()[0] == 0


def test_entity_and_predicate_allowlists_match_v1_contract():
    assert {"security", "currency", "risk", "catalyst", "concept"} <= GRAPH_ENTITY_TYPES
    assert {"EXPOSED_TO", "HAS_RISK", "RECOMMENDS", "TARGET_PRICE"} <= GRAPH_PREDICATES


def test_latest_policy_is_scoped_and_preserves_pdf_and_same_cutoff(storage):
    ready_digest(
        storage,
        document_id="old-fpt",
        ticker="FPT",
        text="FPT bản cũ.",
        analysis_date="2026-08-20",
    )
    ready_digest(
        storage,
        document_id="new-fpt-digest",
        ticker="FPT",
        text="FPT bản mới.",
        analysis_date="2026-08-21",
    )
    ready_digest(
        storage,
        document_id="new-fpt-full",
        ticker="FPT",
        text="FPT Full Analysis cùng cutoff.",
        analysis_date="2026-08-21",
        content_kind="full_report_v1",
    )
    with storage.connect() as db:
        db.execute(
            """
            INSERT INTO documents(
                id, filename, sha256, stored_path, size_bytes, page_count,
                status, source_type, content_kind, report_schema,
                created_at, updated_at
            ) VALUES (
                'pdf', 'paper.pdf', 'sha-pdf', '/tmp/paper.pdf', 1, 1,
                'ready', 'pdf', 'pdf', 'pdf', 'now', 'now'
            )
            """
        )

    allowed = ["pdf", "old-fpt", "new-fpt-digest", "new-fpt-full"]
    assert storage.latest_ready_document_ids(allowed) == [
        "pdf",
        "new-fpt-digest",
        "new-fpt-full",
    ]
    # A caller explicitly scoped to an old snapshot never escapes that scope.
    assert storage.latest_ready_document_ids(["old-fpt"]) == ["old-fpt"]


def test_build_is_deterministic_and_graph_failure_does_not_fail_document(storage):
    ready_digest(
        storage,
        document_id="hpg",
        ticker="HPG",
        text="HPG chịu rủi ro tỷ giá JPY.",
    )
    graph = service(storage)
    first = graph.build_document("hpg")
    with storage.connect() as db:
        assertion_ids = [
            row[0]
            for row in db.execute(
                "SELECT id FROM graph_assertions ORDER BY id"
            )
        ]
    assert first["status"] == "ready"
    assert first["assertions"] == 1
    graph.reindex("hpg", apply=True)
    second = graph.build_document("hpg")
    with storage.connect() as db:
        rebuilt_ids = [
            row[0]
            for row in db.execute(
                "SELECT id FROM graph_assertions ORDER BY id"
            )
        ]
    assert second["build_id"] == first["build_id"]
    assert rebuilt_ids == assertion_ids

    graph.reindex("hpg", apply=True)

    def invalid_quote(_value):
        return [
            {
                "predicate": "HAS_RISK",
                "object_literal": "currency",
                "quote": "Nội dung không tồn tại",
            }
        ]

    graph.extractor = invalid_quote
    with pytest.raises(ValueError, match="exact parent substring"):
        graph.build_document("hpg")
    assert storage.get_document("hpg")["status"] == "ready"
    with storage.connect() as db:
        failed = db.execute(
            "SELECT status, active FROM graph_builds WHERE document_id='hpg'"
        ).fetchone()
    assert tuple(failed) == ("failed", 0)


def test_graph_retrieval_traverses_two_hops_and_scopes_every_edge(storage):
    ready_digest(
        storage,
        document_id="hpg",
        ticker="HPG",
        text="HPG chịu rủi ro tỷ giá JPY.",
        analysis_date="2026-08-21",
    )
    ready_digest(
        storage,
        document_id="fpt",
        ticker="FPT",
        text="FPT cũng chịu rủi ro tỷ giá JPY.",
        analysis_date="2026-08-22",
    )
    graph = service(storage)
    graph.build_document("hpg")
    graph.build_document("fpt")
    retriever = GraphRetriever(
        storage,
        extractor_fingerprint=EXTRACTOR_FINGERPRINT,
        max_hops=2,
    )
    results = retriever.search("HPG có quan hệ gì?", k=20)
    assert {item["document_id"] for item in results} == {"hpg", "fpt"}
    assert all(item["score"] == 0.0 for item in results)
    assert all(item["text"] and item["parent_id"] for item in results)
    assert any(path["hop"] == 2 for item in results for path in item["graph_paths"])

    scoped = retriever.search(
        "HPG và JPY", k=20, document_ids=["hpg"], max_hops=2
    )
    assert [item["document_id"] for item in scoped] == ["hpg"]
    dated = retriever.search(
        "HPG và JPY",
        k=20,
        filters={"date_to": "2026-08-21"},
        max_hops=2,
    )
    assert [item["document_id"] for item in dated] == ["hpg"]


def test_ambiguous_alias_is_not_automatically_merged(storage):
    ready_digest(
        storage,
        document_id="aaa",
        ticker="AAA",
        text="AAA hợp tác với ACME.",
    )
    ready_digest(
        storage,
        document_id="bbb",
        ticker="BBB",
        text="BBB hợp tác với ACME.",
    )

    def ambiguous_alias_extractor(value):
        suffix = "a" if value.document["id"] == "aaa" else "b"
        return [
            GraphAssertionInput(
                predicate="PARTNERS_WITH",
                quote=value.parent["text"],
                object_entity=GraphEntityInput(
                    entity_type="organization",
                    canonical_key=f"acme-{suffix}",
                    canonical_name="ACME",
                ),
            )
        ]

    graph = service(storage, extractor=ambiguous_alias_extractor)
    graph.build_document("aaa")
    graph.build_document("bbb")
    retriever = GraphRetriever(
        storage,
        extractor_fingerprint=EXTRACTOR_FINGERPRINT,
    )

    assert retriever.search("ACME") == []
    assert [
        item["document_id"]
        for item in retriever.search("acme-a", max_hops=1)
    ] == ["aaa"]


def test_security_ticker_exact_precedes_same_key_other_entity_type(storage):
    ready_digest(
        storage,
        document_id="jpy-security",
        ticker="JPY",
        text="JPY chịu rủi ro lãi suất.",
    )
    ready_digest(
        storage,
        document_id="hpg-currency",
        ticker="HPG",
        text="HPG chịu rủi ro tỷ giá JPY.",
    )

    def colliding_key_extractor(value):
        if value.document["id"] == "jpy-security":
            return [
                GraphAssertionInput(
                    predicate="HAS_RISK",
                    quote=value.parent["text"],
                    object_entity=GraphEntityInput(
                        entity_type="risk",
                        canonical_key="interest-rate",
                        canonical_name="lãi suất",
                    ),
                )
            ]
        return relation_extractor(value)

    graph = service(storage, extractor=colliding_key_extractor)
    graph.build_document("jpy-security")
    graph.build_document("hpg-currency")
    retriever = GraphRetriever(
        storage,
        extractor_fingerprint=EXTRACTOR_FINGERPRINT,
    )

    assert [
        item["document_id"]
        for item in retriever.search("JPY", max_hops=1)
    ] == ["jpy-security"]


def test_current_build_invariant_hides_stale_content_and_delete_cascades(storage):
    ready_digest(
        storage,
        document_id="hpg",
        ticker="HPG",
        text="HPG chịu rủi ro tỷ giá JPY.",
    )
    graph = service(storage)
    graph.build_document("hpg")
    retriever = GraphRetriever(storage, extractor_fingerprint=EXTRACTOR_FINGERPRINT)
    assert retriever.search("HPG JPY")
    with storage.connect() as db:
        db.execute(
            "UPDATE documents SET content_sha256='new-content' WHERE id='hpg'"
        )
    assert retriever.search("HPG JPY") == []
    graph.reconcile("hpg", apply=True)
    with storage.connect() as db:
        states = [
            tuple(row)
            for row in db.execute(
                "SELECT status, active FROM graph_builds ORDER BY created_at, id"
            )
        ]
    assert ("stale", 0) in states
    assert ("pending", 0) in states
    storage.delete_document_rows("hpg")
    with storage.connect() as db:
        assert db.execute("SELECT count(*) FROM graph_builds").fetchone()[0] == 0


def test_interrupted_graph_build_returns_to_pending_without_touching_document(storage):
    ready_digest(
        storage,
        document_id="hpg",
        ticker="HPG",
        text="HPG có nội dung chuẩn hóa.",
    )
    graph = service(storage, extractor=None)
    graph.reconcile(apply=True)
    claimed = graph.claim_next_build("hpg")
    assert claimed is not None
    assert graph.reset_interrupted_builds() == 1
    with storage.connect() as db:
        status = db.execute(
            "SELECT status, error_code FROM graph_builds WHERE document_id='hpg'"
        ).fetchone()
    assert tuple(status) == ("pending", "interrupted")
    assert storage.get_document("hpg")["status"] == "ready"


def test_complete_structured_decision_creates_deterministic_assertions(storage):
    decision = {
        "schema_version": 1,
        "status": "complete",
        "warnings": [],
        "portfolio": {
            "rating": "Buy",
            "price_target": 85000,
            "price_target_currency": "VND",
        },
        "trading": {"action": "Buy"},
    }
    decision_text = json.dumps(
        decision, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    ready_digest(
        storage,
        document_id="hpg-full",
        ticker="HPG",
        text=decision_text,
        content_kind="full_report_v1",
        section="decision.structured",
        decision=decision,
    )
    graph = service(storage, extractor=None)
    result = graph.build_document("hpg-full")
    assert result["assertions"] == 3
    with storage.connect() as db:
        rows = db.execute(
            """
            SELECT ga.predicate, ga.object_literal, ge.quote, p.text
            FROM graph_assertions ga
            JOIN graph_evidence ge ON ge.assertion_id=ga.id
            JOIN parents p ON p.id=ge.parent_id
            ORDER BY ga.predicate
            """
        ).fetchall()
    assert {row[0] for row in rows} == {"RECOMMENDS", "TARGET_PRICE", "TRADING_ACTION"}
    assert all(row[2] in row[3] for row in rows)


@pytest.mark.asyncio
async def test_async_extractor_releases_shared_lock_per_parent(storage):
    ready_digest(
        storage,
        document_id="hpg",
        ticker="HPG",
        text="HPG chịu rủi ro tỷ giá JPY.",
    )
    generation_lock = asyncio.Semaphore(1)
    observed = []

    async def async_extractor(value):
        observed.append(generation_lock.locked())
        await asyncio.sleep(0)
        return relation_extractor(value)

    graph = GraphService(
        storage,
        index_fingerprint=INDEX_FINGERPRINT,
        extractor_fingerprint=EXTRACTOR_FINGERPRINT,
        extractor=async_extractor,
        generation_lock=generation_lock,
    )
    result = await graph.build_document_async("hpg")
    assert result["status"] == "ready"
    assert observed == [True]
    assert not generation_lock.locked()


@pytest.mark.asyncio
async def test_idle_worker_reconciles_documents_that_become_ready_after_start(storage):
    graph = service(storage, extractor=None)
    await graph.start()
    try:
        ready_digest(
            storage,
            document_id="late-hpg",
            ticker="HPG",
            text="HPG có nội dung chuẩn hóa.",
        )
        for _ in range(30):
            with storage.connect() as db:
                row = db.execute(
                    "SELECT status FROM graph_builds WHERE document_id='late-hpg'"
                ).fetchone()
            if row is not None and row["status"] == "ready":
                break
            await asyncio.sleep(0.05)
        else:
            pytest.fail("idle graph worker did not reconcile the new ready document")
    finally:
        await graph.stop()
