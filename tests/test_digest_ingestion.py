from __future__ import annotations

import asyncio
import hashlib
import json
import sqlite3
from dataclasses import replace

import pytest
from fastapi.testclient import TestClient

from rag_app.api import create_app
from rag_app.ingestion import IngestionService
from rag_app.retrieval import RetrievedSource, format_context
from rag_app.storage import SQLiteStorage


class NullLogger:
    def info(self, *_args, **_kwargs):
        pass

    def error(self, *_args, **_kwargs):
        pass


class CapturingLogger(NullLogger):
    def __init__(self):
        self.records = []

    def info(self, *args, **kwargs):
        self.records.append((args, kwargs))

    def error(self, *args, **kwargs):
        self.records.append((args, kwargs))


class MemoryIndex:
    def __init__(self):
        self.children: dict[str, tuple[str, dict]] = {}
        self.fail_provenance_updates = 0

    def add_children(self, *, ids, texts, metadatas):
        self.children.update(
            {item_id: (text, metadata) for item_id, text, metadata in zip(ids, texts, metadatas)}
        )

    def delete(self, ids, _collection_name=None):
        for item_id in ids:
            self.children.pop(item_id, None)

    def update_provenance(
        self,
        ids,
        research_mode,
        data_provenance,
        _collection_name=None,
    ):
        if self.fail_provenance_updates:
            self.fail_provenance_updates -= 1
            raise RuntimeError("temporary Chroma metadata failure")
        if any(item_id not in self.children for item_id in ids):
            return False
        for item_id in ids:
            text, existing = self.children[item_id]
            self.children[item_id] = (
                text,
                {
                    **existing,
                    "research_mode": research_mode,
                    "data_provenance": data_provenance,
                },
            )
        return True


class FailOnceDeleteIndex(MemoryIndex):
    def __init__(self):
        super().__init__()
        self.fail_next_delete = False

    def delete(self, ids, _collection_name=None):
        if self.fail_next_delete:
            self.fail_next_delete = False
            raise RuntimeError("temporary Chroma delete failure")
        super().delete(ids, _collection_name)


class DigestRuntime:
    def __init__(self, settings):
        self.storage = SQLiteStorage(settings.sqlite_path)
        self.storage.initialise()
        self.index = MemoryIndex()
        self.logger = NullLogger()
        self.ingestion = IngestionService(
            settings, self.storage, self.index, self.logger
        )

    async def start(self):
        pass

    async def stop(self):
        pass


def sections(summary="Thanh khoản cải thiện"):
    return {
        "summary": summary,
        "market": "ADTV20 thuộc nhóm dẫn đầu.",
        "fundamentals": "Chưa có thay đổi trọng yếu.",
        "news_events": "Không có sự kiện mới được xác nhận.",
        "sentiment": "Tâm lý trung tính.",
        "macro": "Điều kiện vĩ mô ổn định.",
        "catalysts": "Theo dõi tăng trưởng doanh thu.",
        "risks_unknowns": "Rủi ro biến động thanh khoản.",
        "next_session_watch": "Theo dõi khối lượng phiên kế tiếp.",
    }


def digest_payload(
    *,
    updated="2026-08-21T17:00:00+07:00",
    summary="Thanh khoản cải thiện",
    ticker="HPG",
    liquidity_rank=1,
):
    return {
        "schema_version": 1,
        "source": "tradingagents_daily_research",
        "source_run_id": f"run-20260821-{ticker}",
        "ticker": ticker,
        "analysis_date": "2026-08-21",
        "cutoff": "2026-08-21T16:45:00+07:00",
        "source_updated_at": updated,
        "liquidity_rank": liquidity_rank,
        "prompt_fingerprint": "prompt-v1",
        "model_fingerprint": "quick-model-v1",
        "evidence_fingerprint": "evidence-v1",
        "status": "complete",
        "sections": sections(summary),
    }


def authorised(key="test-secret"):
    return {"Authorization": f"Bearer {key}"}


def create_batch(client, *, batch_id="eod-20260821", target_count=1):
    return client.put(
        f"/internal/v1/digest-batches/{batch_id}",
        headers=authorised(),
        json={
            "source": "tradingagents_daily_research",
            "analysis_date": "2026-08-21",
            "cutoff": "2026-08-21T16:45:00+07:00",
            "target_count": target_count,
        },
    )


def sha256_json(value):
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def labelled_digest_payload(**overrides):
    payload = digest_payload(**overrides)
    payload.update(
        {
            "identity_hash": "a" * 64,
            "digest_hash": sha256_json(payload["sections"]),
            "research_mode": "eod",
            "data_provenance": "eod_cutoff",
        }
    )
    return payload


def full_report_payload(*, updated="2026-08-21T17:30:00+07:00", suffix=""):
    section_names = (
        "analyst.market",
        "analyst.sentiment",
        "analyst.news",
        "analyst.fundamentals",
        "research.bull",
        "research.bear",
        "research.manager",
        "trading.plan",
        "risk.aggressive",
        "risk.conservative",
        "risk.neutral",
        "portfolio.decision",
    )
    report_sections = {
        section: {"status": "complete", "text": f"Nội dung {section}{suffix}"}
        for section in section_names
    }
    return {
        "schema_version": 2,
        "content_kind": "full_report_v1",
        "report_schema": "full_report_v1",
        "source": "tradingagents_daily_research",
        "ticker": "HPG",
        "analysis_date": "2026-08-21",
        "cutoff": "2026-08-21T16:45:00+07:00",
        "source_updated_at": updated,
        "liquidity_rank": 1,
        "research_mode": "eod",
        "data_provenance": "eod_cutoff",
        "identity_hash": "b" * 64,
        "parent_identity_hash": "a" * 64,
        "expected_digest_hash": sha256_json(sections()),
        "full_report_hash": sha256_json(report_sections),
        "pipeline_version": "gx_full_v1",
        "prompt_fingerprint": "c" * 64,
        "model_fingerprint": "d" * 64,
        "evidence_fingerprint": "e" * 64,
        "execution_generation": 1,
        "contains_decision_content": True,
        "status": "complete",
        "stage_status": {
            stage: "complete"
            for stage in (
                "market",
                "sentiment",
                "news",
                "fundamentals",
                "research",
                "trader",
                "risk",
            )
        },
        "stage_fingerprints": {
            stage: hashlib.sha256(stage.encode()).hexdigest()
            for stage in (
                "market",
                "sentiment",
                "news",
                "fundamentals",
                "research",
                "trader",
                "risk",
            )
        },
        "sections": report_sections,
    }


def structured_full_report_payload(*, target_available=True):
    payload = full_report_payload()
    if target_available:
        target_fields = (
            "**Price Target Currency**: VND\n\n"
            "**Price Target Rationale**: Dựa trên tăng trưởng lợi nhuận và định giá.\n\n"
            "**Price Target**: 77000\n\n"
            "**Price Target Unavailable Reason**: Unavailable\n\n"
            "**Price Target Status**: Available"
        )
    else:
        target_fields = (
            "**Price Target Rationale**: Unavailable\n\n"
            "**Price Target**: Unavailable\n\n"
            "**Price Target Currency**: Unavailable\n\n"
            "**Price Target Status**: Unavailable\n\n"
            "**Price Target Unavailable Reason**: Không có dữ liệu định giá đủ tin cậy."
        )
    payload["sections"]["portfolio.decision"]["text"] = (
        "**Investment Thesis**: Luận điểm thứ nhất.\n\n"
        "Luận điểm thứ hai vẫn thuộc cùng trường thesis.\n\n"
        "**Rating**: Overweight\n\n"
        "**Executive Summary**: Tích lũy dần theo ba đợt.\n\n"
        f"{target_fields}\n\n"
        "**Time Horizon**: 3-6 months"
    )
    payload["sections"]["trading.plan"]["text"] = (
        "**Reasoning**: Định giá và động lượng hỗ trợ giao dịch.\n\n"
        "**Stop Loss**: 66500\n\n"
        "**Action**: Buy\n\n"
        "**Position Sizing**: 3-5% danh mục\n\n"
        "**Entry Price**: 70700.0\n\n"
        "FINAL TRANSACTION PROPOSAL: **BUY**"
    )
    payload["full_report_hash"] = sha256_json(payload["sections"])
    return payload


def create_labelled_daily_and_seal(client, runtime):
    batch = client.put(
        "/internal/v1/digest-batches/full-20260821",
        headers=authorised(),
        json={
            "source": "tradingagents_daily_research",
            "analysis_date": "2026-08-21",
            "cutoff": "2026-08-21T16:45:00+07:00",
            "target_count": 1,
            "research_mode": "eod",
            "data_provenance": "eod_cutoff",
        },
    )
    assert batch.status_code == 200
    created = client.post(
        "/internal/v1/digest-batches/full-20260821/documents:batch",
        headers=authorised(),
        json={"documents": [labelled_digest_payload()]},
    )
    assert created.status_code == 202
    document_id = created.json()["items"][0]["document_id"]
    assert runtime.ingestion.process_next() is True
    assert client.post(
        "/internal/v1/digest-batches/full-20260821:seal",
        headers=authorised(),
    ).status_code == 200
    return document_id


def test_internal_api_is_disabled_until_service_key_is_configured(settings):
    runtime = DigestRuntime(settings)
    with TestClient(create_app(settings, runtime)) as client:
        response = client.get(
            "/internal/v1/digest-batches/missing", headers=authorised()
        )
    assert response.status_code == 503


def test_internal_digest_api_auth_idempotency_stale_update_and_seal(settings):
    configured = replace(settings, rag_service_api_key="test-secret")
    runtime = DigestRuntime(configured)
    app = create_app(configured, runtime)

    with TestClient(app) as client:
        endpoint = "/internal/v1/digest-batches/eod-20260821"
        assert client.put(endpoint, json={}).status_code == 401
        assert client.put(endpoint, headers=authorised("wrong"), json={}).status_code == 401
        created_batch = create_batch(client)
        assert created_batch.status_code == 200
        assert created_batch.json()["action"] == "created"
        assert create_batch(client).json()["action"] == "unchanged"
        duplicate_identity = client.put(
            "/internal/v1/digest-batches/another-id",
            headers=authorised(),
            json={
                "source": "tradingagents_daily_research",
                "analysis_date": "2026-08-21",
                "cutoff": "2026-08-21T16:45:00+07:00",
                "target_count": 1,
            },
        )
        assert duplicate_identity.status_code == 409
        early_seal = client.post(f"{endpoint}:seal", headers=authorised())
        assert early_seal.status_code == 409

        ingest_endpoint = f"{endpoint}/documents:batch"
        first = client.post(
            ingest_endpoint,
            headers=authorised(),
            json={"documents": [digest_payload()]},
        )
        assert first.status_code == 202
        assert first.json()["items"][0]["action"] == "created"
        pending_status = client.get(endpoint, headers=authorised()).json()
        assert pending_status["counts"]["pending"] == 1
        assert pending_status["items"][0]["status"] == "pending"
        assert pending_status["items"][0]["ticker"] == "HPG"
        duplicate = client.post(
            ingest_endpoint,
            headers=authorised(),
            json={"documents": [digest_payload()]},
        )
        assert duplicate.json()["items"][0]["action"] == "unchanged"
        assert len(runtime.storage.list_documents()) == 1

        assert runtime.ingestion.process_next() is True
        document = runtime.storage.list_documents()[0]
        assert document["status"] == "ready"
        assert document["source_type"] == "trading_digest"
        assert runtime.index.children
        assert {
            metadata["digest_section"]
            for _, metadata in runtime.index.children.values()
        } == set(sections())
        assert all(
            metadata["prompt_fingerprint"] == "prompt-v1"
            and metadata["model_fingerprint"] == "quick-model-v1"
            and metadata["evidence_fingerprint"] == "evidence-v1"
            for _, metadata in runtime.index.children.values()
        )
        ready_status = client.get(
            f"{endpoint}?status=ready&ticker=HPG", headers=authorised()
        ).json()
        assert ready_status["items_total"] == 1
        assert ready_status["items"][0]["status"] == "ready"
        catalog = client.get(
            "/api/documents?page=1&page_size=50&source_type=trading_digest&ticker=HPG"
        ).json()
        assert catalog["total"] == 1
        assert catalog["items"][0]["source_type"] == "trading_digest"

        stale = client.post(
            ingest_endpoint,
            headers=authorised(),
            json={
                "documents": [
                    digest_payload(
                        updated="2026-08-21T16:59:00+07:00",
                        summary="Nội dung cũ không được ghi",
                    )
                ]
            },
        )
        assert stale.json()["items"][0]["action"] == "stale"
        assert runtime.storage.digest_sections(document["id"])[0]["text"] == "Thanh khoản cải thiện"

        newer = client.post(
            ingest_endpoint,
            headers=authorised(),
            json={
                "documents": [
                    digest_payload(
                        updated="2026-08-21T17:01:00+07:00",
                        summary="Thanh khoản cải thiện, rủi ro đã cập nhật",
                    )
                ]
            },
        )
        assert newer.json()["items"][0]["action"] == "updated"
        assert newer.json()["items"][0]["status"] == "pending"
        assert runtime.ingestion.process_next() is True
        assert runtime.storage.digest_sections(document["id"])[0]["text"].endswith(
            "rủi ro đã cập nhật"
        )

        sealed = client.post(f"{endpoint}:seal", headers=authorised())
        assert sealed.status_code == 200
        assert sealed.json()["status"] == "sealed"
        assert sealed.json()["rag_ready"] == 1
        assert sealed.json()["complete"] is True
        after_seal = client.post(
            ingest_endpoint,
            headers=authorised(),
            json={"documents": [digest_payload()]},
        )
        assert after_seal.status_code == 409


def test_same_content_advances_stale_watermark_without_reindex(settings):
    configured = replace(settings, rag_service_api_key="test-secret")
    runtime = DigestRuntime(configured)

    with TestClient(create_app(configured, runtime)) as client:
        assert create_batch(client).status_code == 200
        endpoint = "/internal/v1/digest-batches/eod-20260821/documents:batch"
        first = client.post(
            endpoint,
            headers=authorised(),
            json={"documents": [digest_payload()]},
        )
        document_id = first.json()["items"][0]["document_id"]

        newer_same = client.post(
            endpoint,
            headers=authorised(),
            json={
                "documents": [
                    digest_payload(updated="2026-08-21T17:10:00+07:00")
                ]
            },
        )
        assert newer_same.json()["items"][0]["action"] == "unchanged"
        assert runtime.storage.get_document(document_id)["source_updated_at"] == (
            "2026-08-21T10:10:00+00:00"
        )

        older_changed = client.post(
            endpoint,
            headers=authorised(),
            json={
                "documents": [
                    digest_payload(
                        updated="2026-08-21T17:05:00+07:00",
                        summary="Nội dung đến trễ không được ghi",
                    )
                ]
            },
        )
        assert older_changed.json()["items"][0]["action"] == "stale"
        assert runtime.storage.digest_sections(document_id)[0]["text"] == (
            "Thanh khoản cải thiện"
        )


def test_digest_update_uses_durable_vector_deletion_outbox(settings):
    configured = replace(settings, rag_service_api_key="test-secret")
    runtime = DigestRuntime(configured)

    with TestClient(create_app(configured, runtime)) as client:
        assert create_batch(client).status_code == 200
        endpoint = "/internal/v1/digest-batches/eod-20260821/documents:batch"
        created = client.post(
            endpoint,
            headers=authorised(),
            json={"documents": [digest_payload()]},
        )
        document_id = created.json()["items"][0]["document_id"]
        assert runtime.ingestion.process_next() is True
        old_child_ids = set(runtime.index.children)
        assert old_child_ids

        updated = client.post(
            endpoint,
            headers=authorised(),
            json={
                "documents": [
                    digest_payload(
                        updated="2026-08-21T17:01:00+07:00",
                        summary="Nội dung thay thế",
                    )
                ]
            },
        )
        assert updated.json()["items"][0]["action"] == "updated"
        assert runtime.storage.get_document(document_id)["status"] == "pending"
        deletion_groups = runtime.storage.pending_vector_deletions(document_id)
        assert {item for group in deletion_groups for item in group["child_ids"]} == (
            old_child_ids
        )
        # API did not delete Chroma before the durable SQLite transition.
        assert set(runtime.index.children) == old_child_ids

        # A new service instance can resume after a crash between the request
        # and worker execution and will drain the durable deletion outbox.
        restarted_storage = SQLiteStorage(configured.sqlite_path)
        restarted_storage.initialise()
        restarted = IngestionService(
            configured, restarted_storage, runtime.index, runtime.logger
        )
        assert restarted.process_next() is True
        assert restarted_storage.pending_vector_deletions(document_id) == []
        assert restarted_storage.get_document(document_id)["status"] == "ready"
        assert old_child_ids.isdisjoint(runtime.index.children)


def test_failed_digest_can_retry_exact_content_after_batch_is_sealed(settings):
    configured = replace(settings, rag_service_api_key="test-secret")
    runtime = DigestRuntime(configured)

    with TestClient(create_app(configured, runtime)) as client:
        assert create_batch(client).status_code == 200
        endpoint = "/internal/v1/digest-batches/eod-20260821/documents:batch"
        created = client.post(
            endpoint,
            headers=authorised(),
            json={"documents": [digest_payload()]},
        )
        document_id = created.json()["items"][0]["document_id"]
        assert client.post(
            "/internal/v1/digest-batches/eod-20260821:seal",
            headers=authorised(),
        ).status_code == 200
        runtime.storage.mark_failed(document_id, "temporary_embedding_failure")

        retried = client.post(
            endpoint,
            headers=authorised(),
            json={"documents": [digest_payload()]},
        )
        assert retried.status_code == 202
        assert retried.json()["items"][0] == {
            "ticker": "HPG",
            "document_id": document_id,
            "action": "retried",
            "status": "pending",
        }
        assert runtime.storage.get_document(document_id)["status"] == "pending"


def test_vector_cleanup_failure_is_reconcilable_after_seal(settings):
    configured = replace(settings, rag_service_api_key="test-secret")
    runtime = DigestRuntime(configured)
    runtime.index = FailOnceDeleteIndex()
    runtime.ingestion.index = runtime.index
    replacement = digest_payload(
        updated="2026-08-21T17:01:00+07:00", summary="Nội dung thay thế"
    )

    with TestClient(create_app(configured, runtime)) as client:
        assert create_batch(client).status_code == 200
        endpoint = "/internal/v1/digest-batches/eod-20260821/documents:batch"
        created = client.post(
            endpoint,
            headers=authorised(),
            json={"documents": [digest_payload()]},
        )
        document_id = created.json()["items"][0]["document_id"]
        assert runtime.ingestion.process_next() is True
        client.post(endpoint, headers=authorised(), json={"documents": [replacement]})

        runtime.index.fail_next_delete = True
        assert runtime.ingestion.process_next() is True
        assert runtime.storage.get_document(document_id)["status"] == "failed"
        assert runtime.storage.pending_vector_deletions(document_id)
        assert client.post(
            "/internal/v1/digest-batches/eod-20260821:seal",
            headers=authorised(),
        ).status_code == 200

        retried = client.post(
            endpoint, headers=authorised(), json={"documents": [replacement]}
        )
        assert retried.json()["items"][0]["action"] == "retried"
        assert runtime.ingestion.process_next() is True
        assert runtime.storage.get_document(document_id)["status"] == "ready"
        assert runtime.storage.pending_vector_deletions(document_id) == []


def test_rank_must_be_exactly_within_batch_target_and_seal_rechecks(settings):
    configured = replace(settings, rag_service_api_key="test-secret")
    runtime = DigestRuntime(configured)

    with TestClient(create_app(configured, runtime)) as client:
        assert create_batch(client, target_count=2).status_code == 200
        endpoint = "/internal/v1/digest-batches/eod-20260821/documents:batch"
        outside = client.post(
            endpoint,
            headers=authorised(),
            json={
                "documents": [
                    digest_payload(ticker="HPG", liquidity_rank=1),
                    digest_payload(ticker="VCB", liquidity_rank=3),
                ]
            },
        )
        assert outside.status_code == 400
        assert runtime.storage.list_documents() == []

        valid = client.post(
            endpoint,
            headers=authorised(),
            json={
                "documents": [
                    digest_payload(ticker="HPG", liquidity_rank=1),
                    digest_payload(ticker="VCB", liquidity_rank=2),
                ]
            },
        )
        assert valid.status_code == 202
        # The seal guard also protects databases written by an older service.
        with runtime.storage.connect() as db:
            db.execute(
                "UPDATE documents SET liquidity_rank=3 WHERE ticker='VCB'"
            )
        sealed = client.post(
            "/internal/v1/digest-batches/eod-20260821:seal",
            headers=authorised(),
        )
        assert sealed.status_code == 409


def test_raw_evidence_fields_are_rejected_and_never_persisted(settings):
    configured = replace(settings, rag_service_api_key="test-secret")
    runtime = DigestRuntime(configured)
    app = create_app(configured, runtime)

    with TestClient(app) as client:
        assert create_batch(client).status_code == 200
        unsafe = digest_payload()
        unsafe["raw_evidence"] = {
            "title": "RAW_SENTINEL",
            "excerpt": "RAW_SENTINEL",
            "body": "RAW_SENTINEL",
            "author": "RAW_SENTINEL",
            "prompt": "RAW_SENTINEL",
        }
        response = client.post(
            "/internal/v1/digest-batches/eod-20260821/documents:batch",
            headers=authorised(),
            json={"documents": [unsafe]},
        )
        assert response.status_code == 422
        assert runtime.storage.list_documents() == []

    with sqlite3.connect(configured.sqlite_path) as db:
        values = []
        for table, columns in (
            ("documents", "filename, error"),
            ("digest_sections", "section, text"),
            ("parents", "text"),
            ("children", "text"),
        ):
            values.extend(db.execute(f"SELECT {columns} FROM {table}").fetchall())
    assert "RAW_SENTINEL" not in repr(values)


def test_digest_payload_limits_and_duplicate_rank_are_rejected(settings):
    configured = replace(settings, rag_service_api_key="test-secret")
    runtime = DigestRuntime(configured)
    app = create_app(configured, runtime)
    endpoint = "/internal/v1/digest-batches/eod-20260821/documents:batch"

    with TestClient(app) as client:
        assert create_batch(client).status_code == 200
        too_large = digest_payload()
        too_large["sections"] = {key: "🧪" * 20000 for key in sections()}
        response = client.post(
            endpoint, headers=authorised(), json={"documents": [too_large]}
        )
        assert response.status_code == 400
        assert runtime.storage.list_documents() == []

        too_many_claims = digest_payload()
        too_many_claims["sections"]["summary"] = {
            "claims": [
                {"text": f"claim {number}", "evidence_ids": [f"e:{number}"]}
                for number in range(9)
            ]
        }
        response = client.post(
            endpoint, headers=authorised(), json={"documents": [too_many_claims]}
        )
        assert response.status_code == 422

        first = digest_payload()
        second = digest_payload()
        second["ticker"] = "VCB"
        second["source_run_id"] = "run-VCB"
        second["sections"] = sections("VCB digest")
        # The same liquidity rank cannot identify two members of one batch.
        response = client.post(
            endpoint,
            headers=authorised(),
            json={"documents": [first, second]},
        )
        assert response.status_code == 400
        assert runtime.storage.list_documents() == []


def test_digest_batch_item_status_exposes_safe_failure_for_reconciliation(settings):
    configured = replace(settings, rag_service_api_key="test-secret")
    runtime = DigestRuntime(configured)
    app = create_app(configured, runtime)

    with TestClient(app) as client:
        assert create_batch(client).status_code == 200
        response = client.post(
            "/internal/v1/digest-batches/eod-20260821/documents:batch",
            headers=authorised(),
            json={"documents": [digest_payload()]},
        )
        document_id = response.json()["items"][0]["document_id"]
        claimed = runtime.storage.claim_next_pending()
        assert claimed["id"] == document_id
        runtime.storage.mark_failed(document_id, "embedding_failed")

        status = client.get(
            "/internal/v1/digest-batches/eod-20260821?status=failed",
            headers=authorised(),
        ).json()
        assert status["counts"]["failed"] == 1
        assert status["items"] == [
            {
                "document_id": document_id,
                "ticker": "HPG",
                "liquidity_rank": 1,
                "status": "failed",
                "error": "embedding_failed",
                "digest_status": "complete",
                "research_mode": "legacy_unknown",
                "data_provenance": "legacy_unknown",
                "source_updated_at": "2026-08-21T10:00:00+00:00",
                "updated_at": status["items"][0]["updated_at"],
            }
        ]

def test_digest_metadata_filters_and_pdf_default_are_backward_compatible(settings, tmp_path):
    storage = SQLiteStorage(settings.sqlite_path)
    storage.initialise()
    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(b"%PDF-dummy")
    storage.create_pending_document(
        document_id="pdf-doc",
        filename="paper.pdf",
        sha256="pdf-sha",
        stored_path=pdf,
        size_bytes=10,
        page_count=1,
    )
    assert storage.get_document("pdf-doc")["source_type"] == "pdf"
    assert storage.get_document("pdf-doc")["research_mode"] == "legacy_unknown"
    assert storage.get_document("pdf-doc")["data_provenance"] == "legacy_unknown"

    digest = digest_payload()
    from rag_app.digests import canonicalise_digest

    canonical = canonicalise_digest(
        digest, batch_source="tradingagents_daily_research"
    )
    storage.create_digest_batch(
        batch_id="batch",
        source="tradingagents_daily_research",
        analysis_date="2026-08-21",
        cutoff="2026-08-21T16:45:00+07:00",
        target_count=1,
    )
    stored = storage.replace_digest_document(digest=canonical, batch_id="batch")
    storage.claim_pending(2)
    # Mark minimal parent/child rows ready without requiring the embedding stack.
    storage.finish_indexing(
        document_id=stored["id"],
        page_count=0,
        parents=[
            {
                "id": "digest-parent",
                "document_id": stored["id"],
                "page": 0,
                "digest_section": "risks_unknowns",
                "start_index": 0,
                "text": "Rủi ro",
            }
        ],
        children=[
            {
                "id": "digest-child",
                "parent_id": "digest-parent",
                "document_id": stored["id"],
                "page": 0,
                "digest_section": "risks_unknowns",
                "start_index": 0,
                "text": "Rủi ro thanh khoản",
            }
        ],
        fingerprint=settings.index_fingerprint,
        collection_name=settings.collection_name,
    )
    assert storage.ready_document_ids(
        filters={
            "source_types": ["trading_digest"],
            "tickers": ["HPG"],
            "date_from": "2026-08-21",
            "date_to": "2026-08-21",
            "liquidity_rank_max": 3,
            "digest_sections": ["risks_unknowns"],
        }
    ) == [stored["id"]]
    assert storage.ready_document_ids(filters={"tickers": ["VCB"]}) == []
    lexical = storage.lexical_search(
        "thanh khoan",
        k=5,
        document_ids=None,
        filters={
            "source_types": ["trading_digest"],
            "tickers": ["HPG"],
            "digest_sections": ["risks_unknowns"],
        },
    )
    assert lexical[0]["child_id"] == "digest-child"

    parent = storage.parents_by_ids(["digest-parent"])["digest-parent"]
    assert parent["ticker"] == "HPG"
    assert parent["digest_section"] == "risks_unknowns"


def test_digest_citation_uses_ticker_date_cutoff_and_section():
    source = RetrievedSource(
        id="1",
        document_id="digest-1",
        filename="HPG.digest",
        page=0,
        parent_id="parent-1",
        text="Rủi ro thanh khoản",
        score=0.91,
        source_type="trading_digest",
        ticker="HPG",
        analysis_date="2026-08-21",
        analysis_cutoff="2026-08-21T16:45:00+07:00",
        digest_section="risks_unknowns",
        liquidity_rank=3,
    )
    assert source.public()["label"] == "HPG · 21/08/2026 16:45 · Rủi ro"
    context = format_context([source])
    assert 'ticker="HPG"' in context
    assert 'section="risks_unknowns"' in context
    assert "page_start" not in context


def test_historical_provenance_is_persisted_filterable_and_cited(settings):
    configured = replace(settings, rag_service_api_key="test-secret")
    runtime = DigestRuntime(configured)
    batch_id = "historical-20260820"
    batch_endpoint = f"/internal/v1/digest-batches/{batch_id}"
    historical = digest_payload()
    historical.update(
        {
            "source": "tradingagents_adhoc_research",
            "source_run_id": "historical-run-20260820-HPG",
            "analysis_date": "2026-08-20",
            "cutoff": "2026-08-20T15:00:00+07:00",
            "source_updated_at": "2026-08-20T15:01:00+07:00",
            "research_mode": "historical",
            "data_provenance": "historical_replay",
        }
    )

    with TestClient(create_app(configured, runtime)) as client:
        created = client.put(
            batch_endpoint,
            headers=authorised(),
            json={
                "source": "tradingagents_adhoc_research",
                "analysis_date": "2026-08-20",
                "cutoff": "2026-08-20T15:00:00+07:00",
                "target_count": 1,
                "research_mode": "historical",
                "data_provenance": "historical_replay",
            },
        )
        assert created.status_code == 200
        assert created.json()["batch"]["research_mode"] == "historical"
        assert created.json()["batch"]["data_provenance"] == "historical_replay"

        accepted = client.post(
            f"{batch_endpoint}/documents:batch",
            headers=authorised(),
            json={"documents": [historical]},
        )
        assert accepted.status_code == 202
        assert runtime.ingestion.process_next() is True
        document_id = accepted.json()["items"][0]["document_id"]
        document = runtime.storage.get_document(document_id)
        assert document["research_mode"] == "historical"
        assert document["data_provenance"] == "historical_replay"
        assert all(
            metadata["research_mode"] == "historical"
            and metadata["data_provenance"] == "historical_replay"
            for _text, metadata in runtime.index.children.values()
        )
        assert runtime.storage.ready_document_ids(
            filters={
                "source_types": ["trading_digest"],
                "research_modes": ["historical"],
                "data_provenances": ["historical_replay"],
            }
        ) == [document_id]
        assert runtime.storage.ready_document_ids(
            filters={"research_modes": ["live"]}
        ) == []

        catalog = client.get(
            "/api/documents?page=1&page_size=50"
            "&source_type=trading_digest&research_mode=historical"
            "&data_provenance=historical_replay"
        ).json()
        assert catalog["total"] == 1
        assert catalog["items"][0]["data_provenance"] == "historical_replay"

    source = RetrievedSource(
        id="1",
        document_id=document_id,
        filename="HPG.digest",
        page=0,
        parent_id="parent-historical",
        text="Rủi ro thanh khoản",
        score=0.91,
        source_type="trading_digest",
        ticker="HPG",
        analysis_date="2026-08-20",
        analysis_cutoff="2026-08-20T15:00:00+07:00",
        digest_section="risks_unknowns",
        research_mode="historical",
        data_provenance="historical_replay",
    )
    assert source.public()["label"].endswith("· dựng lại lịch sử")
    context = format_context([source])
    assert 'research_mode="historical"' in context
    assert 'data_provenance="historical_replay"' in context


def test_provenance_wire_is_backward_compatible_but_rejects_partial_or_wrong_pairs(
    settings,
):
    configured = replace(settings, rag_service_api_key="test-secret")
    runtime = DigestRuntime(configured)
    with TestClient(create_app(configured, runtime)) as client:
        legacy = create_batch(client)
        assert legacy.status_code == 200
        assert legacy.json()["batch"]["research_mode"] == "legacy_unknown"
        assert legacy.json()["batch"]["data_provenance"] == "legacy_unknown"

        one_field = client.put(
            "/internal/v1/digest-batches/partial-provenance",
            headers=authorised(),
            json={
                "source": "tradingagents_daily_research",
                "analysis_date": "2026-08-22",
                "cutoff": "2026-08-22T16:45:00+07:00",
                "target_count": 1,
                "research_mode": "eod",
            },
        )
        assert one_field.status_code == 422

        wrong_pair = client.put(
            "/internal/v1/digest-batches/wrong-provenance",
            headers=authorised(),
            json={
                "source": "tradingagents_daily_research",
                "analysis_date": "2026-08-22",
                "cutoff": "2026-08-22T16:45:00+07:00",
                "target_count": 1,
                "research_mode": "historical",
                "data_provenance": "request_cutoff",
            },
        )
        assert wrong_pair.status_code == 422

        mismatched_document = digest_payload()
        mismatched_document.update(
            {"research_mode": "eod", "data_provenance": "eod_cutoff"}
        )
        response = client.post(
            "/internal/v1/digest-batches/eod-20260821/documents:batch",
            headers=authorised(),
            json={"documents": [mismatched_document]},
        )
        assert response.status_code == 400
        assert runtime.storage.list_documents() == []


def test_legacy_batch_provenance_can_be_upgraded_exactly_once(settings):
    configured = replace(settings, rag_service_api_key="test-secret")
    runtime = DigestRuntime(configured)
    endpoint = "/internal/v1/digest-batches/legacy-upgrade"
    identity = {
        "source": "tradingagents_daily_research",
        "analysis_date": "2026-08-23",
        "cutoff": "2026-08-23T16:45:00+07:00",
        "target_count": 1,
    }

    with TestClient(create_app(configured, runtime)) as client:
        created = client.put(endpoint, headers=authorised(), json=identity)
        assert created.status_code == 200
        assert created.json()["action"] == "created"

        labelled = {
            **identity,
            "research_mode": "eod",
            "data_provenance": "eod_cutoff",
        }
        upgraded = client.put(endpoint, headers=authorised(), json=labelled)
        assert upgraded.status_code == 200
        assert upgraded.json()["action"] == "upgraded"
        assert upgraded.json()["batch"]["research_mode"] == "eod"
        assert upgraded.json()["batch"]["data_provenance"] == "eod_cutoff"

        unchanged = client.put(endpoint, headers=authorised(), json=labelled)
        assert unchanged.status_code == 200
        assert unchanged.json()["action"] == "unchanged"

        downgrade = client.put(endpoint, headers=authorised(), json=identity)
        assert downgrade.status_code == 409

        relabel = client.put(
            endpoint,
            headers=authorised(),
            json={
                **identity,
                "research_mode": "historical",
                "data_provenance": "historical_replay",
            },
        )
        assert relabel.status_code == 409
        stored = runtime.storage.get_digest_batch("legacy-upgrade")
        assert stored["research_mode"] == "eod"
        assert stored["data_provenance"] == "eod_cutoff"


@pytest.mark.parametrize(
    ("sealed", "research_mode", "data_provenance"),
    [
        (False, "eod", "eod_cutoff"),
        (False, "historical", "historical_replay"),
        (True, "historical", "historical_replay"),
    ],
    ids=["open-ready-eod", "open-ready-historical", "sealed-ready-historical"],
)
def test_legacy_batch_upgrade_relabels_ready_document_and_vectors(
    settings, sealed, research_mode, data_provenance
):
    configured = replace(settings, rag_service_api_key="test-secret")
    runtime = DigestRuntime(configured)
    endpoint = "/internal/v1/digest-batches/legacy-ready"
    identity = {
        "source": "tradingagents_daily_research",
        "analysis_date": "2026-08-21",
        "cutoff": "2026-08-21T16:45:00+07:00",
        "target_count": 1,
    }
    labelled_identity = {
        **identity,
        "research_mode": research_mode,
        "data_provenance": data_provenance,
    }
    labelled_digest = {
        **digest_payload(),
        "research_mode": research_mode,
        "data_provenance": data_provenance,
    }

    with TestClient(create_app(configured, runtime)) as client:
        assert client.put(endpoint, headers=authorised(), json=identity).status_code == 200
        accepted = client.post(
            f"{endpoint}/documents:batch",
            headers=authorised(),
            json={"documents": [digest_payload()]},
        )
        assert accepted.status_code == 202
        document_id = accepted.json()["items"][0]["document_id"]
        assert runtime.ingestion.process_next() is True
        legacy_document = runtime.storage.get_document(document_id)
        legacy_content_hash = legacy_document["content_sha256"]
        legacy_child_ids = set(runtime.index.children)
        assert all(
            metadata["research_mode"] == "legacy_unknown"
            and metadata["data_provenance"] == "legacy_unknown"
            for _text, metadata in runtime.index.children.values()
        )
        if sealed:
            sealed_response = client.post(
                f"/internal/v1/digest-batches/legacy-ready:seal",
                headers=authorised(),
            )
            assert sealed_response.status_code == 200

        upgraded = client.put(
            endpoint, headers=authorised(), json=labelled_identity
        )
        assert upgraded.status_code == 200
        assert upgraded.json()["action"] == "upgraded"
        upgraded_document = runtime.storage.get_document(document_id)
        assert upgraded_document["research_mode"] == research_mode
        assert upgraded_document["data_provenance"] == data_provenance
        assert upgraded_document["content_sha256"] != legacy_content_hash
        assert set(runtime.index.children) == legacy_child_ids
        assert all(
            metadata["research_mode"] == research_mode
            and metadata["data_provenance"] == data_provenance
            for _text, metadata in runtime.index.children.values()
        )
        assert runtime.storage.pending_vector_metadata_updates(document_id) == []

        replay = client.post(
            f"{endpoint}/documents:batch",
            headers=authorised(),
            json={"documents": [labelled_digest]},
        )
        assert replay.status_code == 202
        assert replay.json()["items"][0]["action"] == "unchanged"
        assert set(runtime.index.children) == legacy_child_ids


def test_legacy_batch_upgrade_vector_metadata_recovers_from_crash(settings):
    configured = replace(settings, rag_service_api_key="test-secret")
    runtime = DigestRuntime(configured)
    endpoint = "/internal/v1/digest-batches/legacy-recovery"
    identity = {
        "source": "tradingagents_daily_research",
        "analysis_date": "2026-08-21",
        "cutoff": "2026-08-21T16:45:00+07:00",
        "target_count": 1,
    }

    with TestClient(create_app(configured, runtime)) as client:
        assert client.put(endpoint, headers=authorised(), json=identity).status_code == 200
        accepted = client.post(
            f"{endpoint}/documents:batch",
            headers=authorised(),
            json={"documents": [digest_payload()]},
        )
        document_id = accepted.json()["items"][0]["document_id"]
        assert runtime.ingestion.process_next() is True
        runtime.index.fail_provenance_updates = 1

        upgraded = client.put(
            endpoint,
            headers=authorised(),
            json={
                **identity,
                "research_mode": "eod",
                "data_provenance": "eod_cutoff",
            },
        )
        assert upgraded.status_code == 200
        assert runtime.storage.get_document(document_id)["research_mode"] == "eod"
        assert runtime.storage.pending_vector_metadata_updates(document_id)
        assert all(
            metadata["research_mode"] == "legacy_unknown"
            for _text, metadata in runtime.index.children.values()
        )

        retried = client.put(
            endpoint,
            headers=authorised(),
            json={
                **identity,
                "research_mode": "eod",
                "data_provenance": "eod_cutoff",
            },
        )
        assert retried.status_code == 200
        assert retried.json()["action"] == "unchanged"
        assert runtime.storage.pending_vector_metadata_updates(document_id) == []
        assert all(
            metadata["research_mode"] == "eod"
            and metadata["data_provenance"] == "eod_cutoff"
            for _text, metadata in runtime.index.children.values()
        )


@pytest.mark.parametrize("sealed", [False, True], ids=["open", "sealed"])
async def test_concurrent_legacy_post_and_batch_upgrade_share_one_snapshot(
    settings, sealed
):
    runtime = DigestRuntime(settings)
    batch_id = "legacy-race"
    source = "tradingagents_daily_research"
    identity = {
        "batch_id": batch_id,
        "source": source,
        "analysis_date": "2026-08-21",
        "cutoff": "2026-08-21T16:45:00+07:00",
        "target_count": 1,
    }
    created, _batch = runtime.storage.create_digest_batch(**identity)
    assert created == "created"
    legacy_payload = digest_payload()

    if sealed:
        accepted = await runtime.ingestion.accept_digest_documents(
            batch_id=batch_id,
            batch_source=source,
            payloads=[legacy_payload],
        )
        assert accepted[0]["action"] == "created"
        assert runtime.ingestion.process_next() is True
        action, _batch = await runtime.ingestion.seal_digest_batch(batch_id)
        assert action == "sealed"

    # Queue PUT first and POST second behind the exact service lock. asyncio's
    # fair lock ordering deterministically exercises PUT -> stale-client POST.
    await runtime.ingestion._upload_lock.acquire()
    put_task = asyncio.create_task(
        runtime.ingestion.put_digest_batch(
            **identity,
            research_mode="eod",
            data_provenance="eod_cutoff",
        )
    )
    await asyncio.sleep(0)
    post_task = asyncio.create_task(
        runtime.ingestion.accept_digest_documents(
            batch_id=batch_id,
            batch_source=source,
            payloads=[legacy_payload],
        )
    )
    await asyncio.sleep(0)
    runtime.ingestion._upload_lock.release()
    (put_action, _batch), post_result = await asyncio.gather(put_task, post_task)

    assert put_action == "upgraded"
    assert post_result[0]["action"] == ("unchanged" if sealed else "created")
    if not sealed:
        assert runtime.ingestion.process_next() is True

    batch = runtime.storage.get_digest_batch(batch_id)
    assert batch["research_mode"] == "eod"
    assert batch["data_provenance"] == "eod_cutoff"
    document = runtime.storage.get_document(post_result[0]["document_id"])
    assert document["research_mode"] == "eod"
    assert document["data_provenance"] == "eod_cutoff"
    assert all(
        metadata["research_mode"] == "eod"
        and metadata["data_provenance"] == "eod_cutoff"
        for _text, metadata in runtime.index.children.values()
    )

    exact_replay = await runtime.ingestion.accept_digest_documents(
        batch_id=batch_id,
        batch_source=source,
        payloads=[
            {
                **legacy_payload,
                "research_mode": "eod",
                "data_provenance": "eod_cutoff",
            }
        ],
    )
    assert exact_replay[0]["action"] == "unchanged"


def test_provenance_changes_content_hash_without_changing_logical_identity():
    from rag_app.digests import canonicalise_digest

    historical_payload = digest_payload()
    historical_payload.update(
        {"research_mode": "historical", "data_provenance": "historical_replay"}
    )
    live_payload = digest_payload()
    live_payload.update(
        {"research_mode": "live", "data_provenance": "request_cutoff"}
    )
    historical = canonicalise_digest(
        historical_payload,
        batch_source="tradingagents_daily_research",
        batch_research_mode="historical",
        batch_data_provenance="historical_replay",
    )
    live = canonicalise_digest(
        live_payload,
        batch_source="tradingagents_daily_research",
        batch_research_mode="live",
        batch_data_provenance="request_cutoff",
    )
    assert historical.source_key == live.source_key
    assert historical.content_sha256 != live.content_sha256


def test_sealed_daily_is_promoted_in_place_and_replayed_without_reembedding(settings):
    configured = replace(settings, rag_service_api_key="test-secret")
    runtime = DigestRuntime(configured)

    with TestClient(create_app(configured, runtime)) as client:
        document_id = create_labelled_daily_and_seal(client, runtime)
        old_child_ids = set(runtime.index.children)
        assert old_child_ids
        endpoint = (
            "/internal/v1/digest-batches/full-20260821/documents:promote"
        )
        promoted = client.post(
            endpoint,
            headers=authorised(),
            json={"documents": [full_report_payload()]},
        )
        assert promoted.status_code == 202
        assert promoted.json()["items"] == [
            {
                "ticker": "HPG",
                "document_id": document_id,
                "full_report_hash": full_report_payload()["full_report_hash"],
                "action": "promoted",
                "status": "pending",
            }
        ]
        document = runtime.storage.get_document(document_id)
        assert document["content_kind"] == "full_report_v1"
        assert document["report_schema"] == "full_report_v1"
        assert document["identity_hash"] == "b" * 64
        assert document["parent_identity_hash"] == "a" * 64
        assert document["expected_digest_hash"] == sha256_json(sections())
        assert document["full_report_hash"] == full_report_payload()["full_report_hash"]
        assert document["execution_generation"] == 1
        assert document["contains_decision_content"] == 1
        assert {item["section"] for item in runtime.storage.digest_sections(document_id)} == set(
            full_report_payload()["sections"]
        )
        queued = runtime.storage.pending_vector_deletions(document_id)
        assert {child for group in queued for child in group["child_ids"]} == old_child_ids
        # Chroma remains untouched until the crash-safe writer drains the outbox.
        assert set(runtime.index.children) == old_child_ids
        pending_replay = client.post(
            endpoint,
            headers=authorised(),
            json={"documents": [full_report_payload()]},
        )
        assert pending_replay.json()["items"][0]["action"] == "unchanged"
        assert pending_replay.json()["items"][0]["status"] == "pending"

        restarted_storage = SQLiteStorage(configured.sqlite_path)
        restarted_storage.initialise()
        assert restarted_storage.get_document(document_id)["content_kind"] == (
            "full_report_v1"
        )
        restarted = IngestionService(
            configured,
            restarted_storage,
            runtime.index,
            runtime.logger,
        )
        assert restarted.process_next() is True
        assert runtime.storage.get_document(document_id)["status"] == "ready"
        assert old_child_ids.isdisjoint(runtime.index.children)
        assert runtime.index.children
        assert {
            metadata["digest_section"]
            for _text, metadata in runtime.index.children.values()
        } == {*full_report_payload()["sections"], "decision.structured"}
        assert all(
            metadata["content_kind"] == "full_report_v1"
            and metadata["report_schema"] == "full_report_v1"
            and metadata["contains_decision_content"] is True
            and metadata["full_report_hash"]
            == full_report_payload()["full_report_hash"]
            for _text, metadata in runtime.index.children.values()
        )
        assert any(
            metadata["digest_section"] == "portfolio.decision"
            and metadata["report_stage"] == "risk"
            for _text, metadata in runtime.index.children.values()
        )

        before_ids = set(runtime.index.children)
        replay = client.post(
            endpoint,
            headers=authorised(),
            json={"documents": [full_report_payload()]},
        )
        assert replay.status_code == 202
        assert replay.json()["items"][0]["action"] == "unchanged"
        assert replay.json()["items"][0]["status"] == "ready"
        assert set(runtime.index.children) == before_ids
        assert runtime.storage.pending_vector_deletions(document_id) == []

        runtime.storage.mark_failed(document_id, "temporary_embedding_failure")
        failed_replay = client.post(
            endpoint,
            headers=authorised(),
            json={"documents": [full_report_payload()]},
        )
        assert failed_replay.json()["items"][0]["action"] == "unchanged"
        assert failed_replay.json()["items"][0]["status"] == "pending"
        assert runtime.ingestion.process_next() is True
        assert runtime.storage.get_document(document_id)["status"] == "ready"

        downgrade = client.post(
            "/internal/v1/digest-batches/full-20260821/documents:batch",
            headers=authorised(),
            json={"documents": [labelled_digest_payload()]},
        )
        assert downgrade.status_code == 409
        assert runtime.storage.get_document(document_id)["content_kind"] == (
            "full_report_v1"
        )


def test_sqlite_promotion_cas_prevents_a_second_full_generation(settings):
    """Separate service processes cannot both replace the same daily parent."""

    from rag_app.errors import PromotionConflictError
    from rag_app.full_reports import canonicalise_full_report

    configured = replace(settings, rag_service_api_key="test-secret")
    runtime = DigestRuntime(configured)

    with TestClient(create_app(configured, runtime)) as client:
        document_id = create_labelled_daily_and_seal(client, runtime)

    first = canonicalise_full_report(full_report_payload())
    promoted = runtime.storage.replace_digest_document(
        digest=first,
        batch_id="full-20260821",
        require_daily_parent=True,
    )
    assert promoted["id"] == document_id
    assert promoted["_promotion_action"] == "promoted"

    replay = runtime.storage.replace_digest_document(
        digest=first,
        batch_id="full-20260821",
        require_daily_parent=True,
    )
    assert replay["_promotion_action"] == "unchanged"

    second = canonicalise_full_report(full_report_payload(suffix=" generation 2"))
    with pytest.raises(PromotionConflictError) as conflict:
        runtime.storage.replace_digest_document(
            digest=second,
            batch_id="full-20260821",
            require_daily_parent=True,
        )
    assert conflict.value.error_code == "full_report_already_promoted"
    assert runtime.storage.get_document(document_id)["full_report_hash"] == (
        first.full_report_hash
    )


def test_daily_replace_cas_cannot_downgrade_a_concurrent_full_promotion(
    settings, monkeypatch
):
    """A POST admitted while open must not write after another process seals."""

    from rag_app.full_reports import canonicalise_full_report

    configured = replace(settings, rag_service_api_key="test-secret")
    runtime = DigestRuntime(configured)
    batch_endpoint = "/internal/v1/digest-batches/full-20260821"

    with TestClient(create_app(configured, runtime)) as client:
        assert client.put(
            batch_endpoint,
            headers=authorised(),
            json={
                "source": "tradingagents_daily_research",
                "analysis_date": "2026-08-21",
                "cutoff": "2026-08-21T16:45:00+07:00",
                "target_count": 1,
                "research_mode": "eod",
                "data_provenance": "eod_cutoff",
            },
        ).status_code == 200
        created = client.post(
            f"{batch_endpoint}/documents:batch",
            headers=authorised(),
            json={"documents": [labelled_digest_payload()]},
        )
        document_id = created.json()["items"][0]["document_id"]
        full = canonicalise_full_report(full_report_payload())
        original_replace = runtime.storage.replace_digest_document
        raced = False

        def seal_and_promote_before_daily_write(**kwargs):
            nonlocal raced
            if not raced and not kwargs.get("require_daily_parent", False):
                raced = True
                assert runtime.storage.seal_digest_batch("full-20260821")[0] == (
                    "sealed"
                )
                promoted = original_replace(
                    digest=full,
                    batch_id="full-20260821",
                    require_daily_parent=True,
                )
                assert promoted["_promotion_action"] == "promoted"
            return original_replace(**kwargs)

        monkeypatch.setattr(
            runtime.storage,
            "replace_digest_document",
            seal_and_promote_before_daily_write,
        )
        response = client.post(
            f"{batch_endpoint}/documents:batch",
            headers=authorised(),
            json={
                "documents": [
                    digest_payload(
                        updated="2026-08-21T18:00:00+07:00",
                        summary="Daily update admitted before the seal",
                    )
                ]
            },
        )
        assert response.status_code == 409
        current = runtime.storage.get_document(document_id)
        assert current["content_kind"] == "full_report_v1"
        assert current["full_report_hash"] == full.full_report_hash
        assert current["digest_hash"] is None


def test_daily_watermark_cas_cannot_mutate_a_concurrent_full_promotion(
    settings, monkeypatch
):
    from rag_app.full_reports import canonicalise_full_report

    configured = replace(settings, rag_service_api_key="test-secret")
    runtime = DigestRuntime(configured)
    endpoint = "/internal/v1/digest-batches/full-20260821/documents:batch"

    with TestClient(create_app(configured, runtime)) as client:
        document_id = create_labelled_daily_and_seal(client, runtime)
        full = canonicalise_full_report(full_report_payload())
        original_replace = runtime.storage.replace_digest_document
        original_advance = runtime.storage.advance_digest_source

        def promote_before_daily_advance(document_id, **kwargs):
            promoted = original_replace(
                digest=full,
                batch_id="full-20260821",
                require_daily_parent=True,
            )
            assert promoted["_promotion_action"] == "promoted"
            return original_advance(document_id, **kwargs)

        monkeypatch.setattr(
            runtime.storage,
            "advance_digest_source",
            promote_before_daily_advance,
        )
        response = client.post(
            endpoint,
            headers=authorised(),
            json={
                "documents": [
                    labelled_digest_payload(updated="2026-08-21T18:00:00+07:00")
                ]
            },
        )
        assert response.status_code == 409
        current = runtime.storage.get_document(document_id)
        assert current["content_kind"] == "full_report_v1"
        assert current["full_report_hash"] == full.full_report_hash
        assert current["digest_hash"] is None
        assert current["source_updated_at"] == full.source_updated_at


def test_sealed_daily_retry_cas_never_requeues_a_promoted_full_report(
    settings, monkeypatch
):
    from rag_app.full_reports import canonicalise_full_report

    configured = replace(settings, rag_service_api_key="test-secret")
    runtime = DigestRuntime(configured)
    endpoint = "/internal/v1/digest-batches/full-20260821/documents:batch"

    with TestClient(create_app(configured, runtime)) as client:
        document_id = create_labelled_daily_and_seal(client, runtime)
        runtime.storage.mark_failed(document_id, "temporary_embedding_failure")
        full = canonicalise_full_report(full_report_payload())
        original_replace = runtime.storage.replace_digest_document
        original_requeue = runtime.storage.requeue_daily_document

        def promote_before_daily_requeue(document_id, **kwargs):
            promoted = original_replace(
                digest=full,
                batch_id="full-20260821",
                require_daily_parent=True,
            )
            assert promoted["_promotion_action"] == "promoted"
            return original_requeue(document_id, **kwargs)

        monkeypatch.setattr(
            runtime.storage,
            "requeue_daily_document",
            promote_before_daily_requeue,
        )
        response = client.post(
            endpoint,
            headers=authorised(),
            json={"documents": [labelled_digest_payload()]},
        )
        assert response.status_code == 409
        current = runtime.storage.get_document(document_id)
        assert current["content_kind"] == "full_report_v1"
        assert current["full_report_hash"] == full.full_report_hash
        assert current["status"] == "pending"


def test_exact_full_retry_does_not_reset_a_concurrent_writer_claim(
    settings, monkeypatch
):
    configured = replace(settings, rag_service_api_key="test-secret")
    runtime = DigestRuntime(configured)
    endpoint = "/internal/v1/digest-batches/full-20260821/documents:promote"

    with TestClient(create_app(configured, runtime)) as client:
        document_id = create_labelled_daily_and_seal(client, runtime)
        first = client.post(
            endpoint,
            headers=authorised(),
            json={"documents": [full_report_payload()]},
        )
        assert first.json()["items"][0]["action"] == "promoted"
        runtime.storage.mark_failed(document_id, "temporary_embedding_failure")
        original_requeue = runtime.storage.requeue_full_report

        def requeue_then_claim(document_id, **kwargs):
            pending = original_requeue(document_id, **kwargs)
            assert pending["status"] == "pending"
            claimed = runtime.storage.claim_next_pending()
            assert claimed is not None and claimed["id"] == document_id
            return original_requeue(document_id, **kwargs)

        monkeypatch.setattr(
            runtime.storage,
            "requeue_full_report",
            requeue_then_claim,
        )
        replay = client.post(
            endpoint,
            headers=authorised(),
            json={"documents": [full_report_payload()]},
        )
        assert replay.status_code == 202
        assert replay.json()["items"][0]["action"] == "unchanged"
        assert replay.json()["items"][0]["status"] == "indexing"
        assert runtime.storage.get_document(document_id)["status"] == "indexing"


def test_promotion_requires_seal_and_returns_safe_stale_and_conflict_actions(settings):
    configured = replace(settings, rag_service_api_key="test-secret")
    runtime = DigestRuntime(configured)
    app = create_app(configured, runtime)
    endpoint = "/internal/v1/digest-batches/full-20260821/documents:promote"

    with TestClient(app) as client:
        assert client.put(
            "/internal/v1/digest-batches/full-20260821",
            headers=authorised(),
            json={
                "source": "tradingagents_daily_research",
                "analysis_date": "2026-08-21",
                "cutoff": "2026-08-21T16:45:00+07:00",
                "target_count": 1,
                "research_mode": "eod",
                "data_provenance": "eod_cutoff",
            },
        ).status_code == 200
        assert client.post(
            "/internal/v1/digest-batches/full-20260821/documents:batch",
            headers=authorised(),
            json={"documents": [labelled_digest_payload()]},
        ).status_code == 202
        assert client.post(
            endpoint,
            headers=authorised(),
            json={"documents": [full_report_payload()]},
        ).status_code == 409
        assert client.post(
            "/internal/v1/digest-batches/full-20260821:seal",
            headers=authorised(),
        ).status_code == 200

        stale_payload = full_report_payload(
            updated="2026-08-21T17:00:00+07:00"
        )
        stale = client.post(
            endpoint,
            headers=authorised(),
            json={"documents": [stale_payload]},
        ).json()["items"][0]
        assert stale["action"] == "stale"
        assert stale["error_code"] == "source_updated_at_not_newer"

        wrong_parent = full_report_payload()
        wrong_parent["parent_identity_hash"] = "f" * 64
        conflict = client.post(
            endpoint,
            headers=authorised(),
            json={"documents": [wrong_parent]},
        ).json()["items"][0]
        assert conflict["action"] == "conflict"
        assert conflict["error_code"] == "parent_identity_hash_mismatch"

        first = client.post(
            endpoint,
            headers=authorised(),
            json={"documents": [full_report_payload()]},
        )
        assert first.json()["items"][0]["action"] == "promoted"
        changed = full_report_payload(suffix=" đã thay đổi")
        second = client.post(
            endpoint,
            headers=authorised(),
            json={"documents": [changed]},
        ).json()["items"][0]
        assert second["action"] == "conflict"
        assert second["error_code"] == "full_report_already_promoted"


def test_promotion_rechecks_worker_claim_inside_sqlite_transaction(
    settings, monkeypatch
):
    configured = replace(settings, rag_service_api_key="test-secret")
    runtime = DigestRuntime(configured)
    app = create_app(configured, runtime)
    batch_endpoint = "/internal/v1/digest-batches/full-race"
    promote_endpoint = f"{batch_endpoint}/documents:promote"

    with TestClient(app) as client:
        assert client.put(
            batch_endpoint,
            headers=authorised(),
            json={
                "source": "tradingagents_daily_research",
                "analysis_date": "2026-08-21",
                "cutoff": "2026-08-21T16:45:00+07:00",
                "target_count": 1,
                "research_mode": "eod",
                "data_provenance": "eod_cutoff",
            },
        ).status_code == 200
        created = client.post(
            f"{batch_endpoint}/documents:batch",
            headers=authorised(),
            json={"documents": [labelled_digest_payload()]},
        ).json()
        document_id = created["items"][0]["document_id"]
        assert client.post(
            f"{batch_endpoint}:seal", headers=authorised()
        ).status_code == 200

        original_replace = runtime.storage.replace_digest_document

        def claim_then_replace(*, digest, batch_id, require_daily_parent=False):
            claimed = runtime.storage.claim_next_pending()
            assert claimed["id"] == document_id
            return original_replace(
                digest=digest,
                batch_id=batch_id,
                require_daily_parent=require_daily_parent,
            )

        monkeypatch.setattr(
            runtime.storage, "replace_digest_document", claim_then_replace
        )
        raced = client.post(
            promote_endpoint,
            headers=authorised(),
            json={"documents": [full_report_payload()]},
        )
        assert raced.status_code == 202
        assert raced.json()["items"][0]["action"] == "conflict"
        assert raced.json()["items"][0]["error_code"] == "parent_document_busy"
        assert runtime.storage.get_document(document_id)["content_kind"] == (
            "daily_digest_v1"
        )
        assert len(runtime.storage.digest_sections(document_id)) == 9

        monkeypatch.setattr(
            runtime.storage, "replace_digest_document", original_replace
        )
        assert runtime.storage.reset_interrupted_jobs() == 1
        retried = client.post(
            promote_endpoint,
            headers=authorised(),
            json={"documents": [full_report_payload()]},
        )
        assert retried.json()["items"][0]["action"] == "promoted"


def test_full_report_contract_filters_citation_and_raw_isolation(settings):
    configured = replace(settings, rag_service_api_key="test-secret")
    runtime = DigestRuntime(configured)
    capture = CapturingLogger()
    runtime.logger = capture
    runtime.ingestion.logger = capture
    app = create_app(configured, runtime)
    endpoint = "/internal/v1/digest-batches/full-20260821/documents:promote"

    with TestClient(app) as client:
        document_id = create_labelled_daily_and_seal(client, runtime)
        unsafe = full_report_payload()
        unsafe["raw_evidence"] = {
            "title": "FULL_RAW_SENTINEL",
            "body": "FULL_RAW_SENTINEL",
            "prompt": "FULL_RAW_SENTINEL",
        }
        rejected = client.post(
            endpoint,
            headers=authorised(),
            json={"documents": [unsafe]},
        )
        assert rejected.status_code == 422
        assert runtime.storage.get_document(document_id)["content_kind"] == (
            "daily_digest_v1"
        )

        for forbidden_text in (
            "FULL_RAW_SENTINEL",
            "Chi tiết tại https://example.invalid/raw-evidence",
            "test-secret",
        ):
            unsafe_text = full_report_payload()
            unsafe_text["sections"]["portfolio.decision"]["text"] = forbidden_text
            unsafe_text["full_report_hash"] = sha256_json(unsafe_text["sections"])
            rejected_text = client.post(
                endpoint,
                headers=authorised(),
                json={"documents": [unsafe_text]},
            )
            assert rejected_text.status_code == 400
            assert runtime.storage.get_document(document_id)["content_kind"] == (
                "daily_digest_v1"
            )

        accepted = client.post(
            endpoint,
            headers=authorised(),
            json={"documents": [full_report_payload()]},
        )
        assert accepted.json()["items"][0]["action"] == "promoted"
        assert runtime.ingestion.process_next() is True
        catalog = client.get(
            "/api/documents?page=1&page_size=50&source_type=trading_digest"
            "&content_kind=full_report_v1&report_stage=risk"
            "&digest_section=portfolio.decision"
        )
        assert catalog.status_code == 200
        assert catalog.json()["total"] == 1
        assert catalog.json()["items"][0]["content_kind"] == "full_report_v1"
        assert client.get(
            "/api/documents?page=1&page_size=50&content_kind=daily_digest_v1"
        ).json()["total"] == 0

    with sqlite3.connect(configured.sqlite_path) as db:
        persisted = repr(
            {
                table: db.execute(f"SELECT * FROM {table}").fetchall()
                for table in ("documents", "digest_sections", "parents", "children")
            }
        )
    for forbidden in (
        "FULL_RAW_SENTINEL",
        "https://example.invalid/raw-evidence",
        "test-secret",
    ):
        assert forbidden not in persisted
        assert forbidden not in repr(runtime.index.children)
        assert forbidden not in repr(capture.records)
        for artifact in configured.data_dir.rglob("*"):
            if artifact.is_file():
                assert forbidden.encode() not in artifact.read_bytes()

    source = RetrievedSource(
        id="1",
        document_id=document_id,
        filename="HPG.full.digest",
        page=0,
        parent_id="parent-full",
        text="Quyết định danh mục có điều kiện.",
        score=0.9,
        source_type="trading_digest",
        ticker="MWG",
        analysis_date="2026-08-18",
        analysis_cutoff="2026-08-18T15:45:00+07:00",
        digest_section="portfolio.decision",
        report_stage="risk",
        section_status="complete",
        content_kind="full_report_v1",
        report_schema="full_report_v1",
        contains_decision_content=True,
        research_mode="eod",
        data_provenance="eod_cutoff",
    )
    public = source.public(cited=True)
    assert public["analysis_badge"] == "Full Analysis"
    assert public["report_stage"] == "risk"
    assert public["label"] == (
        "MWG · 18/08/2026 15:45 · Risk · Portfolio decision"
    )
    context = format_context([source])
    assert 'content_kind="full_report_v1"' in context
    assert 'report_stage="risk"' in context


def test_full_report_canonicaliser_is_fail_closed_for_direct_callers():
    from rag_app.errors import ValidationError
    from rag_app.full_reports import canonicalise_full_report

    canonical = canonicalise_full_report(full_report_payload())
    assert canonical.content_kind == "full_report_v1"
    assert canonical.full_report_hash == full_report_payload()["full_report_hash"]

    malformed = []
    extra = full_report_payload()
    extra["raw_title"] = "RAW_DIRECT_SENTINEL"
    malformed.append(extra)
    missing_hash = full_report_payload()
    missing_hash["parent_identity_hash"] = None
    malformed.append(missing_hash)
    boolean_generation = full_report_payload()
    boolean_generation["execution_generation"] = True
    malformed.append(boolean_generation)
    wrong_pipeline = full_report_payload()
    wrong_pipeline["pipeline_version"] = "gx_full_v2"
    malformed.append(wrong_pipeline)
    wrong_sections_hash = full_report_payload()
    wrong_sections_hash["sections"]["portfolio.decision"]["text"] += " changed"
    malformed.append(wrong_sections_hash)
    missing_core = full_report_payload()
    missing_core["stage_status"]["risk"] = "partial"
    malformed.append(missing_core)
    mismatched_section = full_report_payload()
    mismatched_section["sections"]["analyst.market"]["status"] = "partial"
    mismatched_section["full_report_hash"] = sha256_json(
        mismatched_section["sections"]
    )
    malformed.append(mismatched_section)
    inconsistent_overall = full_report_payload()
    inconsistent_overall["stage_status"]["market"] = "partial"
    malformed.append(inconsistent_overall)
    sentinel_text = full_report_payload()
    sentinel_text["sections"]["portfolio.decision"]["text"] = (
        "FULL_CREDENTIAL_SENTINEL"
    )
    sentinel_text["full_report_hash"] = sha256_json(sentinel_text["sections"])
    malformed.append(sentinel_text)
    url_text = full_report_payload()
    url_text["sections"]["portfolio.decision"]["text"] = (
        "Tham khảo https://example.invalid/report"
    )
    url_text["full_report_hash"] = sha256_json(url_text["sections"])
    malformed.append(url_text)

    for payload in malformed:
        with pytest.raises(ValidationError):
            canonicalise_full_report(payload)

    partial = full_report_payload()
    partial["status"] = "partial"
    partial["stage_status"]["market"] = "partial"
    partial["sections"]["analyst.market"]["status"] = "partial"
    partial["full_report_hash"] = sha256_json(partial["sections"])
    assert canonicalise_full_report(partial).digest_status == "partial"

    # The URL/sentinel guard is scoped to Full Analysis; daily compatibility
    # remains unchanged for already-supported sanitized digest text.
    from rag_app.digests import canonicalise_digest

    daily = digest_payload(summary="Theo dõi https://example.invalid FULL_RAW_SENTINEL")
    assert canonicalise_digest(
        daily, batch_source="tradingagents_daily_research"
    ).sections["summary"].endswith("FULL_RAW_SENTINEL")


def test_decision_projection_handles_field_order_paragraphs_and_target_modes():
    from rag_app.full_reports import canonicalise_full_report

    available = canonicalise_full_report(structured_full_report_payload())
    assert available.decision["schema_version"] == 1
    assert available.projection_version == 2
    assert available.decision_status == "complete"
    assert available.decision["warnings"] == []
    assert available.decision["portfolio"] == {
        "rating": "Overweight",
        "executive_summary": "Tích lũy dần theo ba đợt.",
        "investment_thesis": (
            "Luận điểm thứ nhất.\n\n"
            "Luận điểm thứ hai vẫn thuộc cùng trường thesis."
        ),
        "time_horizon": "3-6 months",
        "price_target_status": "available",
        "price_target": 77000,
        "price_target_currency": "VND",
        "price_target_rationale": "Dựa trên tăng trưởng lợi nhuận và định giá.",
        "price_target_unavailable_reason": None,
    }
    assert available.decision["trading"] == {
        "action": "Buy",
        "reasoning": "Định giá và động lượng hỗ trợ giao dịch.",
        "entry_price": 70700,
        "stop_loss": 66500,
        "position_sizing": "3-5% danh mục",
    }
    assert json.loads(available.canonical_json) == available.canonical_report
    assert sha256_json(available.canonical_report) == available.canonical_sha256
    assert sha256_json(available.decision) == available.decision_sha256

    for raw_currency in ("Unavailable", "VND", "USD"):
        unavailable_payload = structured_full_report_payload(target_available=False)
        unavailable_payload["sections"]["portfolio.decision"]["text"] = (
            unavailable_payload["sections"]["portfolio.decision"]["text"].replace(
                "**Price Target Currency**: Unavailable",
                f"**Price Target Currency**: {raw_currency}",
            )
        )
        unavailable_payload["full_report_hash"] = sha256_json(
            unavailable_payload["sections"]
        )
        unavailable = canonicalise_full_report(unavailable_payload)
        portfolio = unavailable.decision["portfolio"]
        assert unavailable.decision_status == "complete"
        assert unavailable.decision["warnings"] == []
        assert portfolio["price_target_status"] == "unavailable"
        assert portfolio["price_target"] is None
        assert portfolio["price_target_currency"] is None
        assert portfolio["price_target_rationale"] is None
        assert portfolio["price_target_unavailable_reason"].startswith("Không có")

    for old, new in (
        ("**Price Target Currency**: Unavailable", "**Price Target Currency**: VND/USD"),
        (
            "**Price Target Unavailable Reason**: Không có dữ liệu định giá đủ tin cậy.",
            "**Price Target Unavailable Reason**: Unavailable",
        ),
    ):
        invalid_unavailable = structured_full_report_payload(target_available=False)
        invalid_unavailable["sections"]["portfolio.decision"]["text"] = (
            invalid_unavailable["sections"]["portfolio.decision"]["text"].replace(
                old, new
            )
        )
        invalid_unavailable["full_report_hash"] = sha256_json(
            invalid_unavailable["sections"]
        )
        invalid_projection = canonicalise_full_report(invalid_unavailable).decision
        assert invalid_projection["status"] == "partial"
        assert invalid_projection["portfolio"]["price_target_status"] == "unknown"
        assert "portfolio_price_target_contract_invalid" in invalid_projection["warnings"]

    invalid_vnd = structured_full_report_payload()
    invalid_vnd["sections"]["portfolio.decision"]["text"] = invalid_vnd[
        "sections"
    ]["portfolio.decision"]["text"].replace("77000", "77000.5")
    invalid_vnd["full_report_hash"] = sha256_json(invalid_vnd["sections"])
    projected = canonicalise_full_report(invalid_vnd).decision
    assert projected["status"] == "partial"
    assert projected["portfolio"]["price_target_status"] == "unknown"
    assert projected["portfolio"]["price_target"] is None
    assert "portfolio_price_target_contract_invalid" in projected["warnings"]

    legacy = canonicalise_full_report(full_report_payload()).decision
    assert legacy["status"] == "partial"
    assert legacy["portfolio"]["rating"] is None
    assert legacy["trading"]["action"] is None


def test_full_report_payload_endpoints_and_structured_retrieval(settings):
    configured = replace(settings, rag_service_api_key="test-secret")
    runtime = DigestRuntime(configured)

    with TestClient(create_app(configured, runtime)) as client:
        document_id = create_labelled_daily_and_seal(client, runtime)
        promoted = client.post(
            "/internal/v1/digest-batches/full-20260821/documents:promote",
            headers=authorised(),
            json={"documents": [structured_full_report_payload()]},
        )
        assert promoted.status_code == 202
        assert client.get(f"/api/documents/{document_id}/full-report").status_code == 409
        assert (
            client.get(
                "/internal/v1/digest-batches/full-20260821/documents/HPG/full-report"
            ).status_code
            == 401
        )

        assert runtime.ingestion.process_next() is True
        public = client.get(f"/api/documents/{document_id}/full-report")
        assert public.status_code == 200
        data = public.json()["data"]
        assert set(data) == {
            "document_id",
            "canonical_report_sha256",
            "decision_sha256",
            "report",
            "decision",
        }
        assert data["document_id"] == document_id
        assert data["report"]["full_report_hash"] == structured_full_report_payload()[
            "full_report_hash"
        ]
        assert len(data["report"]["sections"]) == 12
        assert data["decision"]["portfolio"]["price_target"] == 77000
        assert data["decision"]["trading"]["action"] == "Buy"
        serialized = json.dumps(data, ensure_ascii=False).casefold()
        assert "session.json" not in serialized
        assert "credential" not in serialized

        internal = client.get(
            "/internal/v1/digest-batches/full-20260821/documents/HPG/full-report",
            headers=authorised(),
        )
        assert internal.status_code == 200
        assert internal.json() == public.json()

        catalog = client.get(
            "/api/documents?page=1&digest_section=decision.structured"
        )
        assert catalog.status_code == 200
        assert [item["id"] for item in catalog.json()["items"]] == [document_id]
        lexical = runtime.storage.lexical_search(
            "77000",
            k=5,
            filters={"digest_sections": ["decision.structured"]},
        )
        assert lexical
        assert {item["document_id"] for item in lexical} == {document_id}
        assert len(runtime.storage.digest_sections(document_id)) == 12
        assert any(
            metadata["digest_section"] == "decision.structured"
            and metadata["report_stage"] == "risk"
            and metadata["section_status"] == "complete"
            for _text, metadata in runtime.index.children.values()
        )


def test_full_report_payload_insert_rolls_back_promotion_transaction(settings):
    from rag_app.full_reports import canonicalise_full_report

    configured = replace(settings, rag_service_api_key="test-secret")
    runtime = DigestRuntime(configured)
    with TestClient(create_app(configured, runtime)) as client:
        document_id = create_labelled_daily_and_seal(client, runtime)
        invalid = replace(
            canonicalise_full_report(structured_full_report_payload()),
            decision_status="invalid",
        )
        with pytest.raises(sqlite3.IntegrityError):
            runtime.storage.replace_digest_document(
                digest=invalid,
                batch_id="full-20260821",
                require_daily_parent=True,
            )

        document = runtime.storage.get_document(document_id)
        assert document["content_kind"] == "daily_digest_v1"
        assert len(runtime.storage.digest_sections(document_id)) == 9
        assert runtime.storage.get_full_report_payload(document_id) is None


def test_full_report_backfill_is_offline_and_idempotent(
    settings, monkeypatch, capsys
):
    from rag_app.cli import backfill_full_reports_command

    configured = replace(settings, rag_service_api_key="test-secret")
    runtime = DigestRuntime(configured)
    with TestClient(create_app(configured, runtime)) as client:
        document_id = create_labelled_daily_and_seal(client, runtime)
        assert (
            client.post(
                "/internal/v1/digest-batches/full-20260821/documents:promote",
                headers=authorised(),
                json={"documents": [structured_full_report_payload()]},
            ).status_code
            == 202
        )
        assert runtime.ingestion.process_next() is True

    with runtime.storage.connect() as db:
        db.execute(
            "DELETE FROM full_report_payloads WHERE document_id=?", (document_id,)
        )
    monkeypatch.setenv("RAG_BASE_DIR", str(configured.base_dir))
    monkeypatch.setenv("RAG_DATA_DIR", str(configured.data_dir))
    monkeypatch.setenv("RAG_PAPERS_DIR", str(configured.papers_dir))

    assert backfill_full_reports_command(apply=False) == 0
    dry_run = json.loads(capsys.readouterr().out)
    assert dry_run == {
        "mode": "dry-run",
        "scanned": 1,
        "stored": 1,
        "partial": 0,
        "requeued": 1,
        "failed": 0,
    }
    assert runtime.storage.get_full_report_payload(document_id) is None
    assert runtime.storage.get_document(document_id)["status"] == "ready"

    assert backfill_full_reports_command(apply=True) == 0
    applied = json.loads(capsys.readouterr().out)
    assert applied["stored"] == 1
    assert applied["requeued"] == 1
    assert applied["failed"] == 0
    assert runtime.storage.get_document(document_id)["status"] == "pending"
    assert runtime.ingestion.process_next() is True
    assert runtime.storage.full_report_payload_health()[
        "pending_ready_decision_indexes"
    ] == 0

    assert backfill_full_reports_command(apply=True) == 0
    replay = json.loads(capsys.readouterr().out)
    assert replay["scanned"] == 1
    assert replay["stored"] == 0
    assert replay["requeued"] == 0
    assert replay["failed"] == 0

    with runtime.storage.connect() as db:
        db.execute(
            "UPDATE full_report_payloads SET canonical_json='{}' WHERE document_id=?",
            (document_id,),
        )
    assert backfill_full_reports_command(apply=True) == 0
    repaired = json.loads(capsys.readouterr().out)
    assert repaired["stored"] == 1
    assert repaired["requeued"] == 0
    assert json.loads(
        runtime.storage.get_full_report_payload(document_id)["canonical_json"]
    )["content_kind"] == "full_report_v1"


def test_full_report_backfill_upgrades_projection_without_changing_identity(
    settings, monkeypatch, capsys
):
    from rag_app.cli import backfill_full_reports_command

    configured = replace(settings, rag_service_api_key="test-secret")
    runtime = DigestRuntime(configured)
    with TestClient(create_app(configured, runtime)) as client:
        document_id = create_labelled_daily_and_seal(client, runtime)
        assert (
            client.post(
                "/internal/v1/digest-batches/full-20260821/documents:promote",
                headers=authorised(),
                json={"documents": [structured_full_report_payload()]},
            ).status_code
            == 202
        )
        assert runtime.ingestion.process_next() is True

    before_document = runtime.storage.get_document(document_id)
    before_payload = runtime.storage.get_full_report_payload(document_id)
    with runtime.storage.connect() as db:
        db.execute(
            "UPDATE full_report_payloads "
            "SET projection_version=1, decision_index_version=1 "
            "WHERE document_id=?",
            (document_id,),
        )

    monkeypatch.setenv("RAG_BASE_DIR", str(configured.base_dir))
    monkeypatch.setenv("RAG_DATA_DIR", str(configured.data_dir))
    monkeypatch.setenv("RAG_PAPERS_DIR", str(configured.papers_dir))

    assert backfill_full_reports_command(apply=False) == 0
    dry_run = json.loads(capsys.readouterr().out)
    assert dry_run == {
        "mode": "dry-run",
        "scanned": 1,
        "stored": 1,
        "partial": 0,
        "requeued": 1,
        "failed": 0,
    }

    assert backfill_full_reports_command(apply=True) == 0
    applied = json.loads(capsys.readouterr().out)
    assert applied["stored"] == 1
    assert applied["requeued"] == 1
    assert applied["failed"] == 0

    after_document = runtime.storage.get_document(document_id)
    after_payload = runtime.storage.get_full_report_payload(document_id)
    assert after_document["id"] == before_document["id"]
    assert after_document["full_report_hash"] == before_document["full_report_hash"]
    assert after_payload["canonical_sha256"] == before_payload["canonical_sha256"]
    assert after_payload["projection_version"] == 2
    assert json.loads(after_payload["decision_json"])["schema_version"] == 1

    assert runtime.ingestion.process_next() is True
    assert backfill_full_reports_command(apply=True) == 0
    replay = json.loads(capsys.readouterr().out)
    assert replay["stored"] == 0
    assert replay["requeued"] == 0
    assert replay["failed"] == 0
