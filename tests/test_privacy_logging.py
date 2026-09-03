from __future__ import annotations

import json
import logging

from rag_app.logging import PrivacyJsonFormatter


def test_privacy_formatter_only_emits_allowlisted_rag_metadata():
    record = logging.LogRecord(
        "rag_app",
        logging.INFO,
        __file__,
        1,
        "ignored message containing a question",
        (),
        None,
    )
    for key, value in {
        "event": "rag_stage",
        "request_id": "request-1",
        "stage": "retrieve",
        "route": "mixed",
        "round": 1,
        "duration_ms": 12.5,
        "candidate_count": 40,
        "source_count": 5,
        "confidence_bucket": "high",
        "reason_code": "passed",
        # These values must never cross the formatter trust boundary.
        "query": "HPG chịu rủi ro gì?",
        "answer": "private answer",
        "document_id": "doc-secret",
        "chunk_ids": ["child-secret"],
        "entity_label": "HPG",
        "graph_content": "HPG EXPOSED_TO JPY",
        "score": 0.99,
        "model_response": '{"private":"raw verifier output"}',
        "error_detail": "private parser failure detail",
    }.items():
        setattr(record, key, value)

    payload = json.loads(PrivacyJsonFormatter().format(record))

    assert set(payload) == {
        "timestamp",
        "level",
        "event",
        "request_id",
        "stage",
        "route",
        "round",
        "duration_ms",
        "candidate_count",
        "source_count",
        "confidence_bucket",
        "reason_code",
    }
