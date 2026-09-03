from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from rag_app.evaluation import (
    AdvancedEvalCase,
    AdvancedEvalResult,
    AdvancedEvalValidationError,
    EvalEvidenceSpan,
    question_set_fingerprint,
    score_advanced_release_gate,
    validate_advanced_release_corpus,
)


def evidence(case_number: int) -> EvalEvidenceSpan:
    quote = f"Exact evidence for case {case_number}"
    return EvalEvidenceSpan(
        id=f"ev-{case_number}",
        document_id=f"doc-{case_number}",
        parent_id=f"parent-{case_number}",
        quote=quote,
        start_offset=5,
        end_offset=5 + len(quote),
        quote_sha256=hashlib.sha256(quote.encode()).hexdigest(),
        page=1,
    )


def release_fixture() -> tuple[list[AdvancedEvalCase], list[AdvancedEvalResult]]:
    cases: list[AdvancedEvalCase] = []
    results: list[AdvancedEvalResult] = []
    for number in range(150):
        case_id = f"case-{number:03d}"
        document_id = f"doc-{number}"
        if number == 0:
            case = AdvancedEvalCase.model_validate(
                {
                    "schema_version": 1,
                    "id": case_id,
                    "question": "An intentionally unanswerable local question?",
                    "intent": "lookup",
                    "expected_route": "hybrid",
                    "expected_tools": ["hybrid"],
                    "answerable": False,
                    "scope": {"document_ids": [document_id]},
                },
                strict=True,
            )
            result = AdvancedEvalResult(
                schema_version=1,
                id=case_id,
                route="hybrid",
                tools=["hybrid"],
                abstained=True,
                latency_ms=100,
            )
        else:
            span = evidence(number)
            is_graph = number == 2
            is_numeric = number == 1
            claim: dict[str, object] = {
                "id": f"claim-{number}",
                "evidence_ids": [span.id],
            }
            observed_claim: dict[str, object] = {
                "claim_id": f"claim-{number}",
                "support": "supported",
                "citation_evidence_ids": [span.id],
            }
            if is_numeric:
                numeric = {"value": "123.45", "unit": "VND", "period": "FY2026"}
                claim["numeric"] = numeric
                observed_claim["numeric"] = numeric
            entities = []
            relations = []
            resolved_entities = []
            observed_relations = []
            graph_evidence_ids = []
            if is_graph:
                entities = [
                    {"id": "security:AAA", "canonical_key": "AAA"},
                    {"id": "risk:jpy", "canonical_key": "jpy"},
                ]
                relations = [
                    {
                        "id": "rel-2",
                        "subject_entity_id": "security:AAA",
                        "predicate": "EXPOSED_TO",
                        "object_entity_id": "risk:jpy",
                        "evidence_ids": [span.id],
                    }
                ]
                resolved_entities = ["security:AAA", "risk:jpy"]
                observed_relations = [
                    {
                        "expected_relation_id": "rel-2",
                        "correct": True,
                        "evidence_ids": [span.id],
                    }
                ]
                graph_evidence_ids = [span.id]
            simple_non_graph = number == 3
            route = "graph" if is_graph else "hybrid"
            tools = ["graph"] if is_graph else ["hybrid"]
            case = AdvancedEvalCase.model_validate(
                {
                    "schema_version": 1,
                    "id": case_id,
                    "question": f"Question {number}?",
                    "intent": "relationship" if is_graph else "lookup",
                    "expected_route": route,
                    "expected_tools": tools,
                    "answerable": True,
                    "scope": {"document_ids": [document_id]},
                    "evidence": [span.model_dump()],
                    "claims": [claim],
                    "entities": entities,
                    "relations": relations,
                    "simple_non_graph": simple_non_graph,
                },
                strict=True,
            )
            result = AdvancedEvalResult.model_validate(
                {
                    "schema_version": 1,
                    "id": case_id,
                    "route": route,
                    "tools": tools,
                    "candidate_evidence_ids": [span.id],
                    "context_evidence_ids": [span.id],
                    "claims": [observed_claim],
                    "citations": [
                        {
                            "citation_id": "1",
                            "document_id": document_id,
                            "evidence_id": span.id,
                            "supported": True,
                        }
                    ],
                    "abstained": False,
                    "retrieved_document_ids": [document_id],
                    "resolved_entity_ids": resolved_entities,
                    "graph_relations": observed_relations,
                    "graph_evidence_ids": graph_evidence_ids,
                    "latency_ms": 100,
                    "baseline_latency_ms": 100 if simple_non_graph else None,
                },
                strict=True,
            )
        cases.append(case)
        results.append(result)
    return cases, results


LEGACY_30_HASH = "a" * 64
LEGACY_30_METRICS = {
    "parent_recall_at_5": 1.0,
    "citation_correct_page": 0.8947,
    "abstention_accuracy": 0.6667,
    "answers_with_citation": 1.0,
    "irrelevant_citations": 6,
}


def write_legacy_result(
    path,
    *,
    metrics=None,
    question_hash=LEGACY_30_HASH,
    failed_case_ids=None,
) -> None:
    path.write_text(
        json.dumps(
            {
                "cases": 30,
                "question_set_sha256": question_hash,
                "metrics": metrics or LEGACY_30_METRICS,
                "failed_case_ids": failed_case_ids or [],
            }
        )
    )


def test_release_gate_requires_150_cases_only_in_release_validation():
    cases, _ = release_fixture()

    with pytest.raises(AdvancedEvalValidationError, match="ít nhất 150"):
        validate_advanced_release_corpus(cases[:149])


def test_evidence_span_hash_and_offsets_are_part_of_the_schema():
    with pytest.raises(ValueError, match="quote_sha256"):
        EvalEvidenceSpan(
            id="ev",
            document_id="doc",
            parent_id="parent",
            quote="exact",
            start_offset=0,
            end_offset=5,
            quote_sha256="0" * 64,
        )


def test_release_gate_scores_all_approved_metrics(tmp_path):
    cases, results = release_fixture()
    legacy = tmp_path / "legacy-current.json"
    baseline = tmp_path / "legacy-baseline.json"
    write_legacy_result(legacy)
    write_legacy_result(baseline)

    report = score_advanced_release_gate(
        cases,
        results,
        legacy_result_path=legacy,
        legacy_baseline_path=baseline,
    )

    assert report["passed"] is True
    assert report["failed_gates"] == []
    assert report["metrics"]["candidate_recall_at_20"] == 1.0
    assert report["metrics"]["numeric_exactness"] == 1.0
    assert report["metrics"]["graph_relation_precision"] == 1.0
    assert report["metrics"]["simple_non_graph_p95_ratio"] == 1.0
    # Đây là no-regression, không phải áp acceptance tuyệt đối lên legacy suite.
    assert report["legacy_regression"]["passed"] is True
    assert (
        report["legacy_regression"]["comparison"]["abstention_accuracy"][
            "baseline"
        ]
        == 0.6667
    )


def test_release_gate_detects_scope_escape_and_unsupported_citation(tmp_path):
    cases, results = release_fixture()
    legacy = tmp_path / "legacy-current.json"
    baseline = tmp_path / "legacy-baseline.json"
    write_legacy_result(legacy)
    write_legacy_result(baseline)
    bad = results[0].model_dump()
    bad["citations"] = [
        {
            "citation_id": "bad",
            "document_id": "outside-scope",
            "supported": False,
        }
    ]
    results[0] = AdvancedEvalResult.model_validate(bad, strict=True)

    report = score_advanced_release_gate(
        cases,
        results,
        legacy_result_path=legacy,
        legacy_baseline_path=baseline,
    )

    assert report["passed"] is False
    assert "scope_escapes" in report["failed_gates"]
    assert "unsupported_citations" in report["failed_gates"]


def test_release_gate_detects_regression_against_explicit_30_case_baseline(tmp_path):
    cases, results = release_fixture()
    legacy = tmp_path / "legacy-current.json"
    baseline = tmp_path / "legacy-baseline.json"
    write_legacy_result(baseline)
    regressed = {**LEGACY_30_METRICS, "citation_correct_page": 0.80}
    write_legacy_result(legacy, metrics=regressed)

    report = score_advanced_release_gate(
        cases,
        results,
        legacy_result_path=legacy,
        legacy_baseline_path=baseline,
    )

    assert report["passed"] is False
    assert "legacy_30_no_regression" in report["failed_gates"]
    assert report["legacy_regression"]["regressions"] == [
        "citation_correct_page"
    ]


def test_release_gate_rejects_a_new_failed_case_even_if_aggregates_do_not_drop(
    tmp_path,
):
    cases, results = release_fixture()
    legacy = tmp_path / "legacy-current.json"
    baseline = tmp_path / "legacy-baseline.json"
    write_legacy_result(baseline, failed_case_ids=["known-failure"])
    write_legacy_result(
        legacy,
        failed_case_ids=["known-failure", "new-regression"],
    )

    report = score_advanced_release_gate(
        cases,
        results,
        legacy_result_path=legacy,
        legacy_baseline_path=baseline,
    )

    assert "legacy_30_no_regression" in report["failed_gates"]
    assert report["legacy_regression"]["new_failed_case_ids"] == [
        "new-regression"
    ]


def test_legacy_current_and_baseline_must_reference_same_question_set(tmp_path):
    cases, results = release_fixture()
    legacy = tmp_path / "legacy-current.json"
    baseline = tmp_path / "legacy-baseline.json"
    write_legacy_result(legacy)
    write_legacy_result(baseline, question_hash="b" * 64)

    with pytest.raises(AdvancedEvalValidationError, match="questions.jsonl hash"):
        score_advanced_release_gate(
            cases,
            results,
            legacy_result_path=legacy,
            legacy_baseline_path=baseline,
        )


def test_committed_30_case_baseline_is_bound_to_the_current_questions_file():
    repository = Path(__file__).resolve().parents[1]
    questions = [
        json.loads(line)
        for line in (repository / "evals/questions.jsonl").read_text().splitlines()
        if line.strip()
    ]
    baseline = json.loads((repository / "evals/hybrid_full.json").read_text())

    assert baseline["cases"] == 30
    assert len(questions) == 30
    assert baseline["question_set_sha256"] == question_set_fingerprint(questions)
