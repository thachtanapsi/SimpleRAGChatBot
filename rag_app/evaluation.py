"""Bộ eval không dùng model judge/cloud; chấm retrieval, citation và abstention."""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import time
from collections import defaultdict
from dataclasses import replace
from datetime import date
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from .retrieval import ParentChildRetriever
from .runtime import Runtime


ABSTENTION_MARKERS = (
    "không tìm thấy bằng chứng phù hợp",
    "could not find sufficient evidence",
    "do not have information",
    "does not contain information",
    "sources do not contain",
    "not provided in the sources",
    "insufficient evidence",
)

ADVANCED_EVAL_SCHEMA_VERSION = 1
ADVANCED_RELEASE_MIN_CASES = 150
ADVANCED_RELEASE_TIMEOUT_MS = 120_000.0

ADVANCED_RELEASE_THRESHOLDS: dict[str, float | int] = {
    "candidate_recall_at_20": 0.95,
    "final_context_recall_at_5": 0.90,
    "citation_claim_support": 0.95,
    "unsupported_citations": 0,
    "numeric_exactness": 1.0,
    "abstention_accuracy": 0.90,
    "route_accuracy": 0.95,
    "scope_escapes": 0,
    "graph_entity_resolution": 0.95,
    "graph_relation_precision": 0.90,
    "graph_evidence_recall_at_10": 0.85,
    "simple_non_graph_p95_ratio": 2.0,
    "advanced_timeout_failures": 0,
}

LEGACY_REGRESSION_DIRECTIONS: dict[str, Literal["higher", "lower"]] = {
    "parent_recall_at_5": "higher",
    "citation_correct_page": "higher",
    "abstention_accuracy": "higher",
    "answers_with_citation": "higher",
    "irrelevant_citations": "lower",
}

EvalRoute = Literal["hybrid", "structured", "graph", "mixed"]
EvalTool = Literal["hybrid", "structured", "graph"]
EvalIntent = Literal["lookup", "decision", "comparison", "relationship", "summary"]

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

EVAL_FILTER_FIELDS = frozenset(
    {
        "source_types",
        "tickers",
        "date_from",
        "date_to",
        "liquidity_rank_min",
        "liquidity_rank_max",
        "digest_sections",
        "batch_ids",
        "research_modes",
        "data_provenances",
        "content_kinds",
        "report_stages",
    }
)


class AdvancedEvalValidationError(ValueError):
    """The release corpus or its labelled run does not satisfy the contract."""


class EvalEvidenceSpan(BaseModel):
    """Exact raw-parent span used as stable retrieval ground truth."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    id: str = Field(min_length=1, max_length=128)
    document_id: str = Field(min_length=1, max_length=256)
    parent_id: str = Field(min_length=1, max_length=256)
    quote: str = Field(min_length=1, max_length=12_000)
    start_offset: int = Field(ge=0)
    end_offset: int = Field(gt=0)
    quote_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    page: int | None = Field(default=None, ge=1)

    @model_validator(mode="after")
    def validate_exact_span(self) -> "EvalEvidenceSpan":
        if self.end_offset - self.start_offset != len(self.quote):
            raise ValueError("evidence offsets must use Unicode character offsets")
        digest = hashlib.sha256(self.quote.encode("utf-8")).hexdigest()
        if self.quote_sha256 != digest:
            raise ValueError("quote_sha256 does not match the exact evidence quote")
        return self


class EvalNumericFact(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    value: str = Field(min_length=1, max_length=128)
    unit: str = Field(min_length=1, max_length=64)
    period: str = Field(min_length=1, max_length=128)


class EvalExpectedClaim(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    id: str = Field(min_length=1, max_length=128)
    evidence_ids: list[str] = Field(min_length=1, max_length=20)
    numeric: EvalNumericFact | None = None


class EvalExpectedEntity(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    id: str = Field(min_length=1, max_length=256)
    canonical_key: str = Field(min_length=1, max_length=256)


class EvalExpectedRelation(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    id: str = Field(min_length=1, max_length=256)
    subject_entity_id: str = Field(min_length=1, max_length=256)
    predicate: str = Field(min_length=1, max_length=64)
    object_entity_id: str | None = Field(default=None, min_length=1, max_length=256)
    literal_object: str | None = Field(default=None, min_length=1, max_length=1000)
    evidence_ids: list[str] = Field(min_length=1, max_length=20)

    @model_validator(mode="after")
    def validate_relation(self) -> "EvalExpectedRelation":
        if self.predicate not in GRAPH_PREDICATES:
            raise ValueError("predicate is outside the graph allowlist")
        if (self.object_entity_id is None) == (self.literal_object is None):
            raise ValueError(
                "relation must have exactly one of object_entity_id/literal_object"
            )
        return self


class EvalScope(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    document_ids: list[str] = Field(min_length=1, max_length=1000)
    filters: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_scope(self) -> "EvalScope":
        if len(self.document_ids) != len(set(self.document_ids)):
            raise ValueError("scope document_ids must be unique")
        if any(not item.strip() for item in self.document_ids):
            raise ValueError("scope document_ids must not contain blanks")
        unknown = set(self.filters) - EVAL_FILTER_FIELDS
        if unknown:
            raise ValueError(f"unknown scope filters: {sorted(unknown)!r}")
        list_fields = EVAL_FILTER_FIELDS - {
            "date_from",
            "date_to",
            "liquidity_rank_min",
            "liquidity_rank_max",
        }
        for name in list_fields & set(self.filters):
            value = self.filters[name]
            if (
                not isinstance(value, list)
                or not value
                or any(not isinstance(item, str) or not item.strip() for item in value)
                or len(value) != len(set(value))
            ):
                raise ValueError(f"scope filter {name} must be unique non-blank strings")
        parsed_dates: dict[str, date] = {}
        for name in ("date_from", "date_to"):
            if name not in self.filters:
                continue
            value = self.filters[name]
            if not isinstance(value, str):
                raise ValueError(f"scope filter {name} must be an ISO date")
            try:
                parsed_dates[name] = date.fromisoformat(value)
            except ValueError as exc:
                raise ValueError(f"scope filter {name} must be an ISO date") from exc
        if (
            parsed_dates.get("date_from")
            and parsed_dates.get("date_to")
            and parsed_dates["date_from"] > parsed_dates["date_to"]
        ):
            raise ValueError("scope date_from must not be after date_to")
        for name in ("liquidity_rank_min", "liquidity_rank_max"):
            if name in self.filters:
                value = self.filters[name]
                if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                    raise ValueError(f"scope filter {name} must be a positive integer")
        minimum = self.filters.get("liquidity_rank_min")
        maximum = self.filters.get("liquidity_rank_max")
        if isinstance(minimum, int) and isinstance(maximum, int) and minimum > maximum:
            raise ValueError("scope liquidity rank range is invalid")
        return self


class AdvancedEvalCase(BaseModel):
    """Versioned human-reviewed input for the advanced offline release gate."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    schema_version: Literal[1]
    id: str = Field(min_length=1, max_length=128)
    question: str = Field(min_length=1, max_length=4000)
    intent: EvalIntent
    expected_route: EvalRoute
    expected_tools: list[EvalTool] = Field(min_length=1, max_length=3)
    answerable: bool
    scope: EvalScope
    evidence: list[EvalEvidenceSpan] = Field(default_factory=list, max_length=50)
    claims: list[EvalExpectedClaim] = Field(default_factory=list, max_length=50)
    entities: list[EvalExpectedEntity] = Field(default_factory=list, max_length=100)
    relations: list[EvalExpectedRelation] = Field(default_factory=list, max_length=50)
    simple_non_graph: bool = False

    @model_validator(mode="after")
    def validate_ground_truth_links(self) -> "AdvancedEvalCase":
        if len(self.expected_tools) != len(set(self.expected_tools)):
            raise ValueError("expected_tools must be unique")
        if self.expected_route == "mixed" and len(self.expected_tools) < 2:
            raise ValueError("mixed route requires at least two tools")
        if self.expected_route != "mixed" and self.expected_route not in self.expected_tools:
            raise ValueError("expected route must be represented in expected_tools")
        if self.simple_non_graph and "graph" in self.expected_tools:
            raise ValueError("simple_non_graph case cannot expect graph")

        def unique_ids(items: list[Any], label: str) -> set[str]:
            ids = [item.id for item in items]
            if len(ids) != len(set(ids)):
                raise ValueError(f"{label} ids must be unique")
            return set(ids)

        evidence_ids = unique_ids(self.evidence, "evidence")
        claim_ids = unique_ids(self.claims, "claim")
        entity_ids = unique_ids(self.entities, "entity")
        unique_ids(self.relations, "relation")
        if self.answerable and (not evidence_ids or not claim_ids):
            raise ValueError("answerable case requires evidence spans and claims")
        if not self.answerable and (evidence_ids or claim_ids or self.relations):
            raise ValueError("unanswerable case cannot declare supported claims/relations")

        scoped_documents = set(self.scope.document_ids)
        if any(item.document_id not in scoped_documents for item in self.evidence):
            raise ValueError("evidence document must stay inside the caller scope")
        for claim in self.claims:
            if not set(claim.evidence_ids).issubset(evidence_ids):
                raise ValueError("claim references an unknown evidence id")
        for relation in self.relations:
            if relation.subject_entity_id not in entity_ids:
                raise ValueError("relation subject references an unknown entity")
            if relation.object_entity_id and relation.object_entity_id not in entity_ids:
                raise ValueError("relation object references an unknown entity")
            if not set(relation.evidence_ids).issubset(evidence_ids):
                raise ValueError("relation references an unknown evidence id")
        return self


class EvalObservedClaim(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    claim_id: str = Field(min_length=1, max_length=128)
    support: Literal["supported", "unsupported", "contradicted", "missing_citation"]
    citation_evidence_ids: list[str] = Field(default_factory=list, max_length=20)
    numeric: EvalNumericFact | None = None


class EvalObservedCitation(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    citation_id: str = Field(min_length=1, max_length=128)
    document_id: str = Field(min_length=1, max_length=256)
    evidence_id: str | None = Field(default=None, min_length=1, max_length=128)
    supported: bool


class EvalObservedRelation(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    expected_relation_id: str | None = Field(default=None, min_length=1, max_length=256)
    correct: bool
    evidence_ids: list[str] = Field(default_factory=list, max_length=20)


class AdvancedEvalResult(BaseModel):
    """Labelled, privacy-safe trace emitted by an offline advanced eval run."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    schema_version: Literal[1]
    id: str = Field(min_length=1, max_length=128)
    route: EvalRoute | None = None
    tools: list[EvalTool] = Field(default_factory=list, max_length=3)
    candidate_evidence_ids: list[str] = Field(default_factory=list, max_length=40)
    context_evidence_ids: list[str] = Field(default_factory=list, max_length=12)
    claims: list[EvalObservedClaim] = Field(default_factory=list, max_length=100)
    citations: list[EvalObservedCitation] = Field(default_factory=list, max_length=100)
    abstained: bool
    retrieved_document_ids: list[str] = Field(default_factory=list, max_length=1000)
    scope_filter_violations: int = Field(default=0, ge=0)
    resolved_entity_ids: list[str] = Field(default_factory=list, max_length=100)
    graph_relations: list[EvalObservedRelation] = Field(default_factory=list, max_length=100)
    graph_evidence_ids: list[str] = Field(default_factory=list, max_length=20)
    latency_ms: float = Field(ge=0, allow_inf_nan=False)
    baseline_latency_ms: float | None = Field(
        default=None, gt=0, allow_inf_nan=False
    )
    timed_out: bool = False
    error_code: str | None = Field(default=None, min_length=1, max_length=128)

    @model_validator(mode="after")
    def validate_unique_trace_ids(self) -> "AdvancedEvalResult":
        for values, label in (
            (self.tools, "tools"),
            (self.candidate_evidence_ids, "candidate_evidence_ids"),
            (self.context_evidence_ids, "context_evidence_ids"),
            (self.retrieved_document_ids, "retrieved_document_ids"),
            (self.resolved_entity_ids, "resolved_entity_ids"),
            (self.graph_evidence_ids, "graph_evidence_ids"),
        ):
            if len(values) != len(set(values)):
                raise ValueError(f"{label} must be unique")
        claim_ids = [item.claim_id for item in self.claims]
        if len(claim_ids) != len(set(claim_ids)):
            raise ValueError("observed claim ids must be unique")
        citation_ids = [item.citation_id for item in self.citations]
        if len(citation_ids) != len(set(citation_ids)):
            raise ValueError("observed citation ids must be unique")
        return self


def is_abstention(answer: str, cited: list[dict[str, Any]]) -> bool:
    answer_lower = answer.lower()
    return not cited and any(marker in answer_lower for marker in ABSTENTION_MARKERS)


def percentile_95(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round(0.95 * len(ordered) + 0.5) - 1))
    return round(ordered[index], 2)


def page_range_matches(
    page: int, page_end: int | None, expected_pages: set[int]
) -> bool:
    """Một citation khoảng trang đúng khi chứa ít nhất một trang kỳ vọng."""

    end = page if page_end is None else page_end
    return not expected_pages or any(page <= expected <= end for expected in expected_pages)


def question_set_fingerprint(cases: list[dict[str, Any]]) -> str:
    """Hash ordered JSON records independent of JSONL whitespace/line endings."""

    canonical = "\n".join(
        json.dumps(
            case,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        for case in cases
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _read_jsonl_models(
    path: Path,
    model: type[AdvancedEvalCase] | type[AdvancedEvalResult],
) -> list[AdvancedEvalCase] | list[AdvancedEvalResult]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise AdvancedEvalValidationError(f"Không đọc được {path}: {exc}") from exc
    parsed: list[AdvancedEvalCase] | list[AdvancedEvalResult] = []
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            parsed.append(model.model_validate_json(line, strict=True))
        except ValidationError as exc:
            details = "; ".join(
                f"{'.'.join(str(part) for part in item['loc']) or '<record>'}: "
                f"{item['msg']}"
                for item in exc.errors(include_input=False, include_url=False)
            )
            raise AdvancedEvalValidationError(
                f"{path}: dòng {line_number} không đúng advanced eval schema: "
                f"{details}"
            ) from exc
    if not parsed:
        raise AdvancedEvalValidationError(f"{path}: JSONL không có record")
    return parsed


def load_advanced_eval_cases(path: Path) -> list[AdvancedEvalCase]:
    return list(_read_jsonl_models(path, AdvancedEvalCase))  # type: ignore[arg-type]


def load_advanced_eval_results(path: Path) -> list[AdvancedEvalResult]:
    return list(_read_jsonl_models(path, AdvancedEvalResult))  # type: ignore[arg-type]


def validate_advanced_release_corpus(
    cases: list[AdvancedEvalCase],
) -> dict[str, int]:
    """Validate release-only size and metric coverage without running a model."""

    if len(cases) < ADVANCED_RELEASE_MIN_CASES:
        raise AdvancedEvalValidationError(
            "Bộ advanced release gate phải có ít nhất "
            f"{ADVANCED_RELEASE_MIN_CASES} câu; hiện có {len(cases)}"
        )
    ids = [case.id for case in cases]
    if len(ids) != len(set(ids)):
        raise AdvancedEvalValidationError("Advanced eval case ids phải duy nhất")
    coverage = {
        "answerable": sum(case.answerable for case in cases),
        "unanswerable": sum(not case.answerable for case in cases),
        "numeric": sum(
            any(claim.numeric is not None for claim in case.claims) for case in cases
        ),
        "graph_entity": sum(bool(case.entities) for case in cases),
        "graph_relation": sum(bool(case.relations) for case in cases),
        "simple_non_graph": sum(case.simple_non_graph for case in cases),
    }
    missing = [name for name, count in coverage.items() if not count]
    if missing:
        raise AdvancedEvalValidationError(
            f"Advanced eval thiếu coverage bắt buộc: {', '.join(missing)}"
        )
    return coverage


def _ratio(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 4) if denominator else 0.0


def _mean(values: list[float]) -> float:
    return round(sum(values) / len(values), 4) if values else 0.0


def _load_legacy_eval_result(path: Path, *, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AdvancedEvalValidationError(
            f"Không đọc được legacy {label} {path}: {exc}"
        ) from exc
    if not isinstance(payload, dict):
        raise AdvancedEvalValidationError(f"Legacy {label} phải là JSON object")
    cases = payload.get("cases")
    metrics = payload.get("metrics", payload)
    question_set_sha256 = payload.get("question_set_sha256")
    if not isinstance(cases, int) or cases < 30 or not isinstance(metrics, dict):
        raise AdvancedEvalValidationError(
            f"Legacy {label} phải có ít nhất 30 cases và metrics"
        )
    if (
        not isinstance(question_set_sha256, str)
        or len(question_set_sha256) != 64
        or any(
            character not in "0123456789abcdef"
            for character in question_set_sha256
        )
    ):
        raise AdvancedEvalValidationError(
            f"Legacy {label} thiếu question_set_sha256 hợp lệ; hãy chạy lại "
            "evaluator hiện tại trên đúng evals/questions.jsonl"
        )
    missing = [key for key in LEGACY_REGRESSION_DIRECTIONS if key not in metrics]
    if missing:
        raise AdvancedEvalValidationError(
            f"Legacy {label} thiếu metrics: {', '.join(missing)}"
        )
    selected: dict[str, float | int] = {}
    for name in LEGACY_REGRESSION_DIRECTIONS:
        value = metrics[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise AdvancedEvalValidationError(
                f"Legacy {label} metric {name} phải là số"
            )
        if not math.isfinite(float(value)):
            raise AdvancedEvalValidationError(
                f"Legacy {label} metric {name} phải là số hữu hạn"
            )
        selected[name] = value
    failed_case_ids = payload.get("failed_case_ids")
    if failed_case_ids is None:
        failed_cases = payload.get("failed_cases")
        if not isinstance(failed_cases, list) or any(
            not isinstance(item, dict) for item in failed_cases
        ):
            raise AdvancedEvalValidationError(
                f"Legacy {label} phải có failed_cases/failed_case_ids hợp lệ"
            )
        failed_case_ids = [item.get("id") for item in failed_cases]
    if (
        not isinstance(failed_case_ids, list)
        or any(not isinstance(item, str) or not item for item in failed_case_ids)
        or len(failed_case_ids) != len(set(failed_case_ids))
    ):
        raise AdvancedEvalValidationError(
            f"Legacy {label} phải có failed_cases/failed_case_ids hợp lệ"
        )
    return {
        "cases": cases,
        "question_set_sha256": question_set_sha256,
        "metrics": selected,
        "failed_case_ids": failed_case_ids,
    }


def _legacy_regression(current_path: Path, baseline_path: Path) -> dict[str, Any]:
    current = _load_legacy_eval_result(current_path, label="current result")
    baseline = _load_legacy_eval_result(baseline_path, label="baseline")
    if current["cases"] != baseline["cases"]:
        raise AdvancedEvalValidationError(
            "Legacy current result và baseline phải có cùng số cases "
            f"({current['cases']} != {baseline['cases']})"
        )
    if current["question_set_sha256"] != baseline["question_set_sha256"]:
        raise AdvancedEvalValidationError(
            "Legacy current result và baseline không cùng questions.jsonl hash"
        )

    regressions: list[str] = []
    comparison: dict[str, dict[str, Any]] = {}
    for name, direction in LEGACY_REGRESSION_DIRECTIONS.items():
        current_value = float(current["metrics"][name])
        baseline_value = float(baseline["metrics"][name])
        passed = (
            current_value >= baseline_value
            if direction == "higher"
            else current_value <= baseline_value
        )
        if not passed:
            regressions.append(name)
        comparison[name] = {
            "direction": direction,
            "baseline": baseline["metrics"][name],
            "current": current["metrics"][name],
            "delta": round(current_value - baseline_value, 4),
            "passed": passed,
        }
    new_failed_case_ids = sorted(
        set(current["failed_case_ids"]) - set(baseline["failed_case_ids"])
    )
    if new_failed_case_ids:
        regressions.append("new_failed_cases")
    return {
        "cases": current["cases"],
        "question_set_sha256": current["question_set_sha256"],
        "baseline_path": str(baseline_path),
        "comparison": comparison,
        "new_failed_case_ids": new_failed_case_ids,
        "regressions": regressions,
        "passed": not regressions,
    }


def score_advanced_release_gate(
    cases: list[AdvancedEvalCase],
    results: list[AdvancedEvalResult],
    *,
    legacy_result_path: Path,
    legacy_baseline_path: Path,
) -> dict[str, Any]:
    """Score only labelled traces; this never calls a cloud service or model judge."""

    coverage = validate_advanced_release_corpus(cases)
    result_ids = [item.id for item in results]
    if len(result_ids) != len(set(result_ids)):
        raise AdvancedEvalValidationError("Advanced eval result ids phải duy nhất")
    case_by_id = {case.id: case for case in cases}
    result_by_id = {item.id: item for item in results}
    missing = sorted(set(case_by_id) - set(result_by_id))
    unexpected = sorted(set(result_by_id) - set(case_by_id))
    if missing or unexpected:
        raise AdvancedEvalValidationError(
            "Case/result id không khớp "
            f"(missing={missing[:10]!r}, unexpected={unexpected[:10]!r})"
        )

    candidate_recalls: list[float] = []
    context_recalls: list[float] = []
    supported_claims = 0
    evaluated_claims = 0
    unsupported_citations = 0
    numeric_exact = 0
    numeric_total = 0
    abstention_correct = 0
    route_correct = 0
    scope_escapes = 0
    expected_entities = 0
    resolved_entities = 0
    observed_relations = 0
    correct_relations = 0
    graph_evidence_recalls: list[float] = []
    simple_latencies: list[float] = []
    baseline_latencies: list[float] = []
    timeout_failures = 0
    request_errors = 0

    for case in cases:
        result = result_by_id[case.id]
        expected_evidence = {item.id for item in case.evidence}
        if case.answerable:
            candidate_recalls.append(
                _ratio(
                    len(expected_evidence & set(result.candidate_evidence_ids[:20])),
                    len(expected_evidence),
                )
            )
            context_recalls.append(
                _ratio(
                    len(expected_evidence & set(result.context_evidence_ids[:5])),
                    len(expected_evidence),
                )
            )

        expected_claims = {claim.id: claim for claim in case.claims}
        observed_claims = {claim.claim_id: claim for claim in result.claims}
        for claim_id in sorted(set(expected_claims) | set(observed_claims)):
            evaluated_claims += 1
            observed = observed_claims.get(claim_id)
            expected = expected_claims.get(claim_id)
            citation_matches = bool(observed and observed.citation_evidence_ids)
            if expected and observed:
                citation_matches = bool(
                    set(observed.citation_evidence_ids) & set(expected.evidence_ids)
                )
            if observed and observed.support == "supported" and citation_matches:
                supported_claims += 1
        for expected in expected_claims.values():
            if expected.numeric is None:
                continue
            numeric_total += 1
            observed = observed_claims.get(expected.id)
            if (
                observed
                and observed.support == "supported"
                and observed.numeric == expected.numeric
            ):
                numeric_exact += 1

        unsupported_citations += sum(not item.supported for item in result.citations)
        abstention_correct += result.abstained == (not case.answerable)
        route_correct += (
            result.route == case.expected_route
            and set(result.tools) == set(case.expected_tools)
        )
        scoped_documents = set(case.scope.document_ids)
        observed_documents = set(result.retrieved_document_ids) | {
            citation.document_id for citation in result.citations
        }
        scope_escapes += (
            len(observed_documents - scoped_documents)
            + result.scope_filter_violations
        )

        expected_entity_ids = {item.id for item in case.entities}
        expected_entities += len(expected_entity_ids)
        resolved_entities += len(expected_entity_ids & set(result.resolved_entity_ids))

        relation_ids = {item.id for item in case.relations}
        observed_relations += len(result.graph_relations)
        correct_relations += sum(
            relation.correct and relation.expected_relation_id in relation_ids
            for relation in result.graph_relations
        )
        relation_evidence = {
            evidence_id
            for relation in case.relations
            for evidence_id in relation.evidence_ids
        }
        if relation_evidence:
            graph_evidence_recalls.append(
                _ratio(
                    len(
                        relation_evidence
                        & set(result.graph_evidence_ids[:10])
                    ),
                    len(relation_evidence),
                )
            )

        if case.simple_non_graph:
            if result.baseline_latency_ms is None:
                raise AdvancedEvalValidationError(
                    f"Result {case.id!r} thiếu baseline_latency_ms"
                )
            simple_latencies.append(result.latency_ms)
            baseline_latencies.append(result.baseline_latency_ms)
        if result.timed_out or result.latency_ms > ADVANCED_RELEASE_TIMEOUT_MS:
            timeout_failures += 1
        if result.error_code is not None:
            request_errors += 1

    simple_p95 = percentile_95(simple_latencies)
    baseline_p95 = percentile_95(baseline_latencies)
    latency_ratio = round(simple_p95 / baseline_p95, 4) if baseline_p95 else 0.0
    metrics: dict[str, float | int] = {
        "candidate_recall_at_20": _mean(candidate_recalls),
        "final_context_recall_at_5": _mean(context_recalls),
        "citation_claim_support": _ratio(supported_claims, evaluated_claims),
        "unsupported_citations": unsupported_citations,
        "numeric_exactness": _ratio(numeric_exact, numeric_total),
        "abstention_accuracy": _ratio(abstention_correct, len(cases)),
        "route_accuracy": _ratio(route_correct, len(cases)),
        "scope_escapes": scope_escapes,
        "graph_entity_resolution": _ratio(resolved_entities, expected_entities),
        "graph_relation_precision": _ratio(correct_relations, observed_relations),
        "graph_evidence_recall_at_10": _mean(graph_evidence_recalls),
        "simple_non_graph_p95_ms": simple_p95,
        "simple_non_graph_baseline_p95_ms": baseline_p95,
        "simple_non_graph_p95_ratio": latency_ratio,
        "advanced_timeout_failures": timeout_failures,
        "advanced_request_errors": request_errors,
    }
    legacy = _legacy_regression(legacy_result_path, legacy_baseline_path)

    failed_gates: list[str] = []
    minimum_metrics = {
        "candidate_recall_at_20",
        "final_context_recall_at_5",
        "citation_claim_support",
        "numeric_exactness",
        "abstention_accuracy",
        "route_accuracy",
        "graph_entity_resolution",
        "graph_relation_precision",
        "graph_evidence_recall_at_10",
    }
    for name, threshold in ADVANCED_RELEASE_THRESHOLDS.items():
        actual = metrics[name]
        if name in minimum_metrics:
            if float(actual) < float(threshold):
                failed_gates.append(name)
        elif name == "simple_non_graph_p95_ratio":
            if float(actual) > float(threshold):
                failed_gates.append(name)
        elif int(actual) != int(threshold):
            failed_gates.append(name)
    if request_errors:
        failed_gates.append("advanced_request_errors")
    if not legacy["passed"]:
        failed_gates.append("legacy_30_no_regression")

    return {
        "schema_version": ADVANCED_EVAL_SCHEMA_VERSION,
        "mode": "advanced_release_gate",
        "cases": len(cases),
        "coverage": coverage,
        "metrics": metrics,
        "acceptance": dict(ADVANCED_RELEASE_THRESHOLDS),
        "legacy_regression": legacy,
        "failed_gates": failed_gates,
        "passed": not failed_gates,
    }


def evaluate_advanced_release_gate(
    cases_path: Path,
    *,
    results_path: Path | None,
    legacy_result_path: Path | None,
    legacy_baseline_path: Path | None,
    validate_only: bool = False,
) -> int:
    cases = load_advanced_eval_cases(cases_path)
    coverage = validate_advanced_release_corpus(cases)
    if validate_only:
        print(
            json.dumps(
                {
                    "schema_version": ADVANCED_EVAL_SCHEMA_VERSION,
                    "mode": "advanced_release_gate_validation",
                    "cases": len(cases),
                    "coverage": coverage,
                    "valid": True,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    if (
        results_path is None
        or legacy_result_path is None
        or legacy_baseline_path is None
    ):
        raise AdvancedEvalValidationError(
            "Release gate cần --results, --legacy-results và --legacy-baseline; "
            "dùng --validate-only nếu chỉ kiểm tra golden JSONL"
        )
    results = load_advanced_eval_results(results_path)
    report = score_advanced_release_gate(
        cases,
        results,
        legacy_result_path=legacy_result_path,
        legacy_baseline_path=legacy_baseline_path,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["passed"] else 2


async def evaluate_file(path: Path) -> int:
    raw_questions = path.read_bytes()
    cases = [
        json.loads(line)
        for line in raw_questions.decode("utf-8").splitlines()
        if line.strip()
    ]
    question_set_sha256 = question_set_fingerprint(cases)
    if len(cases) < 20:
        raise ValueError("Bộ eval phải có ít nhất 20 câu")

    runtime = Runtime()
    await runtime.start()
    try:
        while any(
            item["status"] in {"pending", "indexing"}
            for item in runtime.storage.list_documents()
        ):
            await asyncio.sleep(0.25)

        documents = {
            item["filename"]: item["id"]
            for item in runtime.storage.list_ready_documents()
        }
        all_document_ids = list(documents.values())
        dense_retriever = ParentChildRetriever(
            replace(runtime.settings, hybrid_search=False),
            runtime.storage,
            runtime.index,
        )
        hybrid_retriever = ParentChildRetriever(
            replace(runtime.settings, hybrid_search=True),
            runtime.storage,
            runtime.index,
        )
        sessions: dict[str, str] = {}
        totals = defaultdict(int)
        exact_totals = defaultdict(int)
        diagnostics: list[dict[str, Any]] = []
        dense_latencies: list[float] = []
        hybrid_latencies: list[float] = []

        for case in cases:
            expected_document = case.get("document")
            if expected_document and expected_document not in documents:
                raise ValueError(f"Thiếu tài liệu eval: {expected_document}")
            conversation = case.get("conversation")
            if case.get("kind") == "exact_keyword":
                exact_totals["cases"] += 1
                # Đảo thứ tự để warm-up không thiên vị dense hoặc hybrid.
                retrievers = [
                    ("hybrid", hybrid_retriever, hybrid_latencies),
                    ("dense", dense_retriever, dense_latencies),
                ]
                if exact_totals["cases"] % 2 == 0:
                    retrievers.reverse()
                retrieval_outputs: dict[str, list[Any]] = {}
                for label, retriever, latencies in retrievers:
                    started = time.perf_counter()
                    retrieval_outputs[label] = await asyncio.to_thread(
                        retriever.retrieve, case["question"], all_document_ids
                    )
                    latencies.append((time.perf_counter() - started) * 1000)
                expected_pages_for_probe = set(case.get("expected_pages", []))
                for label, sources in retrieval_outputs.items():
                    if any(
                        source.filename == expected_document
                        and page_range_matches(
                            source.page,
                            source.page_end,
                            expected_pages_for_probe,
                        )
                        for source in sources
                    ):
                        exact_totals[f"{label}_hit"] += 1
            response = await runtime.chat.answer(
                case["question"],
                sessions.get(conversation) if conversation else None,
                # Tìm trên toàn corpus để đo cả khả năng tránh tài liệu không liên quan.
                None,
            )
            if conversation:
                sessions[conversation] = response["session_id"]
            expected_pages = set(case.get("expected_pages", []))
            retrieved = response["citations"]
            cited = [item for item in retrieved if item["cited"]]

            if case["kind"] == "unanswerable":
                totals["unanswerable"] += 1
                correct = is_abstention(response["answer"], cited)
                if correct:
                    totals["correct_abstention"] += 1
                if not correct:
                    diagnostics.append(
                        {"id": case["id"], "reason": "incorrect_abstention"}
                    )
                continue

            totals["answerable"] += 1
            retrieval_hit = any(
                item["filename"] == expected_document
                and page_range_matches(
                    int(item["page"]), item.get("page_end"), expected_pages
                )
                for item in retrieved
            )
            if retrieval_hit:
                totals["retrieval_hit"] += 1
            if cited:
                totals["answers_with_citation"] += 1
            invalid_citations: list[dict[str, Any]] = []
            for item in cited:
                totals["cited_sources"] += 1
                if item["filename"] == expected_document and (
                    page_range_matches(
                        int(item["page"]), item.get("page_end"), expected_pages
                    )
                ):
                    totals["correct_citations"] += 1
                else:
                    totals["irrelevant_citations"] += 1
                    invalid_citations.append(
                        {
                            "filename": item["filename"],
                            "page": item["page"],
                            "page_end": item.get("page_end", item["page"]),
                        }
                    )
            if not retrieval_hit or invalid_citations:
                diagnostics.append(
                    {
                        "id": case["id"],
                        "reason": "retrieval_miss" if not retrieval_hit else "invalid_citation",
                        "expected_document": expected_document,
                        "expected_pages": sorted(expected_pages),
                        "invalid_citations": invalid_citations,
                    }
                )

        def ratio(numerator: str, denominator: str) -> float:
            return (
                round(totals[numerator] / totals[denominator], 4)
                if totals[denominator]
                else 0.0
            )

        dense_exact_recall = (
            round(exact_totals["dense_hit"] / exact_totals["cases"], 4)
            if exact_totals["cases"]
            else 0.0
        )
        hybrid_exact_recall = (
            round(exact_totals["hybrid_hit"] / exact_totals["cases"], 4)
            if exact_totals["cases"]
            else 0.0
        )
        dense_p95 = percentile_95(dense_latencies)
        hybrid_p95 = percentile_95(hybrid_latencies)
        latency_ratio = round(hybrid_p95 / dense_p95, 4) if dense_p95 else 0.0
        exact_quality_ok = hybrid_exact_recall >= dense_exact_recall and (
            dense_exact_recall >= 1.0
            or hybrid_exact_recall - dense_exact_recall >= 0.05
        )

        metrics = {
            "cases": len(cases),
            "question_set_sha256": question_set_sha256,
            "parent_recall_at_5": ratio("retrieval_hit", "answerable"),
            "citation_correct_page": ratio("correct_citations", "cited_sources"),
            "abstention_accuracy": ratio("correct_abstention", "unanswerable"),
            "answers_with_citation": ratio("answers_with_citation", "answerable"),
            "irrelevant_citations": totals["irrelevant_citations"],
            "hybrid_comparison": {
                "chat_mode": (
                    "hybrid" if runtime.settings.hybrid_search else "dense"
                ),
                "exact_keyword_cases": exact_totals["cases"],
                "dense_parent_recall_at_5": dense_exact_recall,
                "hybrid_parent_recall_at_5": hybrid_exact_recall,
                "dense_retrieval_p95_ms": dense_p95,
                "hybrid_retrieval_p95_ms": hybrid_p95,
                "latency_ratio": latency_ratio,
                "quality_gate_passed": exact_quality_ok,
                "latency_gate_passed": latency_ratio <= 1.20,
            },
            "acceptance": {
                "parent_recall_at_5": 0.85,
                "citation_correct_page": 0.90,
                "abstention_accuracy": 0.90,
                "irrelevant_citations": 0,
            },
            "failed_cases": diagnostics,
        }
        passed = (
            metrics["parent_recall_at_5"] >= 0.85
            and metrics["citation_correct_page"] >= 0.90
            and metrics["abstention_accuracy"] >= 0.90
            and metrics["irrelevant_citations"] == 0
        )
        metrics["hybrid_comparison"]["recommended_default"] = bool(
            passed
            and runtime.settings.hybrid_search
            and metrics["hybrid_comparison"]["quality_gate_passed"]
            and metrics["hybrid_comparison"]["latency_gate_passed"]
        )
        print(json.dumps(metrics, ensure_ascii=False, indent=2))
        return 0 if passed else 2
    finally:
        await runtime.stop()
