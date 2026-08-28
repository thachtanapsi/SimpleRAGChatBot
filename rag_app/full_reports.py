"""Strict, privacy-safe Full Analysis promotion contract."""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal, InvalidOperation
from typing import Any, Mapping

from .digests import (
    DIGEST_MAX_BYTES,
    SHA256_RE,
    TICKER_RE,
    _normalise_date,
    _normalise_fingerprint,
    _sha256_json,
    digest_document_sha256,
    normalise_cutoff,
    normalise_provenance_pair,
    normalise_source_updated_at,
)
from .errors import ValidationError


FULL_REPORT_V1 = "full_report_v1"
FULL_PIPELINE_VERSION = "gx_full_v1"
DECISION_SCHEMA_VERSION = 1
DECISION_PROJECTION_VERSION = 2
DECISION_STRUCTURED_SECTION = "decision.structured"
DECISION_MAX_BYTES = 64 * 1024
FULL_SECTION_MAX_BYTES = 32 * 1024
FULL_STAGE_KEYS = (
    "market",
    "sentiment",
    "news",
    "fundamentals",
    "research",
    "trader",
    "risk",
)
FULL_CORE_STAGES = ("research", "trader", "risk")
FULL_STAGE_STATUSES = {"complete", "partial", "unavailable"}
FULL_SECTION_ORDER = (
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
FULL_SECTION_STAGE = {
    "analyst.market": "market",
    "analyst.sentiment": "sentiment",
    "analyst.news": "news",
    "analyst.fundamentals": "fundamentals",
    "research.bull": "research",
    "research.bear": "research",
    "research.manager": "research",
    "trading.plan": "trader",
    "risk.aggressive": "risk",
    "risk.conservative": "risk",
    "risk.neutral": "risk",
    "portfolio.decision": "risk",
}
FULL_SECTION_LABELS = {
    "analyst.market": "Phân tích thị trường",
    "analyst.sentiment": "Phân tích tâm lý",
    "analyst.news": "Phân tích tin tức",
    "analyst.fundamentals": "Phân tích cơ bản",
    "research.bull": "Luận điểm tích cực",
    "research.bear": "Luận điểm thận trọng",
    "research.manager": "Kết luận nghiên cứu",
    "trading.plan": "Kế hoạch giao dịch",
    "risk.aggressive": "Góc nhìn rủi ro tích cực",
    "risk.conservative": "Góc nhìn rủi ro thận trọng",
    "risk.neutral": "Góc nhìn rủi ro trung lập",
    "portfolio.decision": "Quyết định danh mục",
    DECISION_STRUCTURED_SECTION: "Quyết định có cấu trúc",
}
FULL_SECTION_CITATION_LABELS = {
    section: section.replace(".", " ").capitalize()
    for section in FULL_SECTION_ORDER
}
FULL_REPORT_KEYS = {
    "schema_version",
    "content_kind",
    "report_schema",
    "source",
    "ticker",
    "analysis_date",
    "cutoff",
    "source_updated_at",
    "liquidity_rank",
    "research_mode",
    "data_provenance",
    "identity_hash",
    "parent_identity_hash",
    "expected_digest_hash",
    "full_report_hash",
    "pipeline_version",
    "prompt_fingerprint",
    "model_fingerprint",
    "evidence_fingerprint",
    "execution_generation",
    "contains_decision_content",
    "status",
    "stage_status",
    "stage_fingerprints",
    "sections",
}
FULL_REPORT_FORBIDDEN_TEXT_RE = re.compile(
    r"(?:[a-z][a-z0-9+.-]{1,15}://|www\.|raw[\s_-]*evidence|"
    r"(?:authorization|api[\s_-]*key|credential|secret)\s*[:=]\s*\S+)",
    re.IGNORECASE,
)

PORTFOLIO_DECISION_LABELS = (
    "Rating",
    "Executive Summary",
    "Investment Thesis",
    "Time Horizon",
    "Price Target Status",
    "Price Target",
    "Price Target Currency",
    "Price Target Rationale",
    "Price Target Unavailable Reason",
)
TRADING_DECISION_LABELS = (
    "Action",
    "Reasoning",
    "Entry Price",
    "Stop Loss",
    "Position Sizing",
)
PORTFOLIO_RATINGS = {
    value.casefold(): value
    for value in ("Buy", "Overweight", "Hold", "Underweight", "Sell")
}
TRADING_ACTIONS = {
    value.casefold(): value for value in ("Buy", "Hold", "Sell")
}
NULLISH_VALUES = {
    "",
    "unavailable",
    "not available",
    "n/a",
    "na",
    "none",
    "null",
    "nil",
    "unknown",
}
REFERENCE_CURRENCY_RE = re.compile(r"^[A-Z]{2,12}$")
FINAL_PROPOSAL_RE = re.compile(
    r"(?im)^\s*FINAL\s+TRANSACTION\s+PROPOSAL\s*:.*$"
)


@dataclass(frozen=True, slots=True)
class CanonicalFullReport:
    document_id: str
    source_key: str
    source: str
    ticker: str
    analysis_date: str
    cutoff: str
    source_updated_at: str
    source_run_id: None
    liquidity_rank: int
    prompt_fingerprint: str
    model_fingerprint: str
    evidence_fingerprint: str
    digest_status: str
    research_mode: str
    data_provenance: str
    identity_hash: str
    digest_hash: None
    parent_identity_hash: str
    expected_digest_hash: str
    full_report_hash: str
    content_kind: str
    report_schema: str
    pipeline_version: str
    execution_generation: int
    contains_decision_content: bool
    stage_status: dict[str, str]
    stage_fingerprints: dict[str, str]
    sections: dict[str, str]
    section_statuses: dict[str, str]
    section_stages: dict[str, str]
    canonical_report: dict[str, Any]
    canonical_json: str
    canonical_sha256: str
    decision: dict[str, Any]
    decision_json: str
    decision_sha256: str
    decision_status: str
    projection_version: int
    content_sha256: str
    sha256: str
    filename: str

    @property
    def size_bytes(self) -> int:
        return len(self.canonical_json.encode("utf-8"))


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _markdown_fields(text: str, labels: tuple[str, ...]) -> dict[str, str]:
    """Extract rendered Markdown fields without collapsing paragraph breaks."""

    alternatives = "|".join(
        re.escape(label) for label in sorted(labels, key=len, reverse=True)
    )
    pattern = re.compile(
        rf"(?im)^[ \t]*(?:[-+]\s+)?(?:\*\*)?"
        rf"(?P<label>{alternatives})(?:\*\*)?[ \t]*:[ \t]*"
        rf"(?P<value>[^\r\n]*)"
    )
    matches = list(pattern.finditer(text))
    fields: dict[str, str] = {}
    canonical_labels = {label.casefold(): label for label in labels}
    for index, match in enumerate(matches):
        label = canonical_labels[match.group("label").casefold()]
        if label in fields:
            continue
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        first_line = match.group("value")
        continuation = text[match.end() : end]
        value = f"{first_line}{continuation}".strip()
        final_proposal = FINAL_PROPOSAL_RE.search(value)
        if final_proposal:
            value = value[: final_proposal.start()].rstrip()
        fields[label] = value
    return fields


def _optional_text(value: str | None) -> str | None:
    if value is None:
        return None
    normalised = value.strip()
    if normalised.casefold().rstrip(".:-") in NULLISH_VALUES:
        return None
    return normalised or None


def _positive_number(value: str | None) -> int | float | None:
    text = _optional_text(value)
    if text is None or not re.fullmatch(
        r"[+]?(?:\d+|\d{1,3}(?:,\d{3})+)(?:\.\d+)?", text
    ):
        return None
    try:
        number = Decimal(text.replace(",", ""))
    except InvalidOperation:
        return None
    if not number.is_finite() or number <= 0:
        return None
    if number == number.to_integral_value():
        return int(number)
    try:
        floating = float(number)
    except (OverflowError, ValueError):
        return None
    return floating if math.isfinite(floating) else None


def _derive_decision(
    portfolio_markdown: str, trading_markdown: str
) -> tuple[dict[str, Any], str, str, str]:
    portfolio_fields = _markdown_fields(
        portfolio_markdown, PORTFOLIO_DECISION_LABELS
    )
    trading_fields = _markdown_fields(trading_markdown, TRADING_DECISION_LABELS)
    warnings: list[str] = []

    raw_rating = _optional_text(portfolio_fields.get("Rating"))
    rating = PORTFOLIO_RATINGS.get(raw_rating.casefold()) if raw_rating else None
    if rating is None:
        warnings.append("portfolio_rating_missing")
    executive_summary = _optional_text(portfolio_fields.get("Executive Summary"))
    if executive_summary is None:
        warnings.append("portfolio_executive_summary_missing")
    investment_thesis = _optional_text(portfolio_fields.get("Investment Thesis"))
    if investment_thesis is None:
        warnings.append("portfolio_investment_thesis_missing")
    time_horizon = _optional_text(portfolio_fields.get("Time Horizon"))

    raw_target_status = _optional_text(portfolio_fields.get("Price Target Status"))
    raw_price_target = portfolio_fields.get("Price Target")
    raw_price_target_currency = portfolio_fields.get("Price Target Currency")
    price_target = _positive_number(raw_price_target)
    price_target_currency = _optional_text(raw_price_target_currency)
    if price_target_currency is not None:
        price_target_currency = price_target_currency.upper()
    price_target_rationale = _optional_text(
        portfolio_fields.get("Price Target Rationale")
    )
    price_target_unavailable_reason = _optional_text(
        portfolio_fields.get("Price Target Unavailable Reason")
    )
    if raw_target_status:
        status_key = raw_target_status.casefold()
        price_target_status = (
            status_key if status_key in {"available", "unavailable"} else "unknown"
        )
    elif price_target is not None:
        price_target_status = "available"
    elif price_target_unavailable_reason is not None:
        price_target_status = "unavailable"
    else:
        price_target_status = "unknown"

    target_contract_valid = False
    if price_target_status == "available":
        target_contract_valid = (
            price_target is not None
            and price_target_currency is not None
            and price_target_rationale is not None
            and price_target_unavailable_reason is None
            and not (
                price_target_currency.casefold() == "vnd"
                and not isinstance(price_target, int)
            )
        )
    elif price_target_status == "unavailable":
        target_contract_valid = (
            price_target is None
            and _optional_text(raw_price_target) is None
            and (
                price_target_currency is None
                or REFERENCE_CURRENCY_RE.fullmatch(price_target_currency) is not None
            )
            and price_target_rationale is None
            and price_target_unavailable_reason is not None
        )
        if target_contract_valid:
            # A producer may retain the instrument's reference currency in raw
            # Markdown even when no target exists. The typed decision remains
            # mutually exclusive and therefore projects that metadata to null.
            price_target_currency = None
    if not target_contract_valid:
        warnings.append("portfolio_price_target_contract_invalid")
        price_target_status = "unknown"
        price_target = None
        price_target_currency = None
        price_target_rationale = None
        price_target_unavailable_reason = None

    raw_action = _optional_text(trading_fields.get("Action"))
    action = TRADING_ACTIONS.get(raw_action.casefold()) if raw_action else None
    if action is None:
        warnings.append("trading_action_missing")
    reasoning = _optional_text(trading_fields.get("Reasoning"))
    if reasoning is None:
        warnings.append("trading_reasoning_missing")
    raw_entry_price = trading_fields.get("Entry Price")
    entry_price = _positive_number(raw_entry_price)
    if _optional_text(raw_entry_price) is not None and entry_price is None:
        warnings.append("trading_entry_price_invalid")
    raw_stop_loss = trading_fields.get("Stop Loss")
    stop_loss = _positive_number(raw_stop_loss)
    if _optional_text(raw_stop_loss) is not None and stop_loss is None:
        warnings.append("trading_stop_loss_invalid")
    position_sizing = _optional_text(trading_fields.get("Position Sizing"))

    decision_status = "complete" if not warnings else "partial"
    decision = {
        "schema_version": DECISION_SCHEMA_VERSION,
        "status": decision_status,
        "warnings": warnings,
        "portfolio": {
            "rating": rating,
            "executive_summary": executive_summary,
            "investment_thesis": investment_thesis,
            "time_horizon": time_horizon,
            "price_target_status": price_target_status,
            "price_target": price_target,
            "price_target_currency": price_target_currency,
            "price_target_rationale": price_target_rationale,
            "price_target_unavailable_reason": price_target_unavailable_reason,
        },
        "trading": {
            "action": action,
            "reasoning": reasoning,
            "entry_price": entry_price,
            "stop_loss": stop_loss,
            "position_sizing": position_sizing,
        },
    }
    decision_json = _canonical_json(decision)
    if len(decision_json.encode("utf-8")) > DECISION_MAX_BYTES:
        raise ValidationError(f"Decision projection vượt quá {DECISION_MAX_BYTES} bytes")
    decision_sha256 = _sha256_json(decision)
    return decision, decision_json, decision_sha256, decision_status


def canonicalise_full_report(payload: Mapping[str, Any]) -> CanonicalFullReport:
    """Canonicalise only the frozen v1 Full Analysis allowlist."""

    if not isinstance(payload, Mapping) or set(payload) != FULL_REPORT_KEYS:
        raise ValidationError("Full Analysis phải tuân thủ đúng contract v2")
    if payload.get("schema_version") != 2 or isinstance(
        payload.get("schema_version"), bool
    ):
        raise ValidationError("schema_version của Full Analysis phải là 2")
    if payload.get("content_kind") != FULL_REPORT_V1:
        raise ValidationError("content_kind không hợp lệ")
    if payload.get("report_schema") != FULL_REPORT_V1:
        raise ValidationError("report_schema không hợp lệ")
    if payload.get("pipeline_version") != FULL_PIPELINE_VERSION:
        raise ValidationError("pipeline_version không hợp lệ")
    if payload.get("contains_decision_content") is not True:
        raise ValidationError("contains_decision_content phải là true")
    source_value = payload.get("source")
    if not isinstance(source_value, str):
        raise ValidationError("source của Full Analysis là bắt buộc")
    source = source_value.strip()
    if not source or len(source) > 128:
        raise ValidationError("source của Full Analysis không hợp lệ")
    ticker_value = payload.get("ticker")
    if not isinstance(ticker_value, str):
        raise ValidationError("ticker không hợp lệ")
    ticker = ticker_value.strip().upper()
    if not TICKER_RE.fullmatch(ticker):
        raise ValidationError("ticker không hợp lệ")
    try:
        analysis_date = _normalise_date(payload.get("analysis_date"))
        cutoff = normalise_cutoff(payload.get("cutoff"))
        source_updated_at = normalise_source_updated_at(
            payload.get("source_updated_at")
        )
    except TypeError as exc:
        raise ValidationError("Ngày giờ của Full Analysis không hợp lệ") from exc
    if datetime.fromisoformat(cutoff).date().isoformat() != analysis_date:
        raise ValidationError("cutoff phải thuộc analysis_date theo Asia/Ho_Chi_Minh")
    rank_value = payload.get("liquidity_rank")
    if isinstance(rank_value, bool) or not isinstance(rank_value, int):
        raise ValidationError("liquidity_rank phải là số nguyên")
    rank = rank_value
    if rank <= 0:
        raise ValidationError("liquidity_rank phải lớn hơn 0")
    research_mode, data_provenance = normalise_provenance_pair(
        payload.get("research_mode"),
        payload.get("data_provenance"),
        allow_legacy=False,
    )
    raw_stage_status = payload.get("stage_status")
    if not isinstance(raw_stage_status, Mapping):
        raise ValidationError("stage_status phải có đúng 7 stage")
    stage_status = dict(raw_stage_status)
    if set(stage_status) != set(FULL_STAGE_KEYS):
        raise ValidationError("stage_status phải có đúng 7 stage")
    if any(value not in FULL_STAGE_STATUSES for value in stage_status.values()):
        raise ValidationError("stage_status không hợp lệ")
    if any(stage_status[stage] != "complete" for stage in FULL_CORE_STAGES):
        raise ValidationError("research, trader và risk phải complete")
    raw_stage_fingerprints = payload.get("stage_fingerprints")
    if not isinstance(raw_stage_fingerprints, Mapping):
        raise ValidationError("stage_fingerprints phải có đúng 7 stage")
    stage_fingerprints = dict(raw_stage_fingerprints)
    if set(stage_fingerprints) != set(FULL_STAGE_KEYS):
        raise ValidationError("stage_fingerprints phải có đúng 7 stage")
    if any(
        not isinstance(value, str) or not SHA256_RE.fullmatch(value)
        for value in stage_fingerprints.values()
    ):
        raise ValidationError("stage_fingerprints phải là SHA-256 lowercase")

    hashes: dict[str, str] = {}
    for field in (
        "identity_hash",
        "parent_identity_hash",
        "expected_digest_hash",
        "full_report_hash",
        "prompt_fingerprint",
        "model_fingerprint",
        "evidence_fingerprint",
    ):
        value = payload.get(field)
        if not isinstance(value, str) or not SHA256_RE.fullmatch(value):
            raise ValidationError(f"{field} phải là SHA-256 lowercase")
        hashes[field] = value
    execution_generation = payload.get("execution_generation")
    if (
        isinstance(execution_generation, bool)
        or not isinstance(execution_generation, int)
        or execution_generation < 1
    ):
        raise ValidationError("execution_generation phải là số nguyên dương")
    digest_status = payload.get("status")
    if digest_status not in {"complete", "partial"}:
        raise ValidationError("status của Full Analysis không hợp lệ")
    expected_status = (
        "complete"
        if all(value == "complete" for value in stage_status.values())
        else "partial"
    )
    if digest_status != expected_status:
        raise ValidationError("status không khớp tổng trạng thái stage")

    raw_sections = payload.get("sections")
    if not isinstance(raw_sections, Mapping) or set(raw_sections) != set(
        FULL_SECTION_ORDER
    ):
        raise ValidationError("Full Analysis phải có đúng 12 sections")
    sections: dict[str, str] = {}
    section_statuses: dict[str, str] = {}
    for section in FULL_SECTION_ORDER:
        raw = raw_sections[section]
        if not isinstance(raw, Mapping) or set(raw) != {"status", "text"}:
            raise ValidationError("Full Analysis section chỉ nhận status và text")
        status = raw.get("status")
        text_value = raw.get("text")
        if not isinstance(status, str) or not isinstance(text_value, str):
            raise ValidationError("Full Analysis section không hợp lệ")
        text = text_value
        if status not in FULL_STAGE_STATUSES:
            raise ValidationError("Trạng thái Full Analysis section không hợp lệ")
        if not text.strip():
            raise ValidationError("Full Analysis section không được để trống")
        if "sentinel" in text.casefold() or FULL_REPORT_FORBIDDEN_TEXT_RE.search(
            text
        ):
            raise ValidationError(
                "Full Analysis section chứa raw marker, credential hoặc URL"
            )
        if len(text.encode("utf-8")) > FULL_SECTION_MAX_BYTES:
            raise ValidationError(
                f"Full Analysis section vượt quá {FULL_SECTION_MAX_BYTES} bytes"
            )
        if status != stage_status[FULL_SECTION_STAGE[section]]:
            raise ValidationError("Section status không khớp stage_status")
        sections[section] = text
        section_statuses[section] = status
    total_size = sum(len(value.encode("utf-8")) for value in sections.values())
    if total_size > DIGEST_MAX_BYTES:
        raise ValidationError(f"Full Analysis vượt quá {DIGEST_MAX_BYTES} bytes")

    canonical_wire_sections = {
        section: {
            "status": section_statuses[section],
            "text": sections[section],
        }
        for section in FULL_SECTION_ORDER
    }
    full_report_hash = hashes["full_report_hash"]
    if full_report_hash != _sha256_json(canonical_wire_sections):
        raise ValidationError("full_report_hash không khớp sections")
    canonical_report = {
        "schema_version": 2,
        "content_kind": FULL_REPORT_V1,
        "report_schema": FULL_REPORT_V1,
        "source": source,
        "ticker": ticker,
        "analysis_date": analysis_date,
        "cutoff": cutoff,
        "source_updated_at": source_updated_at,
        "liquidity_rank": rank,
        "research_mode": research_mode,
        "data_provenance": data_provenance,
        "identity_hash": hashes["identity_hash"],
        "parent_identity_hash": hashes["parent_identity_hash"],
        "expected_digest_hash": hashes["expected_digest_hash"],
        "full_report_hash": full_report_hash,
        "pipeline_version": FULL_PIPELINE_VERSION,
        "prompt_fingerprint": hashes["prompt_fingerprint"],
        "model_fingerprint": hashes["model_fingerprint"],
        "evidence_fingerprint": hashes["evidence_fingerprint"],
        "execution_generation": execution_generation,
        "contains_decision_content": True,
        "status": digest_status,
        "stage_status": stage_status,
        "stage_fingerprints": stage_fingerprints,
        "sections": canonical_wire_sections,
    }
    canonical_json = _canonical_json(canonical_report)
    if len(canonical_json.encode("utf-8")) > DIGEST_MAX_BYTES:
        raise ValidationError(f"Full Analysis vượt quá {DIGEST_MAX_BYTES} bytes")
    canonical_sha256 = _sha256_json(canonical_report)
    decision, decision_json, decision_sha256, decision_status = _derive_decision(
        sections["portfolio.decision"], sections["trading.plan"]
    )
    identity = [source, ticker, analysis_date, cutoff]
    source_key = _sha256_json(identity)
    document_id = f"digest_{source_key}"
    stable_content = {
        "identity": identity,
        "content_kind": FULL_REPORT_V1,
        "report_schema": FULL_REPORT_V1,
        "liquidity_rank": rank,
        "identity_hash": hashes["identity_hash"],
        "parent_identity_hash": hashes["parent_identity_hash"],
        "expected_digest_hash": hashes["expected_digest_hash"],
        "full_report_hash": full_report_hash,
        "prompt_fingerprint": hashes["prompt_fingerprint"],
        "model_fingerprint": hashes["model_fingerprint"],
        "evidence_fingerprint": hashes["evidence_fingerprint"],
        "status": digest_status,
        "research_mode": research_mode,
        "data_provenance": data_provenance,
        "pipeline_version": FULL_PIPELINE_VERSION,
        "execution_generation": execution_generation,
        "contains_decision_content": True,
        "stage_status": stage_status,
        "stage_fingerprints": stage_fingerprints,
        "sections": canonical_wire_sections,
    }
    content_sha256 = _sha256_json(stable_content)
    return CanonicalFullReport(
        document_id=document_id,
        source_key=source_key,
        source=source,
        ticker=ticker,
        analysis_date=analysis_date,
        cutoff=cutoff,
        source_updated_at=source_updated_at,
        source_run_id=None,
        liquidity_rank=rank,
        prompt_fingerprint=_normalise_fingerprint(
            hashes["prompt_fingerprint"], "prompt_fingerprint"
        ),
        model_fingerprint=_normalise_fingerprint(
            hashes["model_fingerprint"], "model_fingerprint"
        ),
        evidence_fingerprint=_normalise_fingerprint(
            hashes["evidence_fingerprint"], "evidence_fingerprint"
        ),
        digest_status=digest_status,
        research_mode=research_mode,
        data_provenance=data_provenance,
        identity_hash=hashes["identity_hash"],
        digest_hash=None,
        parent_identity_hash=hashes["parent_identity_hash"],
        expected_digest_hash=hashes["expected_digest_hash"],
        full_report_hash=full_report_hash,
        content_kind=FULL_REPORT_V1,
        report_schema=FULL_REPORT_V1,
        pipeline_version=FULL_PIPELINE_VERSION,
        execution_generation=execution_generation,
        contains_decision_content=True,
        stage_status=stage_status,
        stage_fingerprints=stage_fingerprints,
        sections=sections,
        section_statuses=section_statuses,
        section_stages=dict(FULL_SECTION_STAGE),
        canonical_report=canonical_report,
        canonical_json=canonical_json,
        canonical_sha256=canonical_sha256,
        decision=decision,
        decision_json=decision_json,
        decision_sha256=decision_sha256,
        decision_status=decision_status,
        projection_version=DECISION_PROJECTION_VERSION,
        content_sha256=content_sha256,
        sha256=digest_document_sha256(source_key, content_sha256),
        filename=f"{ticker}_{analysis_date}_{datetime.fromisoformat(cutoff).strftime('%H%M')}.full.digest",
    )


__all__ = [
    "DECISION_MAX_BYTES",
    "DECISION_PROJECTION_VERSION",
    "DECISION_SCHEMA_VERSION",
    "DECISION_STRUCTURED_SECTION",
    "FULL_CORE_STAGES",
    "FULL_PIPELINE_VERSION",
    "FULL_REPORT_V1",
    "FULL_SECTION_LABELS",
    "FULL_SECTION_CITATION_LABELS",
    "FULL_SECTION_ORDER",
    "FULL_SECTION_STAGE",
    "FULL_STAGE_KEYS",
    "FULL_STAGE_STATUSES",
    "CanonicalFullReport",
    "canonicalise_full_report",
]
