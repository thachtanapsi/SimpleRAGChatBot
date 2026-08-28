"""Canonical, privacy-safe TradingAgents daily digest ingestion helpers."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from datetime import UTC, date, datetime
from typing import Any, Mapping, Sequence
from zoneinfo import ZoneInfo

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .config import Settings
from .errors import ValidationError


DIGEST_SOURCE_TYPE = "trading_digest"
DIGEST_CONTENT_KIND = "daily_digest_v1"
DIGEST_MAX_BYTES = 256 * 1024
LEGACY_PROVENANCE = "legacy_unknown"
RESEARCH_MODE_PROVENANCE = {
    "eod": "eod_cutoff",
    "live": "request_cutoff",
    "historical": "historical_replay",
}
DIGEST_SECTION_ORDER = (
    "summary",
    "market",
    "fundamentals",
    "news_events",
    "sentiment",
    "macro",
    "catalysts",
    "risks_unknowns",
    "next_session_watch",
)
DIGEST_SECTION_LABELS = {
    "summary": "Tổng quan",
    "market": "Thị trường và thanh khoản",
    "fundamentals": "Cơ bản",
    "news_events": "Tin tức và sự kiện",
    "sentiment": "Tâm lý",
    "macro": "Vĩ mô",
    "catalysts": "Yếu tố hỗ trợ",
    "risks_unknowns": "Rủi ro và điểm chưa rõ",
    "next_session_watch": "Theo dõi phiên kế tiếp",
}
TICKER_RE = re.compile(r"^[A-Z0-9][A-Z0-9._-]{0,19}$")
FINGERPRINT_RE = re.compile(r"^[A-Za-z0-9:_-]{1,256}$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def _sha256_json(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def digest_content_sha256(
    *,
    source: str,
    ticker: str,
    analysis_date: str,
    cutoff: str,
    liquidity_rank: int | None,
    prompt_fingerprint: str | None,
    model_fingerprint: str | None,
    evidence_fingerprint: str | None,
    digest_status: str,
    research_mode: str,
    data_provenance: str,
    sections: Mapping[str, str],
) -> str:
    """Hash the persisted digest fields used by canonical ingestion.

    Keeping this in one place lets a legacy batch acquire its asserted
    provenance without requiring the original wire payload or an embedding
    rebuild.  ``source_updated_at`` deliberately remains a watermark rather
    than content identity.
    """

    stable_content: dict[str, Any] = {
        "identity": [source, ticker, analysis_date, cutoff],
        "liquidity_rank": liquidity_rank,
        "prompt_fingerprint": prompt_fingerprint,
        "model_fingerprint": model_fingerprint,
        "evidence_fingerprint": evidence_fingerprint,
        "status": digest_status,
        "sections": dict(sections),
    }
    # Preserve the v1 hash for unlabelled legacy rows.  Provenance is part of
    # every labelled digest hash and therefore cannot change silently.
    if research_mode != LEGACY_PROVENANCE:
        stable_content["research_mode"] = research_mode
        stable_content["data_provenance"] = data_provenance
    return _sha256_json(stable_content)


def digest_document_sha256(source_key: str, content_sha256: str) -> str:
    """Return the globally unique documents.sha256 value for a digest."""

    return hashlib.sha256(f"{source_key}:{content_sha256}".encode()).hexdigest()


def _normalise_datetime(
    value: str | datetime, field: str, *, timezone: ZoneInfo | None = None
) -> str:
    try:
        parsed = value if isinstance(value, datetime) else datetime.fromisoformat(value)
    except ValueError as exc:
        raise ValidationError(f"{field} phải là ISO-8601") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValidationError(f"{field} phải có timezone")
    return parsed.astimezone(timezone or UTC).isoformat()


def normalise_cutoff(value: str | datetime) -> str:
    return _normalise_datetime(
        value, "cutoff", timezone=ZoneInfo("Asia/Ho_Chi_Minh")
    )


def normalise_source_updated_at(value: str | datetime) -> str:
    return _normalise_datetime(value, "source_updated_at")


def normalise_analysis_date(value: str | date) -> str:
    return _normalise_date(value)


def normalise_provenance_pair(
    research_mode: Any,
    data_provenance: Any,
    *,
    allow_legacy: bool = True,
) -> tuple[str, str]:
    """Validate the provenance pair without guessing from source/date/cutoff.

    Older digest clients did not send either field.  They remain readable as
    ``legacy_unknown``; sending only one field is rejected so newly labelled
    data can never silently acquire an incorrect provenance.
    """

    if research_mode is None and data_provenance is None:
        if allow_legacy:
            return LEGACY_PROVENANCE, LEGACY_PROVENANCE
        raise ValidationError("research_mode và data_provenance là bắt buộc")
    if research_mode is None or data_provenance is None:
        raise ValidationError(
            "research_mode và data_provenance phải được gửi cùng nhau"
        )
    mode = str(research_mode).strip()
    provenance = str(data_provenance).strip()
    if allow_legacy and mode == LEGACY_PROVENANCE and provenance == LEGACY_PROVENANCE:
        return mode, provenance
    expected = RESEARCH_MODE_PROVENANCE.get(mode)
    if expected is None:
        raise ValidationError("research_mode không hợp lệ")
    if provenance != expected:
        raise ValidationError("data_provenance không khớp research_mode")
    return mode, provenance


def _normalise_date(value: str | date) -> str:
    try:
        parsed = value if isinstance(value, date) else date.fromisoformat(value)
    except ValueError as exc:
        raise ValidationError("analysis_date phải có dạng YYYY-MM-DD") from exc
    return parsed.isoformat()


def _normalise_fingerprint(value: Any, field: str) -> str | None:
    if value is None:
        return None
    normalised = str(value).strip()
    if not FINGERPRINT_RE.fullmatch(normalised):
        raise ValidationError(f"{field} không hợp lệ")
    return normalised


def _claim_text(claim: Mapping[str, Any]) -> str:
    text = str(claim.get("text") or "").strip()
    if not text:
        return ""
    details: list[str] = []
    confidence = claim.get("confidence")
    if confidence is not None and str(confidence).strip():
        details.append(f"độ tin cậy: {str(confidence).strip()}")
    evidence_ids = [
        str(item).strip()
        for item in claim.get("evidence_ids", [])
        if str(item).strip()
    ]
    if evidence_ids:
        details.append(f"bằng chứng: {', '.join(evidence_ids)}")
    suffix = f" ({'; '.join(details)})" if details else ""
    return f"- {text}{suffix}"


def render_section(value: Any) -> str:
    """Render only the explicitly allowed digest/claim shape.

    Pydantic validates the wire payload first. Keeping this renderer defensive
    makes direct service callers safe as well and prevents arbitrary evidence
    dictionaries from being serialized into SQLite or Chroma.
    """

    if isinstance(value, str):
        return value.strip()
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return "\n".join(
            rendered for item in value if isinstance(item, Mapping)
            if (rendered := _claim_text(item))
        )
    if isinstance(value, Mapping):
        allowed = {"text", "claims"}
        if set(value) - allowed:
            raise ValidationError("Digest section chứa trường không được phép")
        parts: list[str] = []
        text = str(value.get("text") or "").strip()
        if text:
            parts.append(text)
        claims = value.get("claims") or []
        if not isinstance(claims, Sequence) or isinstance(
            claims, (str, bytes, bytearray)
        ):
            raise ValidationError("claims phải là danh sách")
        rendered_claims = [
            rendered for item in claims if isinstance(item, Mapping)
            if (rendered := _claim_text(item))
        ]
        if rendered_claims:
            parts.append("\n".join(rendered_claims))
        return "\n".join(parts)
    raise ValidationError("Digest section không hợp lệ")


@dataclass(frozen=True, slots=True)
class CanonicalDigest:
    document_id: str
    source_key: str
    source: str
    ticker: str
    analysis_date: str
    cutoff: str
    source_updated_at: str
    source_run_id: str | None
    liquidity_rank: int | None
    prompt_fingerprint: str | None
    model_fingerprint: str | None
    evidence_fingerprint: str | None
    digest_status: str
    research_mode: str
    data_provenance: str
    identity_hash: str | None
    digest_hash: str | None
    parent_identity_hash: None
    expected_digest_hash: None
    full_report_hash: None
    content_kind: str
    report_schema: str
    pipeline_version: None
    execution_generation: None
    contains_decision_content: bool
    stage_status: dict[str, str]
    stage_fingerprints: dict[str, str]
    sections: dict[str, str]
    section_statuses: dict[str, str]
    section_stages: dict[str, str]
    content_sha256: str
    sha256: str
    filename: str

    @property
    def size_bytes(self) -> int:
        return sum(len(value.encode("utf-8")) for value in self.sections.values())


def canonicalise_digest(
    payload: Mapping[str, Any],
    *,
    batch_source: str,
    batch_research_mode: str = LEGACY_PROVENANCE,
    batch_data_provenance: str = LEGACY_PROVENANCE,
) -> CanonicalDigest:
    source = str(payload.get("source") or batch_source).strip()
    if source != batch_source:
        raise ValidationError("source của digest không khớp digest batch")
    ticker = str(payload.get("ticker") or "").strip().upper()
    if not TICKER_RE.fullmatch(ticker):
        raise ValidationError("ticker không hợp lệ")
    analysis_date = _normalise_date(payload.get("analysis_date"))
    cutoff = normalise_cutoff(payload.get("cutoff"))
    if datetime.fromisoformat(cutoff).date().isoformat() != analysis_date:
        raise ValidationError("cutoff phải thuộc analysis_date theo Asia/Ho_Chi_Minh")
    source_updated_at = normalise_source_updated_at(payload.get("source_updated_at"))
    rank = payload.get("liquidity_rank")
    if rank is not None:
        rank = int(rank)
        if rank <= 0:
            raise ValidationError("liquidity_rank phải lớn hơn 0")
    digest_status = str(payload.get("status") or "complete")
    if digest_status not in {"complete", "partial"}:
        raise ValidationError("status của digest không hợp lệ")
    batch_provenance = normalise_provenance_pair(
        batch_research_mode, batch_data_provenance
    )
    if (
        payload.get("research_mode") is None
        and payload.get("data_provenance") is None
    ):
        research_mode, data_provenance = batch_provenance
    else:
        research_mode, data_provenance = normalise_provenance_pair(
            payload.get("research_mode"), payload.get("data_provenance")
        )
        if (research_mode, data_provenance) != batch_provenance:
            raise ValidationError("Provenance của digest không khớp digest batch")

    raw_sections = payload.get("sections")
    if not isinstance(raw_sections, Mapping):
        raise ValidationError("sections không hợp lệ")
    if set(raw_sections) != set(DIGEST_SECTION_ORDER):
        raise ValidationError("sections phải có đúng 9 mục daily digest")
    sections = {
        section: render_section(raw_sections[section])
        for section in DIGEST_SECTION_ORDER
    }
    if not any(sections.values()):
        raise ValidationError("Digest không có nội dung")
    digest_size = sum(len(value.encode("utf-8")) for value in sections.values())
    if digest_size > DIGEST_MAX_BYTES:
        raise ValidationError(f"Digest vượt quá {DIGEST_MAX_BYTES} bytes")
    identity_hash = payload.get("identity_hash")
    if identity_hash is not None:
        identity_hash = str(identity_hash).strip()
        if not SHA256_RE.fullmatch(identity_hash):
            raise ValidationError("identity_hash phải là SHA-256 lowercase")
    digest_hash = payload.get("digest_hash")
    if digest_hash is not None:
        digest_hash = str(digest_hash).strip()
        if not SHA256_RE.fullmatch(digest_hash):
            raise ValidationError("digest_hash phải là SHA-256 lowercase")
        if digest_hash != _sha256_json(dict(raw_sections)):
            raise ValidationError("digest_hash không khớp sections")

    identity = [source, ticker, analysis_date, cutoff]
    source_key = _sha256_json(identity)
    document_id = f"digest_{source_key}"
    prompt_fingerprint = _normalise_fingerprint(
        payload.get("prompt_fingerprint"), "prompt_fingerprint"
    )
    model_fingerprint = _normalise_fingerprint(
        payload.get("model_fingerprint"), "model_fingerprint"
    )
    evidence_fingerprint = _normalise_fingerprint(
        payload.get("evidence_fingerprint"), "evidence_fingerprint"
    )
    content_sha256 = digest_content_sha256(
        source=source,
        ticker=ticker,
        analysis_date=analysis_date,
        cutoff=cutoff,
        liquidity_rank=rank,
        prompt_fingerprint=prompt_fingerprint,
        model_fingerprint=model_fingerprint,
        evidence_fingerprint=evidence_fingerprint,
        digest_status=digest_status,
        research_mode=research_mode,
        data_provenance=data_provenance,
        sections=sections,
    )
    # documents.sha256 remains globally unique for backward-compatible PDF
    # deduplication; digest content hash itself is retained separately.
    sha256 = digest_document_sha256(source_key, content_sha256)
    cutoff_compact = datetime.fromisoformat(cutoff).strftime("%H%M")
    return CanonicalDigest(
        document_id=document_id,
        source_key=source_key,
        source=source,
        ticker=ticker,
        analysis_date=analysis_date,
        cutoff=cutoff,
        source_updated_at=source_updated_at,
        source_run_id=_normalise_fingerprint(payload.get("source_run_id"), "source_run_id"),
        liquidity_rank=rank,
        prompt_fingerprint=prompt_fingerprint,
        model_fingerprint=model_fingerprint,
        evidence_fingerprint=evidence_fingerprint,
        digest_status=digest_status,
        research_mode=research_mode,
        data_provenance=data_provenance,
        identity_hash=identity_hash,
        digest_hash=digest_hash,
        parent_identity_hash=None,
        expected_digest_hash=None,
        full_report_hash=None,
        content_kind=DIGEST_CONTENT_KIND,
        report_schema=DIGEST_CONTENT_KIND,
        pipeline_version=None,
        execution_generation=None,
        contains_decision_content=False,
        stage_status={},
        stage_fingerprints={},
        sections=sections,
        section_statuses={
            section: "complete" if text else "unavailable"
            for section, text in sections.items()
        },
        section_stages={section: "digest" for section in sections},
        content_sha256=content_sha256,
        sha256=sha256,
        filename=f"{ticker}_{analysis_date}_{cutoff_compact}.digest",
    )


def _chunk_id(*parts: object, prefix: str) -> str:
    raw = "\x1f".join(str(part) for part in parts).encode("utf-8")
    return f"{prefix}_{hashlib.sha256(raw).hexdigest()}"


def build_digest_chunks(
    *,
    document: Mapping[str, Any],
    sections: Sequence[Mapping[str, Any]],
    settings: Settings,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str], list[dict[str, Any]]]:
    """Chunk each digest section independently; overlap never crosses sections."""

    parent_splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.parent_chunk_size,
        chunk_overlap=settings.parent_chunk_overlap,
        add_start_index=True,
        strip_whitespace=True,
    )
    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.child_chunk_size,
        chunk_overlap=settings.child_chunk_overlap,
        add_start_index=True,
        strip_whitespace=True,
    )
    parents: list[dict[str, Any]] = []
    children: list[dict[str, Any]] = []
    child_texts: list[str] = []
    child_metadatas: list[dict[str, Any]] = []
    document_id = str(document["id"])
    content_hash = str(document["content_sha256"])
    # Imported lazily because full_reports builds on the daily canonical helpers.
    from .full_reports import FULL_SECTION_LABELS
    try:
        stage_status = json.loads(str(document.get("stage_status_json") or "{}"))
        stage_fingerprints = json.loads(
            str(document.get("stage_fingerprints_json") or "{}")
        )
    except (TypeError, ValueError):
        stage_status, stage_fingerprints = {}, {}

    for stored in sorted(sections, key=lambda item: int(item["position"])):
        section = str(stored["section"])
        report_stage = str(stored.get("report_stage") or "digest")
        section_status = str(stored.get("section_status") or "complete")
        text = str(stored["text"]).strip()
        if not text:
            continue
        heading = FULL_SECTION_LABELS.get(
            section, DIGEST_SECTION_LABELS.get(section, section)
        )
        section_text = f"{heading}\n{text}"
        source = Document(page_content=section_text)
        for parent_doc in parent_splitter.split_documents([source]):
            parent_start = int(parent_doc.metadata.get("start_index", 0))
            parent_id = _chunk_id(
                document_id, content_hash, section, parent_start, prefix="parent_digest"
            )
            parents.append(
                {
                    "id": parent_id,
                    "document_id": document_id,
                    "page": 0,
                    "page_end": 0,
                    "kind": "page",
                    "digest_section": section,
                    "report_stage": report_stage,
                    "section_status": section_status,
                    "start_index": parent_start,
                    "text": parent_doc.page_content,
                }
            )
            for child_doc in child_splitter.split_documents([parent_doc]):
                local_start = int(child_doc.metadata.get("start_index", 0))
                child_start = parent_start + local_start
                child_id = _chunk_id(
                    document_id,
                    content_hash,
                    section,
                    parent_start,
                    local_start,
                    prefix="child_digest",
                )
                child = {
                    "id": child_id,
                    "parent_id": parent_id,
                    "document_id": document_id,
                    "page": 0,
                    "page_end": 0,
                    "kind": "page",
                    "digest_section": section,
                    "report_stage": report_stage,
                    "section_status": section_status,
                    "start_index": child_start,
                    "text": child_doc.page_content,
                }
                children.append(child)
                child_texts.append(child_doc.page_content)
                metadata: dict[str, Any] = {
                    "child_id": child_id,
                    "parent_id": parent_id,
                    "document_id": document_id,
                    "filename": str(document["filename"]),
                    "page": 0,
                    "page_end": 0,
                    "kind": "page",
                    "source_type": DIGEST_SOURCE_TYPE,
                    "ticker": str(document["ticker"]),
                    "analysis_date": str(document["analysis_date"]),
                    "analysis_date_ordinal": date.fromisoformat(
                        str(document["analysis_date"])
                    ).toordinal(),
                    "analysis_cutoff": str(document["analysis_cutoff"]),
                    "batch_id": str(document["batch_id"]),
                    "digest_section": section,
                    "report_stage": report_stage,
                    "section_status": section_status,
                    "research_mode": str(document["research_mode"]),
                    "data_provenance": str(document["data_provenance"]),
                    "content_kind": str(document["content_kind"]),
                    "report_schema": str(document["report_schema"]),
                    "contains_decision_content": bool(
                        document.get("contains_decision_content")
                    ),
                    "start_index": child_start,
                }
                if document.get("liquidity_rank") is not None:
                    metadata["liquidity_rank"] = int(document["liquidity_rank"])
                for fingerprint_field in (
                    "prompt_fingerprint",
                    "model_fingerprint",
                    "evidence_fingerprint",
                ):
                    if document.get(fingerprint_field):
                        metadata[fingerprint_field] = str(document[fingerprint_field])
                for report_field in (
                    "identity_hash",
                    "digest_hash",
                    "parent_identity_hash",
                    "expected_digest_hash",
                    "full_report_hash",
                    "pipeline_version",
                ):
                    if document.get(report_field):
                        metadata[report_field] = str(document[report_field])
                if document.get("execution_generation") is not None:
                    metadata["execution_generation"] = int(
                        document["execution_generation"]
                    )
                if report_stage in stage_status:
                    metadata["report_stage_status"] = str(
                        stage_status[report_stage]
                    )
                if report_stage in stage_fingerprints:
                    metadata["stage_fingerprint"] = str(
                        stage_fingerprints[report_stage]
                    )
                child_metadatas.append(metadata)
    return parents, children, child_texts, child_metadatas
