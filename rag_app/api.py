"""FastAPI REST/SSE server và static web UI chạy trên localhost."""

from __future__ import annotations

import hmac
import json
import re
import time
import uuid
from contextlib import asynccontextmanager
from datetime import date, datetime
from typing import Annotated, Any, AsyncIterator, Literal

from fastapi import Depends, FastAPI, File, HTTPException, Query, Request, Response, UploadFile
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .config import Settings
from .digests import (
    DIGEST_CONTENT_KIND,
    DIGEST_SECTION_ORDER,
    RESEARCH_MODE_PROVENANCE,
    normalise_analysis_date,
    normalise_cutoff,
    normalise_provenance_pair,
)
from .errors import ConflictError, DuplicateDocumentError, RAGError
from .full_reports import (
    DECISION_STRUCTURED_SECTION,
    FULL_REPORT_V1,
    FULL_SECTION_ORDER,
    FULL_STAGE_KEYS,
    canonicalise_full_report,
)
from .runtime import Runtime


class RetrievalFilters(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_types: list[str] | None = Field(default=None, max_length=2)
    tickers: list[str] | None = Field(default=None, max_length=500)
    date_from: date | None = None
    date_to: date | None = None
    liquidity_rank_min: int | None = Field(default=None, ge=1, le=10000)
    liquidity_rank_max: int | None = Field(default=None, ge=1, le=10000)
    digest_sections: list[str] | None = Field(default=None, max_length=22)
    batch_ids: list[str] | None = Field(default=None, max_length=100)
    research_modes: list[str] | None = Field(default=None, max_length=4)
    data_provenances: list[str] | None = Field(default=None, max_length=4)
    content_kinds: list[str] | None = Field(default=None, max_length=4)
    report_stages: list[str] | None = Field(default=None, max_length=8)

    @field_validator("source_types")
    @classmethod
    def validate_source_types(cls, value: list[str] | None) -> list[str] | None:
        if value is not None and any(item not in {"pdf", "trading_digest"} for item in value):
            raise ValueError("source_types không hợp lệ")
        return value

    @field_validator("tickers")
    @classmethod
    def validate_tickers(cls, value: list[str] | None) -> list[str] | None:
        if value is None:
            return None
        normalised = [item.strip().upper() for item in value]
        if any(not re.fullmatch(r"[A-Z0-9][A-Z0-9._-]{0,19}", item) for item in normalised):
            raise ValueError("ticker filter không hợp lệ")
        return list(dict.fromkeys(normalised))

    @field_validator("digest_sections")
    @classmethod
    def validate_sections(cls, value: list[str] | None) -> list[str] | None:
        allowed = {
            *DIGEST_SECTION_ORDER,
            *FULL_SECTION_ORDER,
            DECISION_STRUCTURED_SECTION,
        }
        if value is not None and any(item not in allowed for item in value):
            raise ValueError("digest_sections không hợp lệ")
        return value

    @field_validator("content_kinds")
    @classmethod
    def validate_content_kinds(cls, value: list[str] | None) -> list[str] | None:
        allowed = {"pdf", DIGEST_CONTENT_KIND, FULL_REPORT_V1, "legacy_unknown"}
        if value is not None and any(item not in allowed for item in value):
            raise ValueError("content_kinds không hợp lệ")
        return value

    @field_validator("report_stages")
    @classmethod
    def validate_report_stages(cls, value: list[str] | None) -> list[str] | None:
        allowed = {"digest", *FULL_STAGE_KEYS}
        if value is not None and any(item not in allowed for item in value):
            raise ValueError("report_stages không hợp lệ")
        return value

    @field_validator("research_modes")
    @classmethod
    def validate_research_modes(cls, value: list[str] | None) -> list[str] | None:
        allowed = {*RESEARCH_MODE_PROVENANCE, "legacy_unknown"}
        if value is not None and any(item not in allowed for item in value):
            raise ValueError("research_modes không hợp lệ")
        return value

    @field_validator("data_provenances")
    @classmethod
    def validate_data_provenances(cls, value: list[str] | None) -> list[str] | None:
        allowed = {*RESEARCH_MODE_PROVENANCE.values(), "legacy_unknown"}
        if value is not None and any(item not in allowed for item in value):
            raise ValueError("data_provenances không hợp lệ")
        return value

    @model_validator(mode="after")
    def validate_ranges(self) -> "RetrievalFilters":
        if self.date_from and self.date_to and self.date_from > self.date_to:
            raise ValueError("date_from phải trước hoặc bằng date_to")
        if (
            self.liquidity_rank_min is not None
            and self.liquidity_rank_max is not None
            and self.liquidity_rank_min > self.liquidity_rank_max
        ):
            raise ValueError("liquidity_rank_min phải <= liquidity_rank_max")
        return self

    def storage_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json", exclude_none=True)


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    session_id: str | None = None
    document_ids: list[str] | None = None
    filters: RetrievalFilters | None = None


class DigestClaim(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str = Field(min_length=1, max_length=2000)
    confidence: (
        Annotated[str, Field(max_length=32)]
        | Annotated[float, Field(ge=0, le=1)]
        | None
    ) = None
    evidence_ids: list[str] = Field(default_factory=list, max_length=12)

    @field_validator("evidence_ids")
    @classmethod
    def validate_evidence_ids(cls, value: list[str]) -> list[str]:
        if any(
            not re.fullmatch(r"[A-Za-z0-9:._/-]{1,160}", item.strip())
            for item in value
        ):
            raise ValueError("evidence_id không hợp lệ")
        return list(dict.fromkeys(item.strip() for item in value))


class DigestSectionPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str | None = Field(default=None, max_length=20000)
    claims: list[DigestClaim] = Field(default_factory=list, max_length=8)

    @model_validator(mode="after")
    def non_empty(self) -> "DigestSectionPayload":
        if not (self.text or "").strip() and not self.claims:
            raise ValueError("Digest section phải có text hoặc claims")
        return self


DigestSectionValue = (
    Annotated[str, Field(max_length=20000)]
    | DigestSectionPayload
    | Annotated[list[DigestClaim], Field(max_length=8)]
)


class DigestSections(BaseModel):
    model_config = ConfigDict(extra="forbid")

    summary: DigestSectionValue
    market: DigestSectionValue
    fundamentals: DigestSectionValue
    news_events: DigestSectionValue
    sentiment: DigestSectionValue
    macro: DigestSectionValue
    catalysts: DigestSectionValue
    risks_unknowns: DigestSectionValue
    next_session_watch: DigestSectionValue


class DigestDocumentPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: int = Field(default=1, ge=1, le=1)
    source: str | None = Field(default=None, min_length=1, max_length=128)
    identity_hash: str | None = Field(default=None, pattern="^[0-9a-f]{64}$")
    digest_hash: str | None = Field(default=None, pattern="^[0-9a-f]{64}$")
    source_run_id: str | None = Field(default=None, min_length=1, max_length=256)
    ticker: str = Field(min_length=1, max_length=20)
    analysis_date: date
    cutoff: datetime
    source_updated_at: datetime
    liquidity_rank: int | None = Field(default=None, ge=1, le=500)
    prompt_fingerprint: str | None = Field(default=None, max_length=256)
    model_fingerprint: str | None = Field(default=None, max_length=256)
    evidence_fingerprint: str | None = Field(default=None, max_length=256)
    status: str = Field(default="complete", pattern="^(complete|partial)$")
    research_mode: Literal["eod", "live", "historical"] | None = None
    data_provenance: Literal[
        "eod_cutoff", "request_cutoff", "historical_replay"
    ] | None = None
    sections: DigestSections

    @model_validator(mode="after")
    def validate_provenance(self) -> "DigestDocumentPayload":
        _validate_wire_provenance(self.research_mode, self.data_provenance)
        return self


class DigestDocumentsRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    documents: list[DigestDocumentPayload] = Field(min_length=1, max_length=50)


SHA256_PATTERN = "^[0-9a-f]{64}$"


class FullReportSectionPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: Literal["complete", "partial", "unavailable"]
    text: str = Field(min_length=1, max_length=32768)


class FullReportDocumentPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[2]
    content_kind: Literal["full_report_v1"]
    report_schema: Literal["full_report_v1"]
    source: str = Field(min_length=1, max_length=128)
    ticker: str = Field(min_length=1, max_length=20)
    analysis_date: date
    cutoff: datetime
    source_updated_at: datetime
    liquidity_rank: int = Field(ge=1, le=500)
    research_mode: Literal["eod", "live", "historical"]
    data_provenance: Literal[
        "eod_cutoff", "request_cutoff", "historical_replay"
    ]
    identity_hash: str = Field(pattern=SHA256_PATTERN)
    parent_identity_hash: str = Field(pattern=SHA256_PATTERN)
    expected_digest_hash: str = Field(pattern=SHA256_PATTERN)
    full_report_hash: str = Field(pattern=SHA256_PATTERN)
    pipeline_version: Literal["gx_full_v1"]
    prompt_fingerprint: str = Field(pattern=SHA256_PATTERN)
    model_fingerprint: str = Field(pattern=SHA256_PATTERN)
    evidence_fingerprint: str = Field(pattern=SHA256_PATTERN)
    execution_generation: int = Field(ge=1)
    contains_decision_content: Literal[True]
    status: Literal["complete", "partial"]
    stage_status: dict[str, Literal["complete", "partial", "unavailable"]]
    stage_fingerprints: dict[str, str]
    sections: dict[str, FullReportSectionPayload]

    @model_validator(mode="after")
    def validate_frozen_contract(self) -> "FullReportDocumentPayload":
        _validate_wire_provenance(self.research_mode, self.data_provenance)
        if set(self.stage_status) != set(FULL_STAGE_KEYS):
            raise ValueError("stage_status phải có đúng 7 stage")
        if set(self.stage_fingerprints) != set(FULL_STAGE_KEYS):
            raise ValueError("stage_fingerprints phải có đúng 7 stage")
        if any(
            not re.fullmatch(SHA256_PATTERN, value)
            for value in self.stage_fingerprints.values()
        ):
            raise ValueError("stage_fingerprints phải là SHA-256 lowercase")
        if any(self.stage_status[stage] != "complete" for stage in ("research", "trader", "risk")):
            raise ValueError("research, trader và risk phải complete")
        if set(self.sections) != set(FULL_SECTION_ORDER):
            raise ValueError("Full Analysis phải có đúng 12 sections")
        return self


class FullReportPromotionsRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    documents: list[FullReportDocumentPayload] = Field(min_length=1, max_length=50)


class DigestBatchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source: str = Field(default="tradingagents_daily_research", min_length=1, max_length=128)
    analysis_date: date
    cutoff: datetime
    target_count: int = Field(default=500, ge=1, le=500)
    research_mode: Literal["eod", "live", "historical"] | None = None
    data_provenance: Literal[
        "eod_cutoff", "request_cutoff", "historical_replay"
    ] | None = None

    @model_validator(mode="after")
    def validate_provenance(self) -> "DigestBatchRequest":
        _validate_wire_provenance(self.research_mode, self.data_provenance)
        return self


def _validate_wire_provenance(
    research_mode: str | None, data_provenance: str | None
) -> None:
    if (research_mode is None) != (data_provenance is None):
        raise ValueError("research_mode và data_provenance phải được gửi cùng nhau")
    if research_mode is not None and RESEARCH_MODE_PROVENANCE[research_mode] != data_provenance:
        raise ValueError("data_provenance không khớp research_mode")


def public_document(document: dict[str, Any]) -> dict[str, Any]:
    return {
        key: document.get(key)
        for key in (
            "id",
            "filename",
            "size_bytes",
            "page_count",
            "status",
            "error",
            "created_at",
            "updated_at",
            "source_type",
            "ticker",
            "analysis_date",
            "analysis_cutoff",
            "batch_id",
            "liquidity_rank",
            "digest_status",
            "research_mode",
            "data_provenance",
            "content_kind",
            "report_schema",
            "identity_hash",
            "digest_hash",
            "parent_identity_hash",
            "expected_digest_hash",
            "full_report_hash",
            "pipeline_version",
            "execution_generation",
            "contains_decision_content",
        )
    }


def sse(event: str, data: dict[str, Any]) -> str:
    encoded = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    return f"event: {event}\ndata: {encoded}\n\n"


def full_report_response(
    document: dict[str, Any], payload: dict[str, Any] | None
) -> dict[str, Any]:
    """Read and independently verify a persisted canonical report."""

    if payload is None:
        raise HTTPException(status_code=503, detail="Full Analysis payload bị thiếu")
    try:
        report = json.loads(payload["canonical_json"])
        decision = json.loads(payload["decision_json"])
        canonical = canonicalise_full_report(report)
        valid = (
            isinstance(report, dict)
            and isinstance(decision, dict)
            and canonical.document_id == document["id"]
            and canonical.full_report_hash == document["full_report_hash"]
            and canonical.canonical_json == payload["canonical_json"]
            and canonical.canonical_sha256 == payload["canonical_sha256"]
            and canonical.decision_json == payload["decision_json"]
            and canonical.decision_sha256 == payload["decision_sha256"]
            and canonical.decision_status == payload["decision_status"]
            and canonical.projection_version == payload["projection_version"]
            and int(payload["decision_index_version"])
            == canonical.projection_version
        )
    except (KeyError, TypeError, ValueError, RAGError):
        valid = False
    if not valid:
        raise HTTPException(status_code=503, detail="Full Analysis payload không hợp lệ")
    return {
        "data": {
            "document_id": document["id"],
            "canonical_report_sha256": payload["canonical_sha256"],
            "decision_sha256": payload["decision_sha256"],
            "report": report,
            "decision": decision,
        }
    }


def create_app(
    settings: Settings | None = None, runtime: Any | None = None
) -> FastAPI:
    configured_settings = settings or Settings.from_env()
    managed_runtime = runtime or Runtime(configured_settings)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.runtime = managed_runtime
        await managed_runtime.start()
        try:
            yield
        finally:
            await managed_runtime.stop()

    app = FastAPI(
        title="Simple Local RAG",
        version="1.0.0",
        lifespan=lifespan,
        docs_url="/api/docs",
        redoc_url=None,
    )

    @app.exception_handler(RAGError)
    async def rag_error_handler(_request: Request, exc: RAGError) -> JSONResponse:
        payload: dict[str, Any] = {"detail": str(exc)}
        if isinstance(exc, DuplicateDocumentError):
            payload["document_id"] = exc.document_id
        if error_code := getattr(exc, "error_code", None):
            payload["code"] = error_code
        return JSONResponse(status_code=exc.status_code, content=payload)

    @app.middleware("http")
    async def request_metadata(request: Request, call_next: Any) -> Response:
        request_id = request.headers.get("x-request-id") or str(uuid.uuid4())
        request.state.request_id = request_id
        started = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception as exc:
            managed_runtime.logger.error(
                "request failed",
                extra={
                    "event": "request_failed",
                    "request_id": request_id,
                    "elapsed_ms": round((time.perf_counter() - started) * 1000, 2),
                    "error": type(exc).__name__,
                },
            )
            raise
        response.headers["x-request-id"] = request_id
        managed_runtime.logger.info(
            "request complete",
            extra={
                "event": "request_complete",
                "request_id": request_id,
                "elapsed_ms": round((time.perf_counter() - started) * 1000, 2),
            },
        )
        return response

    @app.get("/api/health")
    async def health() -> dict[str, Any]:
        return await managed_runtime.health()

    @app.get("/api/documents")
    async def list_documents(
        page: int | None = Query(default=None, ge=1),
        page_size: int = Query(default=50, ge=1, le=100),
        source_type: str | None = Query(default=None),
        status: str | None = Query(default=None),
        ticker: str | None = Query(default=None, max_length=20),
        batch_id: str | None = Query(default=None, max_length=128),
        date_from: date | None = Query(default=None),
        date_to: date | None = Query(default=None),
        digest_section: str | None = Query(default=None),
        content_kind: str | None = Query(default=None),
        report_stage: str | None = Query(default=None),
        research_mode: str | None = Query(default=None),
        data_provenance: str | None = Query(default=None),
    ) -> Any:
        # A request without pagination retains the v1 list shape for existing clients.
        if page is None:
            return [
                public_document(item)
                for item in managed_runtime.storage.list_documents()
            ]
        filters: dict[str, Any] = {}
        if source_type:
            if source_type not in {"pdf", "trading_digest"}:
                raise HTTPException(status_code=422, detail="source_type không hợp lệ")
            filters["source_types"] = [source_type]
        if batch_id:
            filters["batch_ids"] = [batch_id]
        if date_from:
            filters["date_from"] = date_from.isoformat()
        if date_to:
            filters["date_to"] = date_to.isoformat()
        if digest_section:
            if digest_section not in {
                *DIGEST_SECTION_ORDER,
                *FULL_SECTION_ORDER,
                DECISION_STRUCTURED_SECTION,
            }:
                raise HTTPException(status_code=422, detail="digest_section không hợp lệ")
            filters["digest_sections"] = [digest_section]
        if content_kind:
            if content_kind not in {
                "pdf",
                DIGEST_CONTENT_KIND,
                FULL_REPORT_V1,
                "legacy_unknown",
            }:
                raise HTTPException(status_code=422, detail="content_kind không hợp lệ")
            filters["content_kinds"] = [content_kind]
        if report_stage:
            if report_stage not in {"digest", *FULL_STAGE_KEYS}:
                raise HTTPException(status_code=422, detail="report_stage không hợp lệ")
            filters["report_stages"] = [report_stage]
        if research_mode:
            if research_mode not in {*RESEARCH_MODE_PROVENANCE, "legacy_unknown"}:
                raise HTTPException(status_code=422, detail="research_mode không hợp lệ")
            filters["research_modes"] = [research_mode]
        if data_provenance:
            if data_provenance not in {
                *RESEARCH_MODE_PROVENANCE.values(),
                "legacy_unknown",
            }:
                raise HTTPException(status_code=422, detail="data_provenance không hợp lệ")
            filters["data_provenances"] = [data_provenance]
        result = managed_runtime.storage.list_documents_page(
            page=page,
            page_size=page_size,
            filters=filters,
            status=status,
            ticker_query=ticker,
        )
        return {**result, "items": [public_document(item) for item in result["items"]]}

    @app.post("/api/documents", status_code=202)
    async def upload_document(file: UploadFile = File(...)) -> dict[str, Any]:
        document = await managed_runtime.ingestion.accept_upload(file)
        return {"document_id": document["id"], "status": document["status"]}

    @app.get("/api/digest-batches/latest")
    async def latest_digest_batch() -> dict[str, Any]:
        batch = managed_runtime.storage.latest_digest_batch()
        if not batch:
            from .errors import DocumentNotFoundError

            raise DocumentNotFoundError("Chưa có daily digest batch")
        status = managed_runtime.storage.digest_batch_status(batch["id"])
        return {
            key: status.get(key)
            for key in (
                "id",
                "analysis_date",
                "cutoff",
                "target_count",
                "accepted_count",
                "rag_ready",
                "counts",
                "research_mode",
                "data_provenance",
            )
        }

    @app.get("/api/documents/{document_id}")
    async def get_document(document_id: str) -> dict[str, Any]:
        from .errors import DocumentNotFoundError

        document = managed_runtime.storage.get_document(document_id)
        if not document:
            raise DocumentNotFoundError("Không tìm thấy tài liệu")
        return public_document(document)

    @app.get("/api/documents/{document_id}/full-report")
    async def get_document_full_report(document_id: str) -> dict[str, Any]:
        from .errors import DocumentNotFoundError

        document = managed_runtime.storage.get_document(document_id)
        if not document:
            raise DocumentNotFoundError("Không tìm thấy tài liệu")
        if (
            document.get("source_type") != "trading_digest"
            or document.get("content_kind") != FULL_REPORT_V1
            or document.get("status") != "ready"
        ):
            raise HTTPException(
                status_code=409, detail="Full Analysis chưa được promote/ready"
            )
        return full_report_response(
            document,
            managed_runtime.storage.get_full_report_payload(document_id),
        )

    @app.delete("/api/documents/{document_id}", status_code=204)
    async def delete_document(document_id: str) -> Response:
        await managed_runtime.delete_document(document_id)
        return Response(status_code=204)

    @app.post("/api/chat")
    async def chat(payload: ChatRequest) -> dict[str, Any]:
        if payload.filters:
            return await managed_runtime.chat.answer(
                payload.question,
                payload.session_id,
                payload.document_ids,
                payload.filters.storage_dict(),
            )
        return await managed_runtime.chat.answer(
            payload.question, payload.session_id, payload.document_ids
        )

    @app.post("/api/chat/stream")
    async def chat_stream(payload: ChatRequest, request: Request) -> StreamingResponse:
        async def event_source() -> AsyncIterator[str]:
            generator = (
                managed_runtime.chat.stream(
                    payload.question,
                    payload.session_id,
                    payload.document_ids,
                    payload.filters.storage_dict(),
                )
                if payload.filters
                else managed_runtime.chat.stream(
                    payload.question, payload.session_id, payload.document_ids
                )
            )
            try:
                async for item in generator:
                    if await request.is_disconnected():
                        await generator.aclose()
                        return
                    yield sse(item["event"], item["data"])
            except RAGError as exc:
                error_data = {"detail": str(exc)}
                if error_code := getattr(exc, "error_code", None):
                    error_data["code"] = error_code
                yield sse("error", error_data)
            except Exception as exc:
                managed_runtime.logger.error(
                    "stream failed",
                    extra={
                        "event": "stream_failed",
                        "request_id": request.state.request_id,
                        "error": type(exc).__name__,
                    },
                )
                yield sse("error", {"detail": "Lỗi streaming nội bộ"})
            finally:
                await generator.aclose()

        return StreamingResponse(
            event_source(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.get("/api/sessions/{session_id}/messages")
    async def session_messages(session_id: str) -> dict[str, Any]:
        from .errors import DocumentNotFoundError

        messages = managed_runtime.storage.all_messages(session_id)
        if messages is None:
            raise DocumentNotFoundError("Không tìm thấy hội thoại")
        return {"session_id": session_id, "messages": messages}

    @app.delete("/api/sessions/{session_id}", status_code=204)
    async def delete_session(session_id: str) -> Response:
        from .errors import DocumentNotFoundError

        if not managed_runtime.storage.delete_session(session_id):
            raise DocumentNotFoundError("Không tìm thấy hội thoại")
        return Response(status_code=204)

    async def require_internal_bearer(request: Request) -> None:
        configured = configured_settings.rag_service_api_key
        if not configured:
            raise HTTPException(
                status_code=503, detail="RAG_SERVICE_API_KEY chưa được cấu hình"
            )
        header = request.headers.get("authorization", "")
        scheme, separator, supplied = header.partition(" ")
        if (
            separator != " "
            or scheme.lower() != "bearer"
            or not hmac.compare_digest(
                supplied.encode("utf-8"), configured.encode("utf-8")
            )
        ):
            raise HTTPException(
                status_code=401,
                detail="Bearer token không hợp lệ",
                headers={"WWW-Authenticate": "Bearer"},
            )

    def validate_batch_id(batch_id: str) -> str:
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}", batch_id):
            raise HTTPException(status_code=422, detail="batch_id không hợp lệ")
        return batch_id

    @app.put(
        "/internal/v1/digest-batches/{batch_id}",
        dependencies=[Depends(require_internal_bearer)],
    )
    async def put_digest_batch(
        batch_id: str, payload: DigestBatchRequest
    ) -> dict[str, Any]:
        batch_id = validate_batch_id(batch_id)
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}", payload.source):
            raise HTTPException(status_code=422, detail="source không hợp lệ")
        analysis_date = normalise_analysis_date(payload.analysis_date)
        cutoff = normalise_cutoff(payload.cutoff)
        if datetime.fromisoformat(cutoff).date().isoformat() != analysis_date:
            raise HTTPException(
                status_code=422,
                detail="cutoff phải thuộc analysis_date theo Asia/Ho_Chi_Minh",
            )
        research_mode, data_provenance = normalise_provenance_pair(
            payload.research_mode, payload.data_provenance
        )
        action, batch = await managed_runtime.ingestion.put_digest_batch(
            batch_id=batch_id,
            source=payload.source,
            analysis_date=analysis_date,
            cutoff=cutoff,
            target_count=payload.target_count,
            research_mode=research_mode,
            data_provenance=data_provenance,
        )
        if action == "conflict":
            raise ConflictError("batch_id đã tồn tại với identity khác")
        return {"action": action, "batch": managed_runtime.storage.digest_batch_status(batch_id)}

    @app.post(
        "/internal/v1/digest-batches/{batch_id}/documents:batch",
        status_code=202,
        dependencies=[Depends(require_internal_bearer)],
    )
    async def post_digest_documents(
        batch_id: str, payload: DigestDocumentsRequest
    ) -> dict[str, Any]:
        batch_id = validate_batch_id(batch_id)
        if len(payload.documents) > configured_settings.digest_max_batch_size:
            raise HTTPException(status_code=422, detail="Vượt giới hạn digest mỗi request")
        batch = managed_runtime.storage.get_digest_batch(batch_id)
        if not batch:
            from .errors import DocumentNotFoundError

            raise DocumentNotFoundError("Không tìm thấy digest batch")
        items = await managed_runtime.ingestion.accept_digest_documents(
            batch_id=batch_id,
            batch_source=batch["source"],
            payloads=[
                item.model_dump(mode="json", exclude_unset=True)
                for item in payload.documents
            ],
        )
        return {
            "batch_id": batch_id,
            "items": items,
            "status": managed_runtime.storage.digest_batch_status(batch_id),
        }

    @app.post(
        "/internal/v1/digest-batches/{batch_id}/documents:promote",
        status_code=202,
        dependencies=[Depends(require_internal_bearer)],
    )
    async def promote_full_report_documents(
        batch_id: str, payload: FullReportPromotionsRequest
    ) -> dict[str, Any]:
        batch_id = validate_batch_id(batch_id)
        if len(payload.documents) > configured_settings.digest_max_batch_size:
            raise HTTPException(
                status_code=422, detail="Vượt giới hạn Full Analysis mỗi request"
            )
        items = await managed_runtime.ingestion.accept_full_report_promotions(
            batch_id=batch_id,
            payloads=[item.model_dump(mode="json") for item in payload.documents],
        )
        return {
            "batch_id": batch_id,
            "items": items,
            "status": managed_runtime.storage.digest_batch_status(batch_id),
        }

    @app.post(
        "/internal/v1/digest-batches/{batch_id}:seal",
        dependencies=[Depends(require_internal_bearer)],
    )
    async def seal_digest_batch(batch_id: str) -> dict[str, Any]:
        from .errors import DocumentNotFoundError

        batch_id = validate_batch_id(batch_id)
        action, _batch = await managed_runtime.ingestion.seal_digest_batch(batch_id)
        if action == "missing":
            raise DocumentNotFoundError("Không tìm thấy digest batch")
        if action == "incomplete":
            raise ConflictError(
                "Chỉ seal khi đủ target_count với ticker và liquidity_rank duy nhất"
            )
        return managed_runtime.storage.digest_batch_status(batch_id)  # type: ignore[return-value]

    @app.get(
        "/internal/v1/digest-batches/{batch_id}/documents/{ticker}/full-report",
        dependencies=[Depends(require_internal_bearer)],
    )
    async def get_internal_full_report(
        batch_id: str, ticker: str
    ) -> dict[str, Any]:
        from .errors import DocumentNotFoundError

        batch_id = validate_batch_id(batch_id)
        ticker = ticker.strip().upper()
        if not re.fullmatch(r"[A-Z0-9][A-Z0-9._-]{0,19}", ticker):
            raise HTTPException(status_code=422, detail="ticker không hợp lệ")
        document = managed_runtime.storage.get_digest_batch_member(
            batch_id, ticker=ticker
        )
        if not document:
            raise DocumentNotFoundError("Không tìm thấy Full Analysis item")
        if (
            document.get("content_kind") != FULL_REPORT_V1
            or document.get("status") != "ready"
        ):
            raise HTTPException(
                status_code=409, detail="Full Analysis chưa được promote/ready"
            )
        return full_report_response(
            document,
            managed_runtime.storage.get_full_report_payload(str(document["id"])),
        )

    @app.get(
        "/internal/v1/digest-batches/{batch_id}",
        dependencies=[Depends(require_internal_bearer)],
    )
    async def get_digest_batch(
        batch_id: str,
        page: int = Query(default=1, ge=1),
        page_size: int = Query(default=100, ge=1, le=500),
        status: str | None = Query(default=None),
        ticker: str | None = Query(default=None, max_length=20),
    ) -> dict[str, Any]:
        from .errors import DocumentNotFoundError

        batch_id = validate_batch_id(batch_id)
        result = managed_runtime.storage.digest_batch_status(batch_id)
        if not result:
            raise DocumentNotFoundError("Không tìm thấy digest batch")
        if status and status not in {"pending", "indexing", "ready", "failed"}:
            raise HTTPException(status_code=422, detail="status không hợp lệ")
        items = managed_runtime.storage.digest_batch_items(
            batch_id,
            page=page,
            page_size=page_size,
            status=status,
            ticker=ticker,
        )
        return {**result, **items}

    app.mount("/static", StaticFiles(directory=configured_settings.static_dir), name="static")

    @app.get("/", include_in_schema=False)
    async def index_page() -> FileResponse:
        return FileResponse(configured_settings.static_dir / "index.html")

    return app


app = create_app()
