"""FastAPI REST/SSE server và static web UI chạy trên localhost."""

from __future__ import annotations

import json
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Sequence

from fastapi import FastAPI, File, Request, Response, UploadFile
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .config import Settings
from .errors import DuplicateDocumentError, RAGError
from .runtime import Runtime


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    session_id: str | None = None
    document_ids: list[str] | None = None


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
        )
    }


def sse(event: str, data: dict[str, Any]) -> str:
    encoded = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    return f"event: {event}\ndata: {encoded}\n\n"


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
    async def list_documents() -> list[dict[str, Any]]:
        return [public_document(item) for item in managed_runtime.storage.list_documents()]

    @app.post("/api/documents", status_code=202)
    async def upload_document(file: UploadFile = File(...)) -> dict[str, Any]:
        document = await managed_runtime.ingestion.accept_upload(file)
        return {"document_id": document["id"], "status": document["status"]}

    @app.get("/api/documents/{document_id}")
    async def get_document(document_id: str) -> dict[str, Any]:
        from .errors import DocumentNotFoundError

        document = managed_runtime.storage.get_document(document_id)
        if not document:
            raise DocumentNotFoundError("Không tìm thấy tài liệu")
        return public_document(document)

    @app.delete("/api/documents/{document_id}", status_code=204)
    async def delete_document(document_id: str) -> Response:
        await managed_runtime.delete_document(document_id)
        return Response(status_code=204)

    @app.post("/api/chat")
    async def chat(payload: ChatRequest) -> dict[str, Any]:
        return await managed_runtime.chat.answer(
            payload.question, payload.session_id, payload.document_ids
        )

    @app.post("/api/chat/stream")
    async def chat_stream(payload: ChatRequest, request: Request) -> StreamingResponse:
        async def event_source() -> AsyncIterator[str]:
            generator = managed_runtime.chat.stream(
                payload.question, payload.session_id, payload.document_ids
            )
            try:
                async for item in generator:
                    if await request.is_disconnected():
                        await generator.aclose()
                        return
                    yield sse(item["event"], item["data"])
            except RAGError as exc:
                yield sse("error", {"detail": str(exc)})
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

    app.mount("/static", StaticFiles(directory=configured_settings.static_dir), name="static")

    @app.get("/", include_in_schema=False)
    async def index_page() -> FileResponse:
        return FileResponse(configured_settings.static_dir / "index.html")

    return app


app = create_app()

