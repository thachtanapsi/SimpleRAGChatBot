from __future__ import annotations

from fastapi.testclient import TestClient

from rag_app.api import create_app
from rag_app.errors import GenerationTruncatedError


class FakeLogger:
    def info(self, *_args, **_kwargs): pass
    def error(self, *_args, **_kwargs): pass


class FakeStorage:
    def __init__(self):
        self.documents = {
            "doc1": {
                "id": "doc1",
                "filename": "paper.pdf",
                "size_bytes": 100,
                "page_count": 1,
                "status": "ready",
                "error": None,
                "created_at": "now",
                "updated_at": "now",
            }
        }

    def list_documents(self): return list(self.documents.values())
    def get_document(self, document_id): return self.documents.get(document_id)
    def all_messages(self, session_id):
        return [] if session_id == "session" else None
    def delete_session(self, session_id): return session_id == "session"
    def lock_document_for_deletion(self, document_id): return self.documents.get(document_id)


class FakeIngestion:
    async def accept_upload(self, upload):
        assert await upload.read() == b"%PDF-test"
        await upload.seek(0)
        return {"id": "new-doc", "status": "pending"}


class FakeChat:
    citation = {
        "id": "1",
        "filename": "paper.pdf",
        "page": 3,
        "page_end": 4,
        "cited": True,
    }

    async def answer(self, question, session_id, document_ids):
        return {
            "session_id": session_id or "new-session",
            "answer": "answer [1]",
            "citations": [self.citation],
        }

    async def stream(self, question, session_id, document_ids):
        yield {"event": "meta", "data": {"session_id": session_id or "new-session"}}
        yield {"event": "token", "data": {"text": "answer"}}
        yield {"event": "sources", "data": {"citations": [self.citation]}}
        yield {"event": "done", "data": {"answer": "answer"}}


class TruncatedFakeChat(FakeChat):
    async def answer(self, question, session_id, document_ids):
        raise GenerationTruncatedError(2048)

    async def stream(self, question, session_id, document_ids):
        yield {"event": "meta", "data": {"session_id": session_id or "new-session"}}
        yield {"event": "token", "data": {"text": "incomplete"}}
        raise GenerationTruncatedError(2048)


class FakeRuntime:
    def __init__(self):
        self.storage = FakeStorage()
        self.ingestion = FakeIngestion()
        self.chat = FakeChat()
        self.logger = FakeLogger()
        self.deleted = []

    async def start(self): pass
    async def stop(self): pass
    async def health(self): return {"status": "ok", "sqlite": True, "chroma": True, "ollama": True}
    async def delete_document(self, document_id):
        self.deleted.append(document_id)
        self.storage.documents.pop(document_id, None)


def test_api_upload_status_delete_chat_sse_and_health(settings):
    runtime = FakeRuntime()
    app = create_app(settings, runtime)
    with TestClient(app) as client:
        assert client.get("/api/health").json()["status"] == "ok"
        assert client.get("/api/documents").json()[0]["id"] == "doc1"
        assert client.get("/api/documents/doc1").status_code == 200

        uploaded = client.post(
            "/api/documents",
            files={"file": ("sample.pdf", b"%PDF-test", "application/pdf")},
        )
        assert uploaded.status_code == 202
        assert uploaded.json() == {"document_id": "new-doc", "status": "pending"}

        answer = client.post("/api/chat", json={"question": "Question"})
        assert answer.json()["session_id"] == "new-session"
        assert answer.json()["citations"][0]["page_end"] == 4

        stream = client.post("/api/chat/stream", json={"question": "Question"})
        assert "event: meta" in stream.text
        assert "event: token" in stream.text
        assert "event: sources" in stream.text
        assert '"page_end":4' in stream.text
        assert "event: done" in stream.text

        assert client.get("/api/sessions/session/messages").status_code == 200
        assert client.delete("/api/sessions/session").status_code == 204
        assert client.delete("/api/documents/doc1").status_code == 204
        assert runtime.deleted == ["doc1"]


def test_static_ui_has_no_cdn(settings):
    app = create_app(settings, FakeRuntime())
    with TestClient(app) as client:
        page = client.get("/")
        assert page.status_code == 200
        assert "cdn" not in page.text.lower()
        assert "/static/markdown.js" in page.text
        assert "/static/app.js" in page.text
        renderer = client.get("/static/markdown.js")
        assert renderer.status_code == 200
        assert "safeUrl" in renderer.text
        script = client.get("/static/app.js")
        assert "source.page_end" in script.text
        assert "renderAssistantAnswer" in script.text


def test_truncated_generation_has_machine_readable_error(settings):
    runtime = FakeRuntime()
    runtime.chat = TruncatedFakeChat()
    app = create_app(settings, runtime)

    with TestClient(app) as client:
        answer = client.post("/api/chat", json={"question": "Question"})
        assert answer.status_code == 503
        assert answer.json()["code"] == "response_truncated"

        stream = client.post("/api/chat/stream", json={"question": "Question"})
        assert "event: token" in stream.text
        assert "event: error" in stream.text
        assert '"code":"response_truncated"' in stream.text
        assert "event: done" not in stream.text
