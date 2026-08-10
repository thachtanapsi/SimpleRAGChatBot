"""Upload validation, PDF parsing, parent-child chunking và worker tuần tự."""

from __future__ import annotations

import asyncio
import hashlib
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Awaitable, Callable, Sequence

from fastapi import UploadFile
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

from .config import Settings
from .errors import DuplicateDocumentError, ValidationError
from .storage import SQLiteStorage
from .vector_index import ChromaIndex


SEPARATORS = ["\n#{1,6} ", "```\n", "\n\n", "\n", " ", ""]
ALLOWED_PDF_MIME = {"application/pdf", "application/x-pdf", "application/octet-stream"}


def deterministic_id(*parts: object, prefix: str) -> str:
    raw = "\x1f".join(str(part) for part in parts).encode("utf-8")
    return f"{prefix}_{hashlib.sha256(raw).hexdigest()}"


def sha256_file(path: Path, block_size: int = 1024 * 1024) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        while block := stream.read(block_size):
            digest.update(block)
            size += len(block)
    return digest.hexdigest(), size


def safe_pdf_name(filename: str | None) -> str:
    if not filename:
        raise ValidationError("Thiếu tên file")
    normalised = filename.replace("\\", "/")
    safe = Path(normalised).name
    if safe != normalised or safe in {"", ".", ".."}:
        raise ValidationError("Tên file không an toàn")
    if len(safe) > 200 or any(ord(char) < 32 for char in safe):
        raise ValidationError("Tên file không hợp lệ")
    if Path(safe).suffix.lower() != ".pdf":
        raise ValidationError("Chỉ hỗ trợ file có đuôi .pdf")
    return safe


def inspect_pdf(path: Path, settings: Settings) -> int:
    with path.open("rb") as stream:
        if stream.read(5) != b"%PDF-":
            raise ValidationError("Chữ ký file không phải PDF")
    try:
        reader = PdfReader(str(path), strict=False)
        if reader.is_encrypted:
            raise ValidationError("Chưa hỗ trợ PDF được mã hóa")
        page_count = len(reader.pages)
        if page_count == 0:
            raise ValidationError("PDF không có trang")
        if page_count > settings.max_pdf_pages:
            raise ValidationError(f"PDF vượt quá {settings.max_pdf_pages} trang")
        # Truy cập trang cuối để phát hiện sớm xref/page tree bị hỏng.
        _ = reader.pages[page_count - 1]
        return page_count
    except ValidationError:
        raise
    except Exception as exc:
        raise ValidationError("PDF bị hỏng hoặc không đọc được") from exc


def _splitter(size: int, overlap: int) -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=size,
        chunk_overlap=overlap,
        add_start_index=True,
        strip_whitespace=True,
        separators=SEPARATORS,
        is_separator_regex=True,
    )


def build_parent_child_chunks(
    *, file_hash: str, filename: str, pages: Sequence[str], settings: Settings
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str], list[dict[str, Any]]]:
    """Chunk từng trang riêng, nên parent/child không thể vượt ranh giới trang."""

    parent_splitter = _splitter(settings.parent_chunk_size, settings.parent_chunk_overlap)
    child_splitter = _splitter(settings.child_chunk_size, settings.child_chunk_overlap)
    document_id = f"doc_{file_hash}"
    parents: list[dict[str, Any]] = []
    children: list[dict[str, Any]] = []
    child_texts: list[str] = []
    child_metadatas: list[dict[str, Any]] = []

    for page_number, page_text in enumerate(pages, start=1):
        page_document = Document(page_content=page_text, metadata={"page": page_number})
        for parent_doc in parent_splitter.split_documents([page_document]):
            parent_start = int(parent_doc.metadata.get("start_index", 0))
            parent_id = deterministic_id(
                file_hash, page_number, parent_start, prefix="parent"
            )
            parents.append(
                {
                    "id": parent_id,
                    "document_id": document_id,
                    "page": page_number,
                    "start_index": parent_start,
                    "text": parent_doc.page_content,
                }
            )
            for child_doc in child_splitter.split_documents([parent_doc]):
                child_local_start = int(child_doc.metadata.get("start_index", 0))
                child_start = parent_start + child_local_start
                child_id = deterministic_id(
                    file_hash,
                    page_number,
                    parent_start,
                    child_local_start,
                    prefix="child",
                )
                children.append(
                    {
                        "id": child_id,
                        "parent_id": parent_id,
                        "document_id": document_id,
                        "page": page_number,
                        "start_index": child_start,
                    }
                )
                child_texts.append(child_doc.page_content)
                child_metadatas.append(
                    {
                        "child_id": child_id,
                        "parent_id": parent_id,
                        "document_id": document_id,
                        "filename": filename,
                        "page": page_number,
                        "start_index": child_start,
                    }
                )
    return parents, children, child_texts, child_metadatas


class IngestionService:
    def __init__(
        self,
        settings: Settings,
        storage: SQLiteStorage,
        index: ChromaIndex,
        logger: Any,
    ):
        self.settings = settings
        self.storage = storage
        self.index = index
        self.logger = logger
        self._upload_lock = asyncio.Lock()
        self._wake_worker: Callable[[], None] = lambda: None

    def set_worker_notifier(self, notifier: Callable[[], None]) -> None:
        self._wake_worker = notifier

    async def accept_upload(self, upload: UploadFile) -> dict[str, Any]:
        filename = safe_pdf_name(upload.filename)
        content_type = (upload.content_type or "application/octet-stream").lower()
        if content_type not in ALLOWED_PDF_MIME:
            raise ValidationError("MIME type không phải PDF")

        fd, temp_name = tempfile.mkstemp(prefix="upload_", suffix=".pdf", dir=self.settings.uploads_dir)
        os.close(fd)
        temp_path = Path(temp_name)
        digest = hashlib.sha256()
        total = 0
        try:
            with temp_path.open("wb") as stream:
                while chunk := await upload.read(1024 * 1024):
                    total += len(chunk)
                    if total > self.settings.max_upload_bytes:
                        raise ValidationError("PDF vượt quá giới hạn 50 MB")
                    digest.update(chunk)
                    stream.write(chunk)
            if total == 0:
                raise ValidationError("File upload rỗng")
            page_count = await asyncio.to_thread(inspect_pdf, temp_path, self.settings)
            file_hash = digest.hexdigest()
            document_id = f"doc_{file_hash}"

            async with self._upload_lock:
                duplicate = self.storage.get_document_by_hash(file_hash)
                if duplicate:
                    if (
                        duplicate["status"] == "failed"
                        and duplicate.get("error") != "deleting"
                        and Path(duplicate["stored_path"]).is_file()
                    ):
                        self.storage.requeue_document(duplicate["id"])
                        self._wake_worker()
                        return self.storage.get_document(duplicate["id"])
                    raise DuplicateDocumentError(duplicate["id"])
                destination = self.settings.uploads_dir / f"{document_id}.pdf"
                if destination.exists():
                    raise DuplicateDocumentError(document_id)
                os.replace(temp_path, destination)
                try:
                    document = self.storage.create_pending_document(
                        document_id=document_id,
                        filename=filename,
                        sha256=file_hash,
                        stored_path=destination,
                        size_bytes=total,
                        page_count=page_count,
                    )
                except Exception:
                    destination.unlink(missing_ok=True)
                    raise
            self._wake_worker()
            self.logger.info("document queued", extra={"event": "document_queued", "document_id": document_id})
            return document
        finally:
            temp_path.unlink(missing_ok=True)
            await upload.close()

    def seed_papers_once(self) -> int:
        if self.storage.get_meta("papers_seeded") == "1":
            return 0
        imported = 0
        if self.settings.papers_dir.exists():
            for source in sorted(self.settings.papers_dir.rglob("*.pdf")):
                try:
                    file_hash, size = sha256_file(source)
                    if size > self.settings.max_upload_bytes:
                        continue
                    if self.storage.get_document_by_hash(file_hash):
                        continue
                    page_count = inspect_pdf(source, self.settings)
                    document_id = f"doc_{file_hash}"
                    destination = self.settings.uploads_dir / f"{document_id}.pdf"
                    if not destination.exists():
                        temporary = destination.with_suffix(".copying")
                        shutil.copyfile(source, temporary)
                        os.replace(temporary, destination)
                    self.storage.create_pending_document(
                        document_id=document_id,
                        filename=safe_pdf_name(source.name),
                        sha256=file_hash,
                        stored_path=destination,
                        size_bytes=size,
                        page_count=page_count,
                    )
                    imported += 1
                except Exception as exc:
                    self.logger.error(
                        "seed failed",
                        extra={"event": "seed_failed", "error": type(exc).__name__},
                    )
        self.storage.set_meta("papers_seeded", "1")
        if imported:
            self._wake_worker()
        return imported

    def process_document(self, document: dict[str, Any]) -> None:
        document_id = document["id"]
        child_ids: list[str] = []
        try:
            reader = PdfReader(document["stored_path"], strict=False)
            pages = [(page.extract_text() or "").strip() for page in reader.pages]
            if sum(len(text) for text in pages) < self.settings.min_pdf_text_chars:
                raise ValidationError("PDF scan/ảnh chưa được hỗ trợ; cần OCR trước")
            parents, children, child_texts, metadata = build_parent_child_chunks(
                file_hash=document["sha256"],
                filename=document["filename"],
                pages=pages,
                settings=self.settings,
            )
            if not children:
                raise ValidationError("PDF không có nội dung text có thể index")
            child_ids = [child["id"] for child in children]
            self.index.add_children(ids=child_ids, texts=child_texts, metadatas=metadata)
            try:
                self.storage.finish_indexing(
                    document_id=document_id,
                    page_count=len(pages),
                    parents=parents,
                    children=children,
                    fingerprint=self.settings.index_fingerprint,
                    collection_name=self.settings.collection_name,
                )
            except Exception:
                self.index.delete(child_ids)
                raise
            self.logger.info(
                "document ready",
                extra={
                    "event": "document_ready",
                    "document_id": document_id,
                    # Giữ log nhỏ nhưng vẫn đủ truy vết một batch.
                    "chunk_ids": child_ids[:20],
                },
            )
        except Exception as exc:
            if child_ids:
                self.index.delete(child_ids)
            public_error = (
                str(exc) if isinstance(exc, ValidationError) else type(exc).__name__
            )
            self.storage.mark_failed(document_id, public_error)
            self.logger.error(
                "index failed",
                extra={"event": "index_failed", "document_id": document_id, "error": type(exc).__name__},
            )

    def process_next(self) -> bool:
        document = self.storage.claim_next_pending()
        if not document:
            return False
        self.process_document(document)
        return True


class IngestionWorker:
    """Một worker trong process; SQLite claim bảo vệ không xử lý trùng."""

    def __init__(self, ingestion: IngestionService):
        self.ingestion = ingestion
        self._event = asyncio.Event()
        self._task: asyncio.Task[None] | None = None
        self.ingestion.set_worker_notifier(self.wake)

    def wake(self) -> None:
        self._event.set()

    def start(self) -> None:
        if self._task is None:
            self._task = asyncio.create_task(self._run(), name="rag-ingestion-worker")
            self.wake()

    async def stop(self) -> None:
        if self._task is None:
            return
        self._task.cancel()
        try:
            await self._task
        except asyncio.CancelledError:
            pass
        self._task = None

    async def _run(self) -> None:
        while True:
            processed = await asyncio.to_thread(self.ingestion.process_next)
            if processed:
                continue
            self._event.clear()
            try:
                await asyncio.wait_for(self._event.wait(), timeout=1.0)
            except TimeoutError:
                pass
