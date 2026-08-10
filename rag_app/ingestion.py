"""Upload validation, PDF parsing, parent-child chunking và worker tuần tự."""

from __future__ import annotations

import asyncio
import hashlib
import os
import re
import shutil
import tempfile
from dataclasses import dataclass
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
TABLE_TAIL_LINES = 60
TABLE_HEAD_LINES = 20


@dataclass(frozen=True, slots=True)
class CrossPageBoundary:
    """Một ranh giới trang có bằng chứng nội dung đang tiếp diễn."""

    page: int
    page_end: int
    boundary_type: str
    header_cells: tuple[str, ...] = ()


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


def _normalise_line(line: str) -> str:
    return re.sub(r"\s+", " ", line).strip().casefold()


def _nonempty_lines(text: str) -> list[str]:
    return [line.strip() for line in text.splitlines() if line.strip()]


def _is_page_number(line: str) -> bool:
    return bool(re.fullmatch(r"(?:page\s+)?\d+(?:\s*/\s*\d+)?", line, re.I))


def _layout_cells(line: str) -> tuple[str, ...]:
    cells = tuple(
        _normalise_line(cell)
        for cell in re.split(r"\s{3,}", line.strip())
        if cell.strip()
    )
    if not 3 <= len(cells) <= 8:
        return ()
    if not all(any(character.isalpha() for character in cell) for cell in cells):
        return ()
    if sum(len(cell) for cell in cells) > 240:
        return ()
    return cells


def _table_header_candidates(lines: Sequence[str]) -> list[tuple[str, ...]]:
    return [cells for line in lines if (cells := _layout_cells(line))]


def _strip_pair_page_noise(
    left_text: str,
    right_text: str,
    *,
    table_header: tuple[str, ...] = (),
) -> tuple[list[str], list[str]]:
    """Bỏ page number/running header chỉ ở đầu trang, không sửa text gốc."""

    left = _nonempty_lines(left_text)
    right = _nonempty_lines(right_text)
    while left and _is_page_number(left[0]):
        left.pop(0)
    while right and _is_page_number(right[0]):
        right.pop(0)

    header_text = _normalise_line(" ".join(table_header)) if table_header else ""
    if left and right:
        repeated = _normalise_line(left[0])
        if (
            repeated == _normalise_line(right[0])
            and repeated != header_text
            and len(repeated) <= 160
        ):
            left.pop(0)
            right.pop(0)

    while left and _is_page_number(left[0]):
        left.pop(0)
    while right and _is_page_number(right[0]):
        right.pop(0)
    return left, right


def _starts_with_lowercase(text: str) -> bool:
    for character in text:
        if character.isalpha():
            return character.islower()
    return False


def _is_caption(text: str) -> bool:
    return bool(
        re.match(
            r"^(?:table|figure|fig\.?|bảng|hình)\s+\d+\b",
            text.strip(),
            flags=re.I,
        )
    )


def detect_cross_page_boundaries(
    pages: Sequence[str], layout_pages: Sequence[str] | None = None
) -> list[CrossPageBoundary]:
    """Phát hiện tiếp nối bảng/đoạn văn giữa các cặp trang liền kề.

    Layout text chỉ dùng để nhận diện cột; nội dung được index luôn lấy từ text
    thường để không phụ thuộc khoảng trắng căn lề của PDF.
    """

    layouts = list(layout_pages) if layout_pages is not None else [""] * len(pages)
    if len(layouts) != len(pages):
        raise ValueError("Số trang layout không khớp số trang text")

    boundaries: list[CrossPageBoundary] = []
    for index in range(len(pages) - 1):
        left_candidates = _table_header_candidates(
            layouts[index].splitlines()[-TABLE_TAIL_LINES:]
        )
        right_candidates = _table_header_candidates(
            layouts[index + 1].splitlines()[:TABLE_HEAD_LINES]
        )
        right_set = set(right_candidates)
        repeated_header = next(
            (cells for cells in reversed(left_candidates) if cells in right_set),
            (),
        )
        if repeated_header:
            boundaries.append(
                CrossPageBoundary(
                    page=index + 1,
                    page_end=index + 2,
                    boundary_type="table",
                    header_cells=repeated_header,
                )
            )
            continue

        left_lines, right_lines = _strip_pair_page_noise(
            pages[index], pages[index + 1]
        )
        if not left_lines or not right_lines:
            continue
        previous = left_lines[-1]
        following = right_lines[0]
        unfinished = previous.endswith("-") or previous[-1:] not in ".?!:"
        if unfinished and _starts_with_lowercase(following) and not _is_caption(following):
            boundaries.append(
                CrossPageBoundary(
                    page=index + 1,
                    page_end=index + 2,
                    boundary_type="prose",
                )
            )
    return boundaries


def _remove_repeated_table_header(
    lines: list[str], header_cells: tuple[str, ...]
) -> list[str]:
    if not header_cells:
        return lines
    expected = _normalise_line(" ".join(header_cells))
    for index, line in enumerate(lines[:10]):
        if _normalise_line(line) == expected:
            return [*lines[:index], *lines[index + 1 :]]
    return lines


def _bridge_text(
    left: str,
    right: str,
    *,
    page: int,
    page_end: int,
    max_chars: int,
    side_limit: int,
) -> tuple[str, int]:
    """Ghép hai phía trong giới hạn và trả offset bắt đầu ở trang trái."""

    marker = f"\n[PAGE {page}->{page_end}]\n"
    available = max_chars - len(marker)
    if available < 2:
        combined = (left + "\n" + right).strip()[:max_chars]
        return combined, max(0, len(left) - len(combined))

    half = available // 2
    left_budget = min(side_limit, half)
    right_budget = min(side_limit, available - left_budget)
    left_part = left[-left_budget:].lstrip() if left_budget else ""
    right_part = right[:right_budget].rstrip() if right_budget else ""
    text = f"{left_part}{marker}{right_part}".strip()
    return text, max(0, len(left) - len(left_part))


def _boundary_source_texts(
    left_text: str, right_text: str, boundary: CrossPageBoundary
) -> tuple[str, str]:
    left_lines, right_lines = _strip_pair_page_noise(
        left_text,
        right_text,
        table_header=boundary.header_cells,
    )
    if boundary.boundary_type == "table":
        right_lines = _remove_repeated_table_header(
            right_lines, boundary.header_cells
        )
    return "\n".join(left_lines), "\n".join(right_lines)


def build_parent_child_chunks(
    *,
    file_hash: str,
    filename: str,
    pages: Sequence[str],
    settings: Settings,
    layout_pages: Sequence[str] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str], list[dict[str, Any]]]:
    """Tạo chunk theo trang và bridge có chọn lọc cho nội dung bị ngắt trang."""

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
                    "page_end": page_number,
                    "kind": "page",
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
                        "page_end": page_number,
                        "kind": "page",
                        "start_index": child_start,
                        "text": child_doc.page_content,
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
                        "page_end": page_number,
                        "kind": "page",
                        "start_index": child_start,
                    }
                )

    if settings.cross_page_enabled:
        for boundary in detect_cross_page_boundaries(pages, layout_pages):
            left, right = _boundary_source_texts(
                pages[boundary.page - 1], pages[boundary.page_end - 1], boundary
            )
            parent_text, parent_start = _bridge_text(
                left,
                right,
                page=boundary.page,
                page_end=boundary.page_end,
                max_chars=settings.parent_chunk_size,
                side_limit=settings.cross_page_context_chars,
            )
            child_text, child_start = _bridge_text(
                left,
                right,
                page=boundary.page,
                page_end=boundary.page_end,
                max_chars=settings.child_chunk_size,
                side_limit=settings.child_chunk_size,
            )
            if not parent_text or not child_text:
                continue
            parent_id = deterministic_id(
                file_hash,
                boundary.page,
                boundary.page_end,
                boundary.boundary_type,
                parent_start,
                prefix="parent_bridge",
            )
            parents.append(
                {
                    "id": parent_id,
                    "document_id": document_id,
                    "page": boundary.page,
                    "page_end": boundary.page_end,
                    "kind": "bridge",
                    "start_index": parent_start,
                    "text": parent_text,
                }
            )

            # Child boundary bảo đảm luôn có một embedding nhìn thấy cả hai
            # phía. Các child context phủ toàn bộ bridge parent để một cụm nằm
            # trước ranh giới (ví dụ tiêu đề/đoạn dẫn của bảng) vẫn truy xuất
            # được parent chứa phần tiếp nối ở trang sau.
            bridge_child_specs: list[tuple[str, str, int, int]] = [
                ("boundary", child_text, child_start, 0)
            ]
            seen_child_texts = {child_text}
            bridge_document = Document(page_content=parent_text)
            for context_doc in child_splitter.split_documents([bridge_document]):
                context_text = context_doc.page_content
                if context_text in seen_child_texts:
                    continue
                local_start = int(context_doc.metadata.get("start_index", 0))
                bridge_child_specs.append(
                    (
                        "context",
                        context_text,
                        parent_start + local_start,
                        local_start,
                    )
                )
                seen_child_texts.add(context_text)

            for (
                variant,
                bridge_child_text,
                bridge_child_start,
                local_start,
            ) in bridge_child_specs:
                child_id = deterministic_id(
                    file_hash,
                    boundary.page,
                    boundary.page_end,
                    boundary.boundary_type,
                    parent_start,
                    variant,
                    local_start,
                    prefix="child_bridge",
                )
                child = {
                    "id": child_id,
                    "parent_id": parent_id,
                    "document_id": document_id,
                    "page": boundary.page,
                    "page_end": boundary.page_end,
                    "kind": "bridge",
                    "start_index": bridge_child_start,
                    "text": bridge_child_text,
                }
                children.append(child)
                child_texts.append(bridge_child_text)
                child_metadatas.append(
                    {
                        "child_id": child_id,
                        "parent_id": parent_id,
                        "document_id": document_id,
                        "filename": filename,
                        "page": boundary.page,
                        "page_end": boundary.page_end,
                        "kind": "bridge",
                        "bridge_variant": variant,
                        "start_index": bridge_child_start,
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
        *,
        index_fingerprint: str | None = None,
        collection_name: str | None = None,
    ):
        self.settings = settings
        self.storage = storage
        self.index = index
        self.logger = logger
        self.index_fingerprint = index_fingerprint or settings.index_fingerprint
        self.collection_name = collection_name or settings.collection_name
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
            layout_pages: list[str] = []
            for page, fallback_text in zip(reader.pages, pages):
                try:
                    layout_pages.append(
                        (page.extract_text(extraction_mode="layout") or "").strip()
                    )
                except Exception:
                    # Layout chỉ hỗ trợ heuristic bảng; text thường vẫn đủ để
                    # index và phát hiện đoạn văn tiếp nối.
                    layout_pages.append(fallback_text)
            parents, children, child_texts, metadata = build_parent_child_chunks(
                file_hash=document["sha256"],
                filename=document["filename"],
                pages=pages,
                settings=self.settings,
                layout_pages=layout_pages,
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
                    fingerprint=self.index_fingerprint,
                    collection_name=self.collection_name,
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
                    "cross_page_bridges": sum(
                        parent.get("kind") == "bridge" for parent in parents
                    ),
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
