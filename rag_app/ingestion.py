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
from .digests import DIGEST_CONTENT_KIND, build_digest_chunks, canonicalise_digest
from .errors import (
    ConflictError,
    DocumentBusyError,
    DuplicateDocumentError,
    PromotionConflictError,
    ValidationError,
)
from .full_reports import FULL_REPORT_V1, canonicalise_full_report
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

    async def put_digest_batch(
        self,
        *,
        batch_id: str,
        source: str,
        analysis_date: str,
        cutoff: str,
        target_count: int,
        research_mode: str,
        data_provenance: str,
    ) -> tuple[str, dict[str, Any]]:
        """Serialize batch provenance changes with document acceptance/sealing."""

        async with self._upload_lock:
            action, batch = self.storage.create_digest_batch(
                batch_id=batch_id,
                source=source,
                analysis_date=analysis_date,
                cutoff=cutoff,
                target_count=target_count,
                research_mode=research_mode,
                data_provenance=data_provenance,
            )
        if action in {"upgraded", "unchanged"}:
            # Reconciliation intentionally runs after releasing the mutation
            # lock. SQLite already owns the consistent pair and durable outbox;
            # POST may proceed using that pair while Chroma updates in place.
            await asyncio.to_thread(
                self.reconcile_vector_metadata_updates,
                batch_id=batch_id,
            )
        return action, batch

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

    async def accept_digest_documents(
        self,
        *,
        batch_id: str,
        batch_source: str,
        payloads: Sequence[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Queue a bounded batch of canonical digest documents idempotently."""

        results: list[dict[str, Any]] = []
        async with self._upload_lock:
            # PUT provenance changes and sealing use this same lock. Resolve
            # the batch and canonical hashes only after acquiring it so every
            # validation observes one immutable pair/status snapshot.
            batch = self.storage.get_digest_batch(batch_id)
            if not batch:
                from .errors import DocumentNotFoundError

                raise DocumentNotFoundError("Không tìm thấy digest batch")
            if batch["source"] != batch_source:
                raise ValidationError("source không khớp digest batch")
            canonical = [
                canonicalise_digest(
                    payload,
                    batch_source=batch_source,
                    batch_research_mode=str(batch["research_mode"]),
                    batch_data_provenance=str(batch["data_provenance"]),
                )
                for payload in payloads
            ]
            seen_keys: set[str] = set()
            seen_ranks: set[int] = set()
            for digest in canonical:
                if digest.source_key in seen_keys:
                    raise ValidationError(
                        "Một request không được chứa digest trùng identity"
                    )
                seen_keys.add(digest.source_key)
                if digest.liquidity_rank is None:
                    raise ValidationError(
                        "liquidity_rank là bắt buộc với daily digest"
                    )
                if digest.liquidity_rank > int(batch["target_count"]):
                    raise ValidationError(
                        "liquidity_rank phải nằm trong khoảng 1..target_count"
                    )
                if digest.liquidity_rank in seen_ranks:
                    raise ValidationError(
                        "liquidity_rank bị trùng trong request"
                    )
                seen_ranks.add(digest.liquidity_rank)
                if (
                    digest.analysis_date != batch["analysis_date"]
                    or digest.cutoff != batch["cutoff"]
                ):
                    raise ValidationError(
                        "Ngày/cutoff của digest không khớp batch"
                    )
            current_status = self.storage.digest_batch_status(batch_id)
            assert current_status is not None
            existing_by_key = {
                digest.source_key: self.storage.get_document_by_source_key(
                    digest.source_key
                )
                for digest in canonical
            }
            if batch["status"] == "sealed":
                return self._retry_sealed_digest_documents(
                    batch_id=batch_id,
                    digests=canonical,
                    existing_by_key=existing_by_key,
                )
            new_count = sum(item is None for item in existing_by_key.values())
            if current_status["accepted_count"] + new_count > int(batch["target_count"]):
                raise ConflictError("Số digest vượt target_count của batch")

            # Validate the entire request before mutating the first document.
            for digest in canonical:
                existing = existing_by_key[digest.source_key]
                if existing:
                    if existing.get("batch_id") != batch_id:
                        raise ConflictError("Digest identity đã thuộc batch khác")
                    if digest.source_updated_at < str(existing["source_updated_at"]):
                        continue
                    rank_owner = self.storage.get_digest_batch_member(
                        batch_id, liquidity_rank=digest.liquidity_rank
                    )
                    if rank_owner and rank_owner["id"] != existing["id"]:
                        raise ConflictError("liquidity_rank bị trùng trong batch")
                    self._validate_daily_hashes(existing, digest)
                    if existing.get("content_sha256") != digest.content_sha256:
                        if digest.source_updated_at == str(existing["source_updated_at"]):
                            raise DocumentBusyError(
                                "Digest cùng source_updated_at nhưng nội dung khác nhau"
                            )
                        if existing["status"] == "indexing":
                            raise DocumentBusyError("Digest đang indexing; hãy retry sau")
                else:
                    duplicate_ticker = self.storage.get_digest_batch_member(
                        batch_id, ticker=digest.ticker
                    )
                    duplicate_rank = self.storage.get_digest_batch_member(
                        batch_id, liquidity_rank=digest.liquidity_rank
                    )
                    if duplicate_ticker or duplicate_rank:
                        raise ConflictError("ticker hoặc liquidity_rank bị trùng trong batch")

            for digest in canonical:
                existing = existing_by_key[digest.source_key]
                if existing:
                    if existing.get("batch_id") != batch_id:
                        raise ConflictError("Digest identity đã thuộc batch khác")
                    rank_owner = self.storage.get_digest_batch_member(
                        batch_id, liquidity_rank=digest.liquidity_rank
                    )
                    if rank_owner and rank_owner["id"] != existing["id"]:
                        raise ConflictError("liquidity_rank bị trùng trong batch")
                    if digest.source_updated_at < str(existing["source_updated_at"]):
                        results.append(
                            {
                                "ticker": digest.ticker,
                                "document_id": existing["id"],
                                "action": "stale",
                                "status": existing["status"],
                            }
                        )
                        continue
                    if existing.get("content_sha256") == digest.content_sha256:
                        needs_hash_backfill = (
                            (not existing.get("identity_hash") and digest.identity_hash)
                            or (not existing.get("digest_hash") and digest.digest_hash)
                        )
                        if (
                            digest.source_updated_at
                            > str(existing["source_updated_at"])
                            or needs_hash_backfill
                        ):
                            existing = self.storage.advance_digest_source(
                                existing["id"],
                                batch_id=batch_id,
                                expected_content_sha256=str(
                                    existing["content_sha256"]
                                ),
                                expected_source_updated_at=str(
                                    existing["source_updated_at"]
                                ),
                                allow_sealed=False,
                                source_updated_at=digest.source_updated_at,
                                source_run_id=digest.source_run_id,
                                identity_hash=digest.identity_hash,
                                digest_hash=digest.digest_hash,
                            )
                        if existing["status"] == "failed":
                            existing = self.storage.requeue_daily_document(
                                str(existing["id"]),
                                batch_id=batch_id,
                                expected_content_sha256=str(
                                    existing["content_sha256"]
                                ),
                                expected_source_updated_at=str(
                                    existing["source_updated_at"]
                                ),
                                allow_sealed=False,
                            )
                            status = str(existing["status"])
                            action = "retried" if status == "pending" else "unchanged"
                            if status == "pending":
                                self._wake_worker()
                        else:
                            action = "unchanged"
                            status = existing["status"]
                        results.append(
                            {
                                "ticker": digest.ticker,
                                "document_id": existing["id"],
                                "action": action,
                                "status": status,
                            }
                        )
                        continue
                    if digest.source_updated_at == str(existing["source_updated_at"]):
                        raise DocumentBusyError(
                            "Digest cùng source_updated_at nhưng nội dung khác nhau"
                        )
                    if existing["status"] == "indexing":
                        raise DocumentBusyError("Digest đang indexing; hãy retry sau")
                    action = "updated"
                else:
                    duplicate_ticker = self.storage.get_digest_batch_member(
                        batch_id, ticker=digest.ticker
                    )
                    duplicate_rank = self.storage.get_digest_batch_member(
                        batch_id, liquidity_rank=digest.liquidity_rank
                    )
                    if duplicate_ticker or duplicate_rank:
                        raise ConflictError("ticker hoặc liquidity_rank bị trùng trong batch")
                    action = "created"
                document = self.storage.replace_digest_document(
                    digest=digest,
                    batch_id=batch_id,
                    expected_existing_content_sha256=(
                        str(existing["content_sha256"]) if existing else None
                    ),
                    expected_existing_source_updated_at=(
                        str(existing["source_updated_at"]) if existing else None
                    ),
                )
                results.append(
                    {
                        "ticker": digest.ticker,
                        "document_id": document["id"],
                        "action": action,
                        "status": document["status"],
                    }
                )
                self.logger.info(
                    "digest queued",
                    extra={
                        "event": "digest_queued",
                        "document_id": document["id"],
                        "ticker": digest.ticker,
                        "batch_id": batch_id,
                        "action": action,
                    },
                )
        if any(item["action"] in {"created", "updated", "retried"} for item in results):
            self._wake_worker()
        return results

    @staticmethod
    def _validate_daily_hashes(existing: dict[str, Any], digest: Any) -> None:
        """Never let an exact-text retry rewrite immutable promotion anchors."""

        if existing.get("content_kind") != DIGEST_CONTENT_KIND:
            raise ConflictError("Không thể ghi đè Full Analysis bằng daily digest")
        for field in ("identity_hash", "digest_hash"):
            stored = existing.get(field)
            incoming = getattr(digest, field)
            if stored and incoming and str(stored) != str(incoming):
                raise ConflictError(f"{field} không khớp digest hiện tại")

    async def accept_full_report_promotions(
        self,
        *,
        batch_id: str,
        payloads: Sequence[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Promote sealed daily documents to Full Analysis without changing identity.

        SQLite replacement and the old-vector deletion outbox commit together.
        Chroma is then replaced by the ordinary crash-recoverable writer.
        """

        results: list[dict[str, Any]] = []
        promoted = False
        async with self._upload_lock:
            batch = self.storage.get_digest_batch(batch_id)
            if not batch:
                from .errors import DocumentNotFoundError

                raise DocumentNotFoundError("Không tìm thấy digest batch")
            if batch["status"] != "sealed":
                raise ConflictError("Chỉ promote Full Analysis sau khi batch đã seal")
            reports = [canonicalise_full_report(payload) for payload in payloads]
            service_secret = self.settings.rag_service_api_key
            if service_secret and len(service_secret) >= 8 and any(
                service_secret in text
                for report in reports
                for text in report.sections.values()
            ):
                raise ValidationError(
                    "Full Analysis section chứa credential được cấu hình"
                )
            seen_keys: set[str] = set()
            seen_ranks: set[int] = set()
            for report in reports:
                if report.source_key in seen_keys:
                    raise ValidationError(
                        "Một request không được chứa Full Analysis trùng identity"
                    )
                seen_keys.add(report.source_key)
                if report.liquidity_rank in seen_ranks:
                    raise ValidationError("liquidity_rank bị trùng trong request")
                seen_ranks.add(report.liquidity_rank)
                if report.source != batch["source"]:
                    raise ValidationError("source không khớp digest batch")
                if (
                    report.analysis_date != batch["analysis_date"]
                    or report.cutoff != batch["cutoff"]
                ):
                    raise ValidationError(
                        "Ngày/cutoff của Full Analysis không khớp batch"
                    )
                if report.liquidity_rank > int(batch["target_count"]):
                    raise ValidationError(
                        "liquidity_rank phải nằm trong khoảng 1..target_count"
                    )
                if (
                    report.research_mode != batch["research_mode"]
                    or report.data_provenance != batch["data_provenance"]
                ):
                    raise ValidationError(
                        "Provenance của Full Analysis không khớp digest batch"
                    )

            for report in reports:
                existing = self.storage.get_document_by_source_key(report.source_key)
                base = {
                    "ticker": report.ticker,
                    "document_id": str(existing["id"]) if existing else report.document_id,
                    "full_report_hash": report.full_report_hash,
                }

                def reject(error_code: str) -> None:
                    results.append(
                        {
                            **base,
                            "action": "conflict",
                            "status": str(existing.get("status") if existing else "missing"),
                            "error_code": error_code,
                        }
                    )

                if not existing or existing.get("batch_id") != batch_id:
                    reject("parent_digest_not_found")
                    continue
                if existing.get("content_kind") == FULL_REPORT_V1:
                    exact = (
                        existing.get("full_report_hash") == report.full_report_hash
                        and existing.get("identity_hash") == report.identity_hash
                        and existing.get("parent_identity_hash")
                        == report.parent_identity_hash
                        and existing.get("expected_digest_hash")
                        == report.expected_digest_hash
                        and int(existing.get("execution_generation") or 0)
                        == report.execution_generation
                    )
                    if exact:
                        replay_status = str(existing["status"])
                        if existing["status"] == "failed":
                            existing = self.storage.requeue_full_report(
                                str(existing["id"]),
                                batch_id=batch_id,
                                digest=report,
                            )
                            replay_status = str(existing["status"])
                            promoted = replay_status == "pending"
                        elif existing["status"] == "pending":
                            # A replay after process crash must wake the writer
                            # even though the immutable Full Analysis is a no-op.
                            promoted = True
                        results.append(
                            {
                                **base,
                                "action": "unchanged",
                                "status": replay_status,
                            }
                        )
                    else:
                        reject("full_report_already_promoted")
                    continue
                if existing.get("content_kind") != DIGEST_CONTENT_KIND:
                    reject("parent_content_kind_invalid")
                    continue
                if not existing.get("identity_hash") or not existing.get("digest_hash"):
                    reject("parent_promotion_hash_missing")
                    continue
                if existing.get("identity_hash") != report.parent_identity_hash:
                    reject("parent_identity_hash_mismatch")
                    continue
                if existing.get("digest_hash") != report.expected_digest_hash:
                    reject("expected_digest_hash_mismatch")
                    continue
                if (
                    existing.get("ticker") != report.ticker
                    or existing.get("analysis_date") != report.analysis_date
                    or existing.get("analysis_cutoff") != report.cutoff
                    or int(existing.get("liquidity_rank") or 0)
                    != report.liquidity_rank
                    or existing.get("research_mode") != report.research_mode
                    or existing.get("data_provenance") != report.data_provenance
                ):
                    reject("parent_identity_mismatch")
                    continue
                if report.source_updated_at <= str(existing["source_updated_at"]):
                    results.append(
                        {
                            **base,
                            "action": "stale",
                            "status": str(existing["status"]),
                            "error_code": "source_updated_at_not_newer",
                        }
                    )
                    continue
                if existing["status"] == "indexing":
                    reject("parent_document_busy")
                    continue
                try:
                    document = self.storage.replace_digest_document(
                        digest=report,
                        batch_id=batch_id,
                        require_daily_parent=True,
                    )
                except DocumentBusyError:
                    existing = self.storage.get_document_by_source_key(
                        report.source_key
                    )
                    reject("parent_document_busy")
                    continue
                except PromotionConflictError as exc:
                    existing = self.storage.get_document_by_source_key(
                        report.source_key
                    )
                    if exc.error_code == "source_updated_at_not_newer":
                        results.append(
                            {
                                **base,
                                "action": "stale",
                                "status": str(
                                    existing.get("status") if existing else "missing"
                                ),
                                "error_code": exc.error_code,
                            }
                        )
                    else:
                        reject(exc.error_code)
                    continue
                storage_action = str(document.pop("_promotion_action", "promoted"))
                if storage_action == "unchanged" and document["status"] == "failed":
                    document = self.storage.requeue_full_report(
                        str(document["id"]),
                        batch_id=batch_id,
                        digest=report,
                    )
                if storage_action == "promoted" or document["status"] in {
                    "pending",
                }:
                    promoted = True
                results.append(
                    {
                        **base,
                        "document_id": str(document["id"]),
                        "action": storage_action,
                        "status": str(document["status"]),
                    }
                )
                self.logger.info(
                    "full analysis queued",
                    extra={
                        "event": "full_analysis_queued",
                        "document_id": document["id"],
                        "ticker": report.ticker,
                        "batch_id": batch_id,
                    },
                )
        if promoted:
            self._wake_worker()
        return results

    def _retry_sealed_digest_documents(
        self,
        *,
        batch_id: str,
        digests: Sequence[Any],
        existing_by_key: dict[str, dict[str, Any] | None],
    ) -> list[dict[str, Any]]:
        """Allow an exact failed document to recover after an immutable seal.

        Sealing closes membership and content changes, but indexing can fail
        later.  Without this narrow retry path a transient embedding/Chroma
        outage would leave a sealed batch permanently unrecoverable.
        """

        for digest in digests:
            existing = existing_by_key[digest.source_key]
            if (
                not existing
                or existing.get("batch_id") != batch_id
                or existing.get("content_sha256") != digest.content_sha256
            ):
                raise ConflictError(
                    "Digest batch đã seal; chỉ cho phép retry đúng nội dung hiện tại"
                )
            self._validate_daily_hashes(existing, digest)

        results: list[dict[str, Any]] = []
        wake = False
        for digest in digests:
            existing = existing_by_key[digest.source_key]
            assert existing is not None
            if digest.source_updated_at < str(existing["source_updated_at"]):
                action = "stale"
                status = str(existing["status"])
            else:
                needs_hash_backfill = (
                    (not existing.get("identity_hash") and digest.identity_hash)
                    or (not existing.get("digest_hash") and digest.digest_hash)
                )
                if (
                    digest.source_updated_at > str(existing["source_updated_at"])
                    or needs_hash_backfill
                ):
                    existing = self.storage.advance_digest_source(
                        existing["id"],
                        batch_id=batch_id,
                        expected_content_sha256=str(existing["content_sha256"]),
                        expected_source_updated_at=str(
                            existing["source_updated_at"]
                        ),
                        allow_sealed=True,
                        source_updated_at=digest.source_updated_at,
                        source_run_id=digest.source_run_id,
                        identity_hash=digest.identity_hash,
                        digest_hash=digest.digest_hash,
                    )
                if existing["status"] == "failed":
                    existing = self.storage.requeue_daily_document(
                        str(existing["id"]),
                        batch_id=batch_id,
                        expected_content_sha256=str(existing["content_sha256"]),
                        expected_source_updated_at=str(
                            existing["source_updated_at"]
                        ),
                        allow_sealed=True,
                    )
                    status = str(existing["status"])
                    if status == "pending":
                        action, wake = "retried", True
                    else:
                        action = "unchanged"
                else:
                    action, status = "unchanged", str(existing["status"])
            results.append(
                {
                    "ticker": digest.ticker,
                    "document_id": existing["id"],
                    "action": action,
                    "status": status,
                }
            )
        if wake:
            self._wake_worker()
        return results

    async def seal_digest_batch(
        self, batch_id: str
    ) -> tuple[str, dict[str, Any] | None]:
        """Serialize sealing with digest upserts in the single-writer process."""

        async with self._upload_lock:
            return self.storage.seal_digest_batch(batch_id)

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
            self._delete_superseded_vectors(document_id)
            if document.get("source_type") == "trading_digest":
                sections = self.storage.digest_sections(document_id)
                if document.get("content_kind") == FULL_REPORT_V1:
                    decision_section = self.storage.structured_decision_section(
                        document_id
                    )
                    if decision_section is None:
                        raise ValidationError(
                            "Full Analysis chưa có canonical payload"
                        )
                    sections.append(decision_section)
                parents, children, child_texts, metadata = build_digest_chunks(
                    document=document,
                    sections=sections,
                    settings=self.settings,
                )
                page_count = 0
                if not children:
                    raise ValidationError("Digest không có nội dung có thể index")
            else:
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
                page_count = len(pages)
                for item in metadata:
                    item["source_type"] = "pdf"
                    item["content_kind"] = "pdf"
                    item["report_schema"] = "pdf"
                    item["research_mode"] = str(
                        document.get("research_mode") or "legacy_unknown"
                    )
                    item["data_provenance"] = str(
                        document.get("data_provenance") or "legacy_unknown"
                    )
                if not children:
                    raise ValidationError("PDF không có nội dung text có thể index")
            child_ids = [child["id"] for child in children]
            self.index.add_children(ids=child_ids, texts=child_texts, metadatas=metadata)
            try:
                self.storage.finish_indexing(
                    document_id=document_id,
                    page_count=page_count,
                    parents=parents,
                    children=children,
                    fingerprint=self.index_fingerprint,
                    collection_name=self.collection_name,
                )
            except Exception:
                self.index.delete(child_ids)
                raise
            # A batch may have acquired provenance while this document was
            # already indexing with a legacy snapshot.  The durable metadata
            # outbox makes this in-place correction idempotent and crash-safe.
            self.reconcile_vector_metadata_updates(document_id)
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
                try:
                    self.index.delete(child_ids)
                except Exception as cleanup_exc:
                    self.logger.error(
                        "index cleanup failed",
                        extra={
                            "event": "index_cleanup_failed",
                            "document_id": document_id,
                            "error": type(cleanup_exc).__name__,
                        },
                    )
            public_error = (
                str(exc) if isinstance(exc, ValidationError) else type(exc).__name__
            )
            self.storage.mark_failed(document_id, public_error)
            self.logger.error(
                "index failed",
                extra={"event": "index_failed", "document_id": document_id, "error": type(exc).__name__},
            )

    def _delete_superseded_vectors(self, document_id: str) -> None:
        """Drain the durable vector-deletion outbox before indexing new chunks."""

        for deletion in self.storage.pending_vector_deletions(document_id):
            child_ids = list(deletion["child_ids"])
            stored_collection = str(deletion["collection_name"])
            self.index.delete(child_ids, stored_collection or self.collection_name)
            self.storage.finish_vector_deletion(
                document_id, stored_collection, child_ids
            )

    def reconcile_vector_metadata_updates(
        self,
        document_id: str | None = None,
        *,
        batch_id: str | None = None,
    ) -> int:
        """Apply durable provenance updates without recomputing embeddings."""

        completed = 0
        try:
            updates = self.storage.pending_vector_metadata_updates(
                document_id, batch_id=batch_id
            )
        except Exception as exc:
            self.logger.error(
                "vector provenance reconciliation scan failed",
                extra={
                    "event": "vector_provenance_reconciliation_scan_failed",
                    "error": type(exc).__name__,
                },
            )
            return 0
        for update in updates:
            try:
                if update["status"] != "ready" or not update.get("collection_name"):
                    continue
                child_ids = self.storage.child_ids_for_document(
                    update["document_id"]
                )
                if not child_ids:
                    continue
                applied = self.index.update_provenance(
                    child_ids,
                    str(update["research_mode"]),
                    str(update["data_provenance"]),
                    str(update["collection_name"]),
                )
                if not applied:
                    continue
                self.storage.finish_vector_metadata_update(
                    str(update["document_id"]),
                    str(update["research_mode"]),
                    str(update["data_provenance"]),
                )
                completed += 1
            except Exception as exc:
                self.logger.error(
                    "vector provenance reconciliation failed",
                    extra={
                        "event": "vector_provenance_reconciliation_failed",
                        "document_id": update["document_id"],
                        "error": type(exc).__name__,
                    },
                )
        return completed

    def process_next(self) -> bool:
        document = self.storage.claim_next_pending()
        if not document:
            return False
        self.process_document(document)
        return True

    def process_pending_batch(self) -> int:
        documents = self.storage.claim_pending(self.settings.digest_writer_claim_size)
        for document in documents:
            self.process_document(document)
        return len(documents)


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
            await asyncio.to_thread(
                self.ingestion.reconcile_vector_metadata_updates
            )
            processed = await asyncio.to_thread(self.ingestion.process_pending_batch)
            if processed:
                continue
            self._event.clear()
            try:
                await asyncio.wait_for(self._event.wait(), timeout=1.0)
            except TimeoutError:
                pass
