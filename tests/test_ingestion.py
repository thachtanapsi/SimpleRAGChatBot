from __future__ import annotations

from dataclasses import replace
from io import BytesIO
from pathlib import Path

import pytest
from fastapi import UploadFile
from pypdf import PdfWriter
from starlette.datastructures import Headers

from rag_app.errors import ValidationError
from rag_app.ingestion import (
    IngestionService,
    build_parent_child_chunks,
    deterministic_id,
    inspect_pdf,
    safe_pdf_name,
)
from rag_app.storage import SQLiteStorage


def test_chunks_never_cross_page_and_ids_are_deterministic(settings):
    pages = ["PAGE_ONE " * 40, "PAGE_TWO " * 40]
    first = build_parent_child_chunks(
        file_hash="abc", filename="paper.pdf", pages=pages, settings=settings
    )
    second = build_parent_child_chunks(
        file_hash="abc", filename="paper.pdf", pages=pages, settings=settings
    )
    parents, children, child_texts, metadata = first

    assert [item["id"] for item in parents] == [item["id"] for item in second[0]]
    assert [item["id"] for item in children] == [item["id"] for item in second[1]]
    assert all("PAGE_ONE" in item["text"] for item in parents if item["page"] == 1)
    assert all("PAGE_TWO" in item["text"] for item in parents if item["page"] == 2)
    assert all(item["page"] == meta["page"] for item, meta in zip(children, metadata))
    assert len(children) == len(child_texts) == len(metadata)
    assert [item["text"] for item in children] == child_texts


def test_deterministic_id_changes_when_position_changes():
    assert deterministic_id("hash", 1, 0, prefix="parent") == deterministic_id(
        "hash", 1, 0, prefix="parent"
    )
    assert deterministic_id("hash", 1, 0, prefix="parent") != deterministic_id(
        "hash", 1, 1, prefix="parent"
    )


@pytest.mark.parametrize(
    "filename", ["../secret.pdf", "folder/file.pdf", "folder\\file.pdf", "not-pdf.txt"]
)
def test_unsafe_pdf_filename_is_rejected(filename):
    with pytest.raises(ValidationError):
        safe_pdf_name(filename)


def test_script_filename_is_data_not_path():
    assert safe_pdf_name("<script>alert(1).pdf") == "<script>alert(1).pdf"


def test_broken_pdf_signature_is_rejected(settings, tmp_path: Path):
    broken = tmp_path / "broken.pdf"
    broken.write_bytes(b"not a pdf")
    with pytest.raises(ValidationError, match="Chữ ký"):
        inspect_pdf(broken, settings)


def write_blank_pdf(path: Path, pages: int = 1) -> None:
    writer = PdfWriter()
    for _ in range(pages):
        writer.add_blank_page(width=100, height=100)
    with path.open("wb") as stream:
        writer.write(stream)


def test_pdf_page_limit_is_enforced(settings, tmp_path: Path):
    pdf = tmp_path / "many.pdf"
    write_blank_pdf(pdf, pages=2)
    limited = replace(settings, max_pdf_pages=1)
    with pytest.raises(ValidationError, match="vượt quá"):
        inspect_pdf(pdf, limited)


class NullLogger:
    def info(self, *_args, **_kwargs): pass
    def error(self, *_args, **_kwargs): pass


class NullIndex:
    def add_children(self, **_kwargs):
        raise AssertionError("PDF scan không được embedding")
    def delete(self, *_args, **_kwargs): pass


@pytest.mark.asyncio
async def test_upload_size_and_mime_are_rejected_before_index(settings):
    storage = SQLiteStorage(settings.sqlite_path)
    storage.initialise()
    limited = replace(settings, max_upload_bytes=5)
    service = IngestionService(limited, storage, NullIndex(), NullLogger())
    oversized = UploadFile(
        file=BytesIO(b"%PDF-too-large"),
        filename="large.pdf",
        headers=Headers({"content-type": "application/pdf"}),
    )
    with pytest.raises(ValidationError, match="50 MB"):
        await service.accept_upload(oversized)

    wrong_mime = UploadFile(
        file=BytesIO(b"%PDF-test"),
        filename="paper.pdf",
        headers=Headers({"content-type": "text/plain"}),
    )
    with pytest.raises(ValidationError, match="MIME"):
        await service.accept_upload(wrong_mime)


def test_textless_pdf_is_marked_as_scan_without_embedding(settings, tmp_path: Path):
    pdf = settings.uploads_dir / "scan.pdf"
    write_blank_pdf(pdf)
    storage = SQLiteStorage(settings.sqlite_path)
    storage.initialise()
    storage.create_pending_document(
        document_id="doc_scan",
        filename="scan.pdf",
        sha256="scan-hash",
        stored_path=pdf,
        size_bytes=pdf.stat().st_size,
        page_count=1,
    )
    document = storage.claim_next_pending()
    service = IngestionService(settings, storage, NullIndex(), NullLogger())
    service.process_document(document)
    failed = storage.get_document("doc_scan")
    assert failed["status"] == "failed"
    assert "OCR" in failed["error"]
