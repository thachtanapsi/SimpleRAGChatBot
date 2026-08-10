"""CLI tương thích ngược và các lệnh vận hành."""

from __future__ import annotations

import argparse
import asyncio
import shutil
import sys
from pathlib import Path

from .config import Settings
from .ingestion import inspect_pdf, safe_pdf_name, sha256_file
from .runtime import Runtime


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Simple Local RAG")
    commands = parser.add_subparsers(dest="command")
    commands.add_parser("chat", help="Chat tương tác trong terminal")
    ingest = commands.add_parser("ingest", help="Import một PDF hoặc thư mục PDF")
    ingest.add_argument("path", nargs="?", default="papers")
    evaluate = commands.add_parser("evaluate", help="Chạy bộ eval local JSONL")
    evaluate.add_argument("path", nargs="?", default="evals/questions.jsonl")
    serve = commands.add_parser("serve", help="Chạy web/API trên localhost")
    serve.add_argument("--host", default="127.0.0.1", choices=["127.0.0.1", "localhost"])
    serve.add_argument("--port", default=8000, type=int)
    return parser


def register_local_file(runtime: Runtime, source: Path) -> bool:
    source = source.resolve()
    file_hash, size = sha256_file(source)
    if runtime.storage.get_document_by_hash(file_hash):
        return False
    if size > runtime.settings.max_upload_bytes:
        raise ValueError(f"{source.name}: vượt quá 50 MB")
    page_count = inspect_pdf(source, runtime.settings)
    document_id = f"doc_{file_hash}"
    destination = runtime.settings.uploads_dir / f"{document_id}.pdf"
    shutil.copyfile(source, destination)
    runtime.storage.create_pending_document(
        document_id=document_id,
        filename=safe_pdf_name(source.name),
        sha256=file_hash,
        stored_path=destination,
        size_bytes=size,
        page_count=page_count,
    )
    runtime.worker.wake()
    return True


async def wait_for_jobs(runtime: Runtime) -> None:
    while any(
        item["status"] in {"pending", "indexing"}
        for item in runtime.storage.list_documents()
    ):
        await asyncio.sleep(0.25)


async def ingest_command(path: Path) -> int:
    runtime = Runtime()
    await runtime.start()
    try:
        candidates = sorted(path.rglob("*.pdf")) if path.is_dir() else [path]
        if not candidates:
            print("Không tìm thấy file PDF.")
            return 1
        imported = 0
        for candidate in candidates:
            if register_local_file(runtime, candidate):
                imported += 1
        await wait_for_jobs(runtime)
        failed = [item for item in runtime.storage.list_documents() if item["status"] == "failed"]
        print(f"Đã thêm {imported} tài liệu; failed={len(failed)}.")
        return 1 if failed else 0
    finally:
        await runtime.stop()


async def chat_command() -> int:
    runtime = Runtime()
    await runtime.start()
    try:
        await wait_for_jobs(runtime)
        session_id: str | None = None
        print("Local RAG sẵn sàng. Gõ /quit để thoát, /new để tạo hội thoại mới.")
        while True:
            try:
                question = input("\nBạn: ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if question in {"/quit", "/exit"}:
                break
            if question == "/new":
                session_id = None
                print("Đã tạo hội thoại mới.")
                continue
            if not question:
                continue
            response = await runtime.chat.answer(question, session_id)
            session_id = response["session_id"]
            print(f"\nGemma: {response['answer']}")
            for citation in response["citations"]:
                if citation["cited"]:
                    print(f"  [{citation['id']}] {citation['filename']} — trang {citation['page']}")
        return 0
    finally:
        await runtime.stop()


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    command = args.command or "chat"
    if command == "serve":
        import uvicorn

        uvicorn.run("rag_app.api:app", host=args.host, port=args.port, reload=False)
        return 0
    if command == "ingest":
        return asyncio.run(ingest_command(Path(args.path)))
    if command == "evaluate":
        from .evaluation import evaluate_file

        return asyncio.run(evaluate_file(Path(args.path)))
    return asyncio.run(chat_command())


if __name__ == "__main__":
    sys.exit(main())
