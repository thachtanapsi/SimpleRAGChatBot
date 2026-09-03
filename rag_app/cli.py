"""CLI tương thích ngược và các lệnh vận hành."""

from __future__ import annotations

import argparse
import asyncio
import json
import shutil
import sys
from pathlib import Path
from typing import Any

from .config import Settings
from .full_reports import canonicalise_full_report
from .ingestion import inspect_pdf, safe_pdf_name, sha256_file
from .runtime import Runtime
from .storage import SQLiteStorage


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Simple Local RAG")
    commands = parser.add_subparsers(dest="command")
    commands.add_parser("chat", help="Chat tương tác trong terminal")
    ingest = commands.add_parser("ingest", help="Import một PDF hoặc thư mục PDF")
    ingest.add_argument("path", nargs="?", default="papers")
    commands.add_parser(
        "reindex", help="Ép tạo lại index cho toàn bộ PDF đang được quản lý"
    )
    evaluate = commands.add_parser("evaluate", help="Chạy bộ eval local JSONL")
    evaluate.add_argument("path", nargs="?", default="evals/questions.jsonl")
    evaluate.add_argument(
        "--release-gate",
        action="store_true",
        help="Dùng schema advanced v1 và quality gates production (tối thiểu 150 câu)",
    )
    evaluate.add_argument(
        "--results",
        help="JSONL trace đã gắn nhãn từ lần chạy advanced local",
    )
    evaluate.add_argument(
        "--legacy-results",
        help="JSON result của lần chạy regression bộ 30 câu hiện tại",
    )
    evaluate.add_argument(
        "--legacy-baseline",
        help="JSON baseline đã commit của đúng bộ 30 câu để kiểm tra no-regression",
    )
    evaluate.add_argument(
        "--validate-only",
        action="store_true",
        help="Chỉ kiểm tra advanced golden JSONL, không chấm release gate",
    )
    serve = commands.add_parser("serve", help="Chạy web/API trên localhost")
    serve.add_argument(
        "--host",
        default="127.0.0.1",
        choices=["127.0.0.1", "localhost", "0.0.0.0"],
        help="Mặc định chỉ loopback; 0.0.0.0 chỉ dành cho mạng container tin cậy",
    )
    serve.add_argument("--port", default=8000, type=int)
    backfill = commands.add_parser(
        "backfill-full-reports",
        help="Dựng canonical Full Analysis payload từ SQLite hiện hữu",
    )
    mode = backfill.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--apply", action="store_true")
    graph_reindex = commands.add_parser(
        "graph-reindex",
        help="Lập kế hoạch hoặc dựng lại evidence graph, không reindex Chroma/PDF",
    )
    graph_mode = graph_reindex.add_mutually_exclusive_group(required=True)
    graph_mode.add_argument("--dry-run", action="store_true")
    graph_mode.add_argument("--apply", action="store_true")
    graph_reindex.add_argument("--document-id")
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


async def reindex_command() -> int:
    runtime = Runtime()
    # Không reset job ``indexing``: nếu server khác đang xử lý, force_reindex
    # phải nhìn thấy trạng thái đó và từ chối thay vì coi là job bị gián đoạn.
    await runtime.start(start_worker=False, reset_interrupted_jobs=False)
    try:
        result = await runtime.force_reindex()
        scheduled = set(result["scheduled"])
        skipped = result["skipped"]
        if scheduled:
            runtime.worker.start()
            await wait_for_jobs(runtime)

        documents = {
            item["id"]: item for item in runtime.storage.list_documents()
        }
        failed = [
            document_id
            for document_id in scheduled
            if documents.get(document_id, {}).get("status") != "ready"
        ]
        succeeded = len(scheduled) - len(failed)
        print(
            f"Đã reindex {succeeded}/{len(scheduled)} tài liệu; "
            f"failed={len(failed)}; skipped={len(skipped)}."
        )
        return 1 if failed or skipped else 0
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
                    page_end = citation.get("page_end", citation["page"])
                    page_label = (
                        str(citation["page"])
                        if page_end == citation["page"]
                        else f"{citation['page']}–{page_end}"
                    )
                    print(
                        f"  [{citation['id']}] {citation['filename']} — trang {page_label}"
                    )
        return 0
    finally:
        await runtime.stop()


def backfill_full_reports_command(*, apply: bool) -> int:
    """Backfill without constructing embeddings, Chroma, Ollama or an LLM."""

    settings = Settings.from_env()
    storage = SQLiteStorage(settings.sqlite_path)
    storage.initialise()
    counts = {
        "scanned": 0,
        "stored": 0,
        "partial": 0,
        "requeued": 0,
        "failed": 0,
    }
    for candidate in storage.full_report_backfill_candidates():
        counts["scanned"] += 1
        try:
            report = canonicalise_full_report(
                storage.reconstruct_full_report_wire(str(candidate["id"]))
            )
            if report.decision_status == "partial":
                counts["partial"] += 1
            needs_store = (
                candidate.get("canonical_sha256") != report.canonical_sha256
                or candidate.get("decision_sha256") != report.decision_sha256
                or candidate.get("stored_canonical_json") != report.canonical_json
                or candidate.get("stored_decision_json") != report.decision_json
                or candidate.get("stored_decision_status")
                != report.decision_status
                or candidate.get("projection_version")
                != report.projection_version
            )
            needs_reindex = (
                candidate.get("decision_sha256") != report.decision_sha256
                or candidate.get("projection_version")
                != report.projection_version
                or candidate.get("decision_index_version")
                != report.projection_version
            )
            if not needs_store and not needs_reindex:
                continue
            if candidate["status"] == "indexing":
                raise RuntimeError("Full Analysis đang indexing")
            if apply:
                storage.store_full_report_payload(str(candidate["id"]), report)
                counts["stored"] += 1
                if needs_reindex and candidate["status"] in {"ready", "failed"}:
                    storage.requeue_document(str(candidate["id"]))
                    counts["requeued"] += 1
            else:
                counts["stored"] += 1
                if needs_reindex and candidate["status"] in {"ready", "failed"}:
                    counts["requeued"] += 1
        except Exception:
            counts["failed"] += 1
    print(
        json.dumps(
            {"mode": "apply" if apply else "dry-run", **counts},
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return 1 if counts["failed"] else 0


async def _local_ollama_model_digest(settings: Settings, model_name: str) -> str | None:
    """Read a local model digest; never contacts a cloud endpoint."""

    import httpx

    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            response = await client.get(f"{settings.ollama_base_url}/api/tags")
            response.raise_for_status()
        for item in response.json().get("models", []):
            if not isinstance(item, dict):
                continue
            name = str(item.get("name") or item.get("model") or "")
            if name in {model_name, f"{model_name}:latest"} and item.get("digest"):
                return str(item["digest"])
    except Exception:
        return None
    return None


async def graph_reindex_command(
    *, apply: bool, document_id: str | None = None
) -> int:
    """Rebuild only derived SQLite graph state for normalized research."""

    from langchain_ollama import ChatOllama

    from .graph import GraphService
    from .graph_extraction import LocalGraphExtractor, graph_extractor_fingerprint

    settings = Settings.from_env()
    storage = SQLiteStorage(settings.sqlite_path)
    storage.initialise()
    clauses = [
        "status='ready'",
        "source_type='trading_digest'",
        "content_kind IN ('daily_digest_v1', 'full_report_v1')",
        "index_fingerprint IS NOT NULL",
    ]
    params: list[str] = []
    if document_id:
        clauses.append("id=?")
        params.append(document_id)
    with storage.connect() as db:
        rows = db.execute(
            f"SELECT DISTINCT index_fingerprint FROM documents "
            f"WHERE {' AND '.join(clauses)} ORDER BY index_fingerprint",
            tuple(params),
        ).fetchall()
    fingerprints = [str(row["index_fingerprint"]) for row in rows]
    model_digest = await _local_ollama_model_digest(
        settings, settings.graph_extractor_model
    )
    summary: dict[str, Any] = {
        "mode": "apply" if apply else "dry-run",
        "document_id": document_id,
        "eligible": 0,
        "scheduled": 0,
        "built": 0,
        "failed": 0,
        "fingerprints": len(fingerprints),
    }
    for index_fingerprint in fingerprints:
        extractor_fingerprint = graph_extractor_fingerprint(
            index_fingerprint=index_fingerprint,
            model_name=settings.graph_extractor_model,
            model_digest=model_digest,
        )
        generation_lock = asyncio.Semaphore(1)
        extractor = None
        if apply:
            graph_llm = ChatOllama(
                model=settings.graph_extractor_model,
                base_url=settings.ollama_base_url,
                reasoning=False,
                temperature=0,
                num_ctx=8192,
                num_predict=min(settings.answer_num_predict, 1536),
            )
            extractor = LocalGraphExtractor(graph_llm)
        service = GraphService(
            storage,
            index_fingerprint=index_fingerprint,
            extractor_fingerprint=extractor_fingerprint,
            extractor=extractor,
            generation_lock=generation_lock,
            worker_concurrency=1,
        )
        plan = await asyncio.to_thread(
            service.reindex,
            document_id,
            apply=apply,
        )
        summary["eligible"] += int(plan["eligible"])
        summary["scheduled"] += len(plan["items"])
        if not apply:
            continue
        for item in plan["items"]:
            try:
                await service.build_document_async(str(item["document_id"]))
                summary["built"] += 1
            except Exception:
                summary["failed"] += 1
    print(
        json.dumps(
            summary,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return 1 if summary["failed"] else 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    command = args.command or "chat"
    if command == "serve":
        import uvicorn

        uvicorn.run("rag_app.api:app", host=args.host, port=args.port, reload=False)
        return 0
    if command == "ingest":
        return asyncio.run(ingest_command(Path(args.path)))
    if command == "reindex":
        return asyncio.run(reindex_command())
    if command == "evaluate":
        from .evaluation import (
            AdvancedEvalValidationError,
            evaluate_advanced_release_gate,
            evaluate_file,
        )

        if args.release_gate:
            try:
                return evaluate_advanced_release_gate(
                    Path(args.path),
                    results_path=Path(args.results) if args.results else None,
                    legacy_result_path=(
                        Path(args.legacy_results) if args.legacy_results else None
                    ),
                    legacy_baseline_path=(
                        Path(args.legacy_baseline) if args.legacy_baseline else None
                    ),
                    validate_only=bool(args.validate_only),
                )
            except AdvancedEvalValidationError as exc:
                print(
                    json.dumps(
                        {
                            "mode": "advanced_release_gate",
                            "valid": False,
                            "error": str(exc),
                        },
                        ensure_ascii=False,
                    ),
                    file=sys.stderr,
                )
                return 2
        if (
            args.results
            or args.legacy_results
            or args.legacy_baseline
            or args.validate_only
        ):
            raise ValueError(
                "--results/--legacy-results/--legacy-baseline/--validate-only "
                "chỉ dùng với --release-gate"
            )
        return asyncio.run(evaluate_file(Path(args.path)))
    if command == "backfill-full-reports":
        return backfill_full_reports_command(apply=bool(args.apply))
    if command == "graph-reindex":
        return asyncio.run(
            graph_reindex_command(
                apply=bool(args.apply),
                document_id=args.document_id,
            )
        )
    return asyncio.run(chat_command())


if __name__ == "__main__":
    sys.exit(main())
