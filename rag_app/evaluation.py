"""Bộ eval không dùng model judge/cloud; chấm retrieval, citation và abstention."""

from __future__ import annotations

import asyncio
import json
import time
from collections import defaultdict
from dataclasses import replace
from pathlib import Path
from typing import Any

from .retrieval import ParentChildRetriever
from .runtime import Runtime


ABSTENTION_MARKERS = (
    "không tìm thấy bằng chứng phù hợp",
    "could not find sufficient evidence",
    "do not have information",
    "does not contain information",
    "sources do not contain",
    "not provided in the sources",
    "insufficient evidence",
)


def is_abstention(answer: str, cited: list[dict[str, Any]]) -> bool:
    answer_lower = answer.lower()
    return not cited and any(marker in answer_lower for marker in ABSTENTION_MARKERS)


def percentile_95(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round(0.95 * len(ordered) + 0.5) - 1))
    return round(ordered[index], 2)


async def evaluate_file(path: Path) -> int:
    cases = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if len(cases) < 20:
        raise ValueError("Bộ eval phải có ít nhất 20 câu")

    runtime = Runtime()
    await runtime.start()
    try:
        while any(
            item["status"] in {"pending", "indexing"}
            for item in runtime.storage.list_documents()
        ):
            await asyncio.sleep(0.25)

        documents = {
            item["filename"]: item["id"]
            for item in runtime.storage.list_ready_documents()
        }
        all_document_ids = list(documents.values())
        dense_retriever = ParentChildRetriever(
            replace(runtime.settings, hybrid_search=False),
            runtime.storage,
            runtime.index,
        )
        hybrid_retriever = ParentChildRetriever(
            replace(runtime.settings, hybrid_search=True),
            runtime.storage,
            runtime.index,
        )
        sessions: dict[str, str] = {}
        totals = defaultdict(int)
        exact_totals = defaultdict(int)
        diagnostics: list[dict[str, Any]] = []
        dense_latencies: list[float] = []
        hybrid_latencies: list[float] = []

        for case in cases:
            expected_document = case.get("document")
            if expected_document and expected_document not in documents:
                raise ValueError(f"Thiếu tài liệu eval: {expected_document}")
            conversation = case.get("conversation")
            if case.get("kind") == "exact_keyword":
                exact_totals["cases"] += 1
                # Đảo thứ tự để warm-up không thiên vị dense hoặc hybrid.
                retrievers = [
                    ("hybrid", hybrid_retriever, hybrid_latencies),
                    ("dense", dense_retriever, dense_latencies),
                ]
                if exact_totals["cases"] % 2 == 0:
                    retrievers.reverse()
                retrieval_outputs: dict[str, list[Any]] = {}
                for label, retriever, latencies in retrievers:
                    started = time.perf_counter()
                    retrieval_outputs[label] = await asyncio.to_thread(
                        retriever.retrieve, case["question"], all_document_ids
                    )
                    latencies.append((time.perf_counter() - started) * 1000)
                expected_pages_for_probe = set(case.get("expected_pages", []))
                for label, sources in retrieval_outputs.items():
                    if any(
                        source.filename == expected_document
                        and (
                            not expected_pages_for_probe
                            or source.page in expected_pages_for_probe
                        )
                        for source in sources
                    ):
                        exact_totals[f"{label}_hit"] += 1
            response = await runtime.chat.answer(
                case["question"],
                sessions.get(conversation) if conversation else None,
                # Tìm trên toàn corpus để đo cả khả năng tránh tài liệu không liên quan.
                None,
            )
            if conversation:
                sessions[conversation] = response["session_id"]
            expected_pages = set(case.get("expected_pages", []))
            retrieved = response["citations"]
            cited = [item for item in retrieved if item["cited"]]

            if case["kind"] == "unanswerable":
                totals["unanswerable"] += 1
                correct = is_abstention(response["answer"], cited)
                if correct:
                    totals["correct_abstention"] += 1
                if not correct:
                    diagnostics.append(
                        {"id": case["id"], "reason": "incorrect_abstention"}
                    )
                continue

            totals["answerable"] += 1
            retrieval_hit = any(
                item["filename"] == expected_document
                and (not expected_pages or item["page"] in expected_pages)
                for item in retrieved
            )
            if retrieval_hit:
                totals["retrieval_hit"] += 1
            if cited:
                totals["answers_with_citation"] += 1
            invalid_citations: list[dict[str, Any]] = []
            for item in cited:
                totals["cited_sources"] += 1
                if item["filename"] == expected_document and (
                    not expected_pages or item["page"] in expected_pages
                ):
                    totals["correct_citations"] += 1
                else:
                    totals["irrelevant_citations"] += 1
                    invalid_citations.append(
                        {"filename": item["filename"], "page": item["page"]}
                    )
            if not retrieval_hit or invalid_citations:
                diagnostics.append(
                    {
                        "id": case["id"],
                        "reason": "retrieval_miss" if not retrieval_hit else "invalid_citation",
                        "expected_document": expected_document,
                        "expected_pages": sorted(expected_pages),
                        "invalid_citations": invalid_citations,
                    }
                )

        def ratio(numerator: str, denominator: str) -> float:
            return (
                round(totals[numerator] / totals[denominator], 4)
                if totals[denominator]
                else 0.0
            )

        dense_exact_recall = (
            round(exact_totals["dense_hit"] / exact_totals["cases"], 4)
            if exact_totals["cases"]
            else 0.0
        )
        hybrid_exact_recall = (
            round(exact_totals["hybrid_hit"] / exact_totals["cases"], 4)
            if exact_totals["cases"]
            else 0.0
        )
        dense_p95 = percentile_95(dense_latencies)
        hybrid_p95 = percentile_95(hybrid_latencies)
        latency_ratio = round(hybrid_p95 / dense_p95, 4) if dense_p95 else 0.0
        exact_quality_ok = hybrid_exact_recall >= dense_exact_recall and (
            dense_exact_recall >= 1.0
            or hybrid_exact_recall - dense_exact_recall >= 0.05
        )

        metrics = {
            "cases": len(cases),
            "parent_recall_at_5": ratio("retrieval_hit", "answerable"),
            "citation_correct_page": ratio("correct_citations", "cited_sources"),
            "abstention_accuracy": ratio("correct_abstention", "unanswerable"),
            "answers_with_citation": ratio("answers_with_citation", "answerable"),
            "irrelevant_citations": totals["irrelevant_citations"],
            "hybrid_comparison": {
                "chat_mode": (
                    "hybrid" if runtime.settings.hybrid_search else "dense"
                ),
                "exact_keyword_cases": exact_totals["cases"],
                "dense_parent_recall_at_5": dense_exact_recall,
                "hybrid_parent_recall_at_5": hybrid_exact_recall,
                "dense_retrieval_p95_ms": dense_p95,
                "hybrid_retrieval_p95_ms": hybrid_p95,
                "latency_ratio": latency_ratio,
                "quality_gate_passed": exact_quality_ok,
                "latency_gate_passed": latency_ratio <= 1.20,
            },
            "acceptance": {
                "parent_recall_at_5": 0.85,
                "citation_correct_page": 0.90,
                "abstention_accuracy": 0.90,
                "irrelevant_citations": 0,
            },
            "failed_cases": diagnostics,
        }
        passed = (
            metrics["parent_recall_at_5"] >= 0.85
            and metrics["citation_correct_page"] >= 0.90
            and metrics["abstention_accuracy"] >= 0.90
            and metrics["irrelevant_citations"] == 0
        )
        metrics["hybrid_comparison"]["recommended_default"] = bool(
            passed
            and runtime.settings.hybrid_search
            and metrics["hybrid_comparison"]["quality_gate_passed"]
            and metrics["hybrid_comparison"]["latency_gate_passed"]
        )
        print(json.dumps(metrics, ensure_ascii=False, indent=2))
        return 0 if passed else 2
    finally:
        await runtime.stop()
