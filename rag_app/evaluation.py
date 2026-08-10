"""Bộ eval không dùng model judge/cloud; chấm retrieval, citation và abstention."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

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
            import asyncio

            await asyncio.sleep(0.25)

        documents = {
            item["filename"]: item["id"]
            for item in runtime.storage.list_ready_documents()
        }
        sessions: dict[str, str] = {}
        totals = defaultdict(int)
        diagnostics: list[dict[str, Any]] = []

        for case in cases:
            expected_document = case.get("document")
            if expected_document and expected_document not in documents:
                raise ValueError(f"Thiếu tài liệu eval: {expected_document}")
            conversation = case.get("conversation")
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
            return round(totals[numerator] / totals[denominator], 4) if totals[denominator] else 0.0

        metrics = {
            "cases": len(cases),
            "parent_recall_at_5": ratio("retrieval_hit", "answerable"),
            "citation_correct_page": ratio("correct_citations", "cited_sources"),
            "abstention_accuracy": ratio("correct_abstention", "unanswerable"),
            "answers_with_citation": ratio("answers_with_citation", "answerable"),
            "irrelevant_citations": totals["irrelevant_citations"],
            "acceptance": {
                "parent_recall_at_5": 0.85,
                "citation_correct_page": 0.90,
                "abstention_accuracy": 0.90,
                "irrelevant_citations": 0,
            },
            "failed_cases": diagnostics,
        }
        print(json.dumps(metrics, ensure_ascii=False, indent=2))
        passed = (
            metrics["parent_recall_at_5"] >= 0.85
            and metrics["citation_correct_page"] >= 0.90
            and metrics["abstention_accuracy"] >= 0.90
            and metrics["irrelevant_citations"] == 0
        )
        return 0 if passed else 2
    finally:
        await runtime.stop()
