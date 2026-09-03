"""Agentic tool adapter for bounded SQLite evidence-graph retrieval."""

from __future__ import annotations

import asyncio
from typing import Any, Sequence

from .graph import GraphRetriever
from .orchestration import SoftFilterConflict, ToolName, ToolRequest
from .reranking import RerankCandidate, RerankerService
from .retrieval import RetrievalDiagnostics, RetrievedSource


class GraphRetrievalTool:
    name: ToolName = "graph"

    def __init__(
        self,
        retriever: GraphRetriever,
        *,
        reranker: RerankerService | None = None,
        child_k: int = 12,
        min_score: float = 0.5,
    ):
        self.retriever = retriever
        self.reranker = reranker
        self.child_k = max(1, min(int(child_k), 20))
        self.min_score = max(0.0, min(float(min_score), 1.0))

    def _retrieve_sync(self, request: ToolRequest) -> list[RetrievedSource]:
        try:
            filters = request.effective_retrieval_filters()
        except SoftFilterConflict:
            return []
        document_ids = list(request.scope.allowed_document_ids)
        if filters and filters.pop("latest_only", False):
            document_ids = self.retriever.storage.latest_ready_document_ids(
                document_ids
            )
            filters = filters or None
        if not document_ids:
            return []
        rows = self.retriever.search(
            request.query,
            # Traverse a wider bounded graph pool, then cross-encode before the
            # orchestrator applies its five-parent context cap.
            k=20,
            document_ids=document_ids,
            filters=filters,
        )
        if self.reranker is not None and rows:
            candidate_rows = {str(row["parent_id"]): row for row in rows}
            reranked = self.reranker.rerank(
                request.query,
                [
                    RerankCandidate(
                        id=str(row["parent_id"]), text=str(row.get("text") or "")
                    )
                    for row in rows
                ],
                top_k=min(self.child_k, 20),
                min_score=self.min_score,
            )
            ranked_rows: list[dict[str, Any]] = []
            for item in reranked:
                candidate = candidate_rows.get(item.id)
                if candidate is None:
                    continue
                ranked_rows.append({**candidate, "rerank_score": item.score})
            rows = ranked_rows
        return [self._source(row, index) for index, row in enumerate(rows, start=1)]

    @staticmethod
    def _source(row: dict[str, Any], index: int) -> RetrievedSource:
        return RetrievedSource(
            id=str(index),
            document_id=str(row["document_id"]),
            filename=str(row["filename"]),
            page=int(row["page"]),
            page_end=int(row.get("page_end") or row["page"]),
            parent_id=str(row["parent_id"]),
            text=str(row["text"]),
            # Graph-only evidence must never masquerade as dense relevance.
            score=0.0,
            kind=str(row.get("kind") or "page"),
            source_type=str(row.get("source_type") or "trading_digest"),
            ticker=row.get("ticker"),
            analysis_date=row.get("analysis_date"),
            analysis_cutoff=row.get("analysis_cutoff"),
            digest_section=row.get("digest_section"),
            liquidity_rank=row.get("liquidity_rank"),
            research_mode=str(row.get("research_mode") or "legacy_unknown"),
            data_provenance=str(
                row.get("data_provenance") or "legacy_unknown"
            ),
            content_kind=str(row.get("content_kind") or "legacy_unknown"),
            report_schema=str(row.get("report_schema") or "legacy_unknown"),
            report_stage=row.get("report_stage"),
            section_status=row.get("section_status"),
            full_report_hash=row.get("full_report_hash"),
            pipeline_version=row.get("pipeline_version"),
            execution_generation=row.get("execution_generation"),
            contains_decision_content=bool(row.get("contains_decision_content")),
            diagnostics=RetrievalDiagnostics(
                child_id=str(row["parent_id"]),
                dense_rank=None,
                lexical_rank=None,
                dense_score=None,
                lexical_score=None,
                fusion_score=0.0,
                rerank_score=(
                    float(row["rerank_score"])
                    if row.get("rerank_score") is not None
                    else None
                ),
                graph_score=(
                    float(row["graph_score"])
                    if row.get("graph_score") is not None
                    else None
                ),
            ),
        )

    async def retrieve(self, request: ToolRequest) -> Sequence[RetrievedSource]:
        return await asyncio.to_thread(self._retrieve_sync, request)


__all__ = ["GraphRetrievalTool"]
