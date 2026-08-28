"""Child retrieval, parent grouping và context/citation an toàn."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date, datetime
from html import escape
from typing import Any, Sequence

from .config import Settings
from .full_reports import (
    FULL_REPORT_V1,
    FULL_SECTION_CITATION_LABELS,
    FULL_SECTION_LABELS,
)
from .storage import SQLiteStorage
from .vector_index import ChromaIndex


@dataclass(slots=True)
class RetrievedSource:
    id: str
    document_id: str
    filename: str
    page: int
    parent_id: str
    text: str
    score: float
    page_end: int | None = None
    kind: str = "page"
    source_type: str = "pdf"
    ticker: str | None = None
    analysis_date: str | None = None
    analysis_cutoff: str | None = None
    digest_section: str | None = None
    liquidity_rank: int | None = None
    research_mode: str = "legacy_unknown"
    data_provenance: str = "legacy_unknown"
    content_kind: str = "legacy_unknown"
    report_schema: str = "legacy_unknown"
    report_stage: str | None = None
    section_status: str | None = None
    full_report_hash: str | None = None
    pipeline_version: str | None = None
    execution_generation: int | None = None
    contains_decision_content: bool = False

    @property
    def resolved_page_end(self) -> int:
        return self.page if self.page_end is None else self.page_end

    def public(self, cited: bool = False) -> dict[str, Any]:
        payload = {
            "id": self.id,
            "document_id": self.document_id,
            "filename": self.filename,
            "page": self.page,
            "page_end": self.resolved_page_end,
            "parent_id": self.parent_id,
            "snippet": self.text[:500],
            "score": round(self.score, 4),
            "cited": cited,
            "source_type": self.source_type,
        }
        if self.source_type == "trading_digest":
            payload.update(
                {
                    "ticker": self.ticker,
                    "analysis_date": self.analysis_date,
                    "analysis_cutoff": self.analysis_cutoff,
                    "digest_section": self.digest_section,
                    "liquidity_rank": self.liquidity_rank,
                    "research_mode": self.research_mode,
                    "data_provenance": self.data_provenance,
                    "content_kind": self.content_kind,
                    "report_schema": self.report_schema,
                    "report_stage": self.report_stage,
                    "section_status": self.section_status,
                    "full_report_hash": self.full_report_hash,
                    "pipeline_version": self.pipeline_version,
                    "execution_generation": self.execution_generation,
                    "contains_decision_content": self.contains_decision_content,
                    "analysis_badge": (
                        "Full Analysis"
                        if self.content_kind == FULL_REPORT_V1
                        else "Daily digest"
                    ),
                    "label": self.digest_label,
                }
            )
        return payload

    @property
    def digest_label(self) -> str | None:
        if self.source_type != "trading_digest":
            return None
        labels = {
            "summary": "Tổng quan",
            "market": "Thị trường",
            "fundamentals": "Cơ bản",
            "news_events": "Tin tức & sự kiện",
            "sentiment": "Tâm lý",
            "macro": "Vĩ mô",
            "catalysts": "Yếu tố hỗ trợ",
            "risks_unknowns": "Rủi ro",
            "next_session_watch": "Theo dõi phiên tới",
            **FULL_SECTION_LABELS,
        }
        try:
            date_label = date.fromisoformat(str(self.analysis_date)).strftime("%d/%m/%Y")
        except ValueError:
            date_label = str(self.analysis_date or "")
        try:
            cutoff_label = datetime.fromisoformat(
                str(self.analysis_cutoff)
            ).strftime("%H:%M")
        except ValueError:
            cutoff_label = ""
        if self.content_kind == FULL_REPORT_V1:
            section_details = (
                str(self.report_stage or "Full").title(),
                FULL_SECTION_CITATION_LABELS.get(
                    str(self.digest_section),
                    str(self.digest_section or "Full report"),
                ),
            )
        else:
            section_details = (
                labels.get(
                    str(self.digest_section),
                    str(self.digest_section or "Digest"),
                ),
            )
        return " · ".join(
            item
            for item in (
                self.ticker or "?",
                f"{date_label} {cutoff_label}".strip(),
                *section_details,
                "dựng lại lịch sử"
                if self.data_provenance == "historical_replay"
                else None,
            )
            if item
        )


class ParentChildRetriever:
    def __init__(
        self,
        settings: Settings,
        storage: SQLiteStorage,
        index: ChromaIndex,
        logger: Any | None = None,
    ):
        self.settings = settings
        self.storage = storage
        self.index = index
        self.logger = logger

    def retrieve(
        self,
        query: str,
        document_ids: Sequence[str],
        *,
        filters: dict[str, Any] | None = None,
        restrict_document_ids: bool = True,
    ) -> list[RetrievedSource]:
        if filters or not restrict_document_ids:
            dense_results = self.index.search(
                query,
                k=self.settings.dense_search_k,
                document_ids=document_ids if restrict_document_ids else None,
                filters=filters,
            )
        else:
            dense_results = self.index.search(
                query, k=self.settings.dense_search_k, document_ids=document_ids
            )
        lexical_results = (
            (
                self.storage.lexical_search(
                    query,
                    k=self.settings.lexical_search_k,
                    document_ids=document_ids if restrict_document_ids else None,
                    filters=filters,
                )
                if filters or not restrict_document_ids
                else self.storage.lexical_search(
                    query,
                    k=self.settings.lexical_search_k,
                    document_ids=document_ids,
                )
            )
            if self.settings.hybrid_search
            else []
        )

        child_parents: dict[str, str] = {}
        child_fusion_scores: dict[str, float] = {}
        parent_dense_scores: dict[str, float] = {}

        for rank, (child, score) in enumerate(dense_results, start=1):
            parent_id = str(child.metadata.get("parent_id", ""))
            if not parent_id:
                continue
            child_id = str(
                child.metadata.get("child_id") or f"dense_{rank}_{parent_id}"
            )
            numeric_score = max(0.0, min(1.0, float(score)))
            child_parents[child_id] = parent_id
            child_fusion_scores[child_id] = child_fusion_scores.get(child_id, 0.0) + (
                self.settings.dense_weight / (self.settings.rrf_k + rank)
            )
            parent_dense_scores[parent_id] = max(
                numeric_score, parent_dense_scores.get(parent_id, 0.0)
            )

        for rank, child in enumerate(lexical_results, start=1):
            child_id = str(child.get("child_id", ""))
            parent_id = str(child.get("parent_id", ""))
            if not child_id or not parent_id:
                continue
            child_parents[child_id] = parent_id
            child_fusion_scores[child_id] = child_fusion_scores.get(child_id, 0.0) + (
                self.settings.lexical_weight / (self.settings.rrf_k + rank)
            )

        parent_fusion_scores: dict[str, float] = {}
        best_child_ids: dict[str, str] = {}
        for child_id, fusion_score in child_fusion_scores.items():
            parent_id = child_parents[child_id]
            # Lexical retrieval chỉ rerank parent đã vượt qua dense evidence gate.
            if parent_dense_scores.get(parent_id, 0.0) < self.settings.relevance_threshold:
                continue
            if fusion_score > parent_fusion_scores.get(parent_id, 0.0):
                parent_fusion_scores[parent_id] = fusion_score
                best_child_ids[parent_id] = child_id

        all_ranked_ids = sorted(
            parent_fusion_scores,
            key=lambda parent_id: (
                parent_fusion_scores[parent_id],
                parent_dense_scores.get(parent_id, 0.0),
            ),
            reverse=True,
        )
        parents = self.storage.parents_by_ids(all_ranked_ids)

        # Bridge thay thế các parent một trang nằm đúng hai trang biên để prompt
        # không nhận cùng một bằng chứng hai lần. Tiếp tục quét để luôn đủ top-k.
        selected: list[dict[str, Any]] = []
        for parent_id in all_ranked_ids:
            parent = parents.get(parent_id)
            if not parent:
                continue
            page = int(parent["page"])
            page_end = int(parent.get("page_end") or page)
            kind = str(parent.get("kind") or "page")
            if kind == "page" and any(
                item["document_id"] == parent["document_id"]
                and str(item.get("kind") or "page") == "bridge"
                and int(item["page"]) <= page <= int(item.get("page_end") or item["page"])
                for item in selected
            ):
                continue
            if kind == "bridge":
                selected = [
                    item
                    for item in selected
                    if not (
                        item["document_id"] == parent["document_id"]
                        and str(item.get("kind") or "page") == "page"
                        and page
                        <= int(item["page"])
                        <= page_end
                    )
                ]
            selected.append(parent)
            if len(selected) >= self.settings.parent_search_k:
                break

        ranked_ids = [str(parent["id"]) for parent in selected]
        sources: list[RetrievedSource] = []
        for number, parent_id in enumerate(ranked_ids, start=1):
            parent = parents.get(parent_id)
            if not parent:
                continue
            sources.append(
                RetrievedSource(
                    id=str(number),
                    document_id=parent["document_id"],
                    filename=parent["filename"],
                    page=int(parent["page"]),
                    page_end=int(parent.get("page_end") or parent["page"]),
                    parent_id=parent_id,
                    text=parent["text"],
                    # API cũ tiếp tục nhận dense cosine relevance trong [0, 1].
                    score=parent_dense_scores[parent_id],
                    kind=str(parent.get("kind") or "page"),
                    source_type=str(parent.get("source_type") or "pdf"),
                    ticker=parent.get("ticker"),
                    analysis_date=parent.get("analysis_date"),
                    analysis_cutoff=parent.get("analysis_cutoff"),
                    digest_section=parent.get("digest_section"),
                    liquidity_rank=parent.get("liquidity_rank"),
                    research_mode=str(parent.get("research_mode") or "legacy_unknown"),
                    data_provenance=str(
                        parent.get("data_provenance") or "legacy_unknown"
                    ),
                    content_kind=str(
                        parent.get("content_kind") or "legacy_unknown"
                    ),
                    report_schema=str(
                        parent.get("report_schema") or "legacy_unknown"
                    ),
                    report_stage=parent.get("report_stage"),
                    section_status=parent.get("section_status"),
                    full_report_hash=parent.get("full_report_hash"),
                    pipeline_version=parent.get("pipeline_version"),
                    execution_generation=parent.get("execution_generation"),
                    contains_decision_content=bool(
                        parent.get("contains_decision_content")
                    ),
                )
            )
            if self.logger:
                self.logger.info(
                    "parent retrieved",
                    extra={
                        "event": "parent_retrieved",
                        "document_id": parent["document_id"],
                        "chunk_ids": [best_child_ids.get(parent_id, ""), parent_id],
                        "score": round(parent_dense_scores[parent_id], 4),
                        "fusion_score": round(parent_fusion_scores[parent_id], 6),
                    },
                )
        return sources


def format_context(sources: Sequence[RetrievedSource]) -> str:
    """XML-escape indexed evidence so it cannot close/open prompt tags."""

    blocks = []
    for source in sources:
        if source.source_type == "trading_digest":
            attributes = (
                f'ticker="{escape(source.ticker or "", quote=True)}" '
                f'analysis_date="{escape(source.analysis_date or "", quote=True)}" '
                f'cutoff="{escape(source.analysis_cutoff or "", quote=True)}" '
                f'section="{escape(source.digest_section or "", quote=True)}" '
                f'content_kind="{escape(source.content_kind, quote=True)}" '
                f'report_stage="{escape(source.report_stage or "", quote=True)}" '
                f'section_status="{escape(source.section_status or "", quote=True)}" '
                f'research_mode="{escape(source.research_mode, quote=True)}" '
                f'data_provenance="{escape(source.data_provenance, quote=True)}"'
            )
        else:
            attributes = (
                f'filename="{escape(source.filename, quote=True)}" '
                f'page_start="{source.page}" page_end="{source.resolved_page_end}"'
            )
        blocks.append(
            f'<source id="[{escape(source.id)}]" type="{source.source_type}" '
            f"{attributes}>\n{escape(source.text)}\n</source>"
        )
    return "\n\n".join(blocks)
