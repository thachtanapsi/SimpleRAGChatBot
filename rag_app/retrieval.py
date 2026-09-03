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
from .reranking import RerankCandidate, RerankResult, RerankerUnavailableError
from .storage import SQLiteStorage
from .vector_index import ChromaIndex


@dataclass(frozen=True, slots=True)
class RetrievalDiagnostics:
    """Điểm theo từng channel; chỉ xuất ra API khi caller yêu cầu explain."""

    child_id: str
    dense_rank: int | None
    lexical_rank: int | None
    dense_score: float | None
    lexical_score: float | None
    fusion_score: float
    rerank_score: float | None
    graph_score: float | None = None

    def public(self) -> dict[str, Any]:
        payload = asdict(self)
        for key in (
            "dense_score",
            "lexical_score",
            "fusion_score",
            "rerank_score",
            "graph_score",
        ):
            value = payload[key]
            if value is not None:
                payload[key] = round(float(value), 6)
        return payload


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
    diagnostics: RetrievalDiagnostics | None = None

    @property
    def resolved_page_end(self) -> int:
        return self.page if self.page_end is None else self.page_end

    def public(self, cited: bool = False, *, explain: bool = False) -> dict[str, Any]:
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
        if explain and self.diagnostics is not None:
            payload["explain"] = {"retrieval": self.diagnostics.public()}
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
        reranker: Any | None = None,
    ):
        self.settings = settings
        self.storage = storage
        self.index = index
        self.logger = logger
        self.reranker = reranker

    def retrieve(
        self,
        query: str,
        document_ids: Sequence[str],
        *,
        filters: dict[str, Any] | None = None,
        restrict_document_ids: bool = True,
        semantic_query: str | None = None,
        lexical_query: str | None = None,
        exact_terms: Sequence[str] | None = None,
    ) -> list[RetrievedSource]:
        dense_query = (semantic_query or query).strip()
        keyword_query = self._lexical_query(lexical_query or query, exact_terms)
        if filters or not restrict_document_ids:
            dense_results = self.index.search(
                dense_query,
                k=int(getattr(self.settings, "dense_search_k", 30)),
                document_ids=document_ids if restrict_document_ids else None,
                filters=filters,
            )
        else:
            dense_results = self.index.search(
                dense_query,
                k=int(getattr(self.settings, "dense_search_k", 30)),
                document_ids=document_ids,
            )
        lexical_results = (
            (
                self.storage.lexical_search(
                    keyword_query,
                    k=int(getattr(self.settings, "lexical_search_k", 30)),
                    document_ids=document_ids if restrict_document_ids else None,
                    filters=filters,
                )
                if filters or not restrict_document_ids
                else self.storage.lexical_search(
                    keyword_query,
                    k=int(getattr(self.settings, "lexical_search_k", 30)),
                    document_ids=document_ids,
                )
            )
            if bool(getattr(self.settings, "hybrid_search", False))
            else []
        )

        candidates: dict[str, _ChildCandidate] = {}
        parent_dense_scores: dict[str, float] = {}
        dense_weight = float(getattr(self.settings, "dense_weight", 0.7))
        lexical_weight = float(getattr(self.settings, "lexical_weight", 0.3))
        rrf_k = int(getattr(self.settings, "rrf_k", 60))
        relevance_threshold = float(
            getattr(self.settings, "relevance_threshold", 0.35)
        )

        for rank, (child, score) in enumerate(dense_results, start=1):
            parent_id = str(child.metadata.get("parent_id", ""))
            if not parent_id:
                continue
            child_id = str(
                child.metadata.get("child_id") or f"dense_{rank}_{parent_id}"
            )
            numeric_score = max(0.0, min(1.0, float(score)))
            candidate = candidates.setdefault(
                child_id,
                _ChildCandidate(
                    id=child_id,
                    parent_id=parent_id,
                    text=str(child.page_content or ""),
                ),
            )
            if candidate.parent_id != parent_id:
                continue
            candidate.text = candidate.text or str(child.page_content or "")
            candidate.dense_rank = rank
            candidate.dense_score = numeric_score
            candidate.fusion_score += dense_weight / (rrf_k + rank)
            parent_dense_scores[parent_id] = max(
                numeric_score, parent_dense_scores.get(parent_id, 0.0)
            )

        for rank, child in enumerate(lexical_results, start=1):
            child_id = str(child.get("child_id", ""))
            parent_id = str(child.get("parent_id", ""))
            if not child_id or not parent_id:
                continue
            candidate = candidates.setdefault(
                child_id,
                _ChildCandidate(
                    id=child_id,
                    parent_id=parent_id,
                    text=str(child.get("text") or ""),
                ),
            )
            if candidate.parent_id != parent_id:
                continue
            candidate.text = candidate.text or str(child.get("text") or "")
            candidate.lexical_rank = rank
            lexical_score = child.get("lexical_score")
            candidate.lexical_score = (
                float(lexical_score) if lexical_score is not None else None
            )
            candidate.fusion_score += lexical_weight / (rrf_k + rank)

        # True union: dense candidate cần qua relevance gate; mọi lexical hit đã
        # được FTS giới hạn đều có thể cứu một dense miss.
        eligible = [
            item
            for item in candidates.values()
            if item.lexical_rank is not None
            or (item.dense_score or 0.0) >= relevance_threshold
        ]
        eligible.sort(key=_candidate_rrf_sort_key)

        rerank_enabled = bool(getattr(self.settings, "rerank_enabled", False))
        if rerank_enabled:
            if self.reranker is None:
                raise RerankerUnavailableError(
                    "Reranker được bật nhưng chưa được khởi tạo"
                )
            candidate_k = int(getattr(self.settings, "rerank_candidate_k", 40))
            child_k = int(getattr(self.settings, "rerank_child_k", 12))
            min_score = float(getattr(self.settings, "rerank_min_score", 0.5))
            eligible = eligible[:candidate_k]

            # Storage cũ không trả child text cho BM25. Ưu tiên c.text khi có và
            # chỉ fallback parent text để vẫn tương thích index đã tồn tại.
            candidate_parents = self.storage.parents_by_ids(
                list(dict.fromkeys(item.parent_id for item in eligible))
            )
            rerank_inputs: list[RerankCandidate] = []
            valid_candidates: dict[str, _ChildCandidate] = {}
            for item in eligible:
                parent = candidate_parents.get(item.parent_id)
                if not parent:
                    continue
                text = item.text or str(parent.get("text") or "")
                if not text:
                    continue
                item.text = text
                valid_candidates[item.id] = item
                rerank_inputs.append(RerankCandidate(id=item.id, text=text))
            reranked = self.reranker.rerank(
                dense_query,
                rerank_inputs,
                top_k=child_k,
                min_score=min_score,
            )
            eligible = []
            for result in reranked:
                result_id, result_score = _rerank_result_values(result)
                candidate = valid_candidates.get(result_id)
                if candidate is None:
                    continue
                candidate.rerank_score = result_score
                eligible.append(candidate)

        # Child được xếp hạng trước, rồi mới group parent.
        best_candidates: dict[str, _ChildCandidate] = {}
        all_ranked_ids: list[str] = []
        for candidate in eligible:
            if candidate.parent_id in best_candidates:
                continue
            best_candidates[candidate.parent_id] = candidate
            all_ranked_ids.append(candidate.parent_id)
        parents = self.storage.parents_by_ids(all_ranked_ids)

        # Bridge thay thế các parent một trang nằm đúng hai trang biên để prompt
        # không nhận cùng một bằng chứng hai lần. Tiếp tục quét để luôn đủ top-k.
        selected: list[dict[str, Any]] = []
        parent_search_k = int(getattr(self.settings, "parent_search_k", 5))
        if rerank_enabled:
            parent_search_k = min(parent_search_k, 5)
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
            if len(selected) >= parent_search_k:
                break

        ranked_ids = [str(parent["id"]) for parent in selected]
        sources: list[RetrievedSource] = []
        for number, parent_id in enumerate(ranked_ids, start=1):
            parent = parents.get(parent_id)
            if not parent:
                continue
            best = best_candidates[parent_id]
            dense_score = parent_dense_scores.get(parent_id, 0.0)
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
                    # Lexical-only evidence không giả tạo dense score.
                    score=dense_score,
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
                    diagnostics=RetrievalDiagnostics(
                        child_id=best.id,
                        dense_rank=best.dense_rank,
                        lexical_rank=best.lexical_rank,
                        dense_score=best.dense_score,
                        lexical_score=best.lexical_score,
                        fusion_score=best.fusion_score,
                        rerank_score=best.rerank_score,
                    ),
                )
            )
            if self.logger:
                self.logger.info(
                    "parent retrieved",
                    extra={
                        "event": "parent_retrieved",
                        "document_id": parent["document_id"],
                        "chunk_ids": [best.id, parent_id],
                        "score": round(dense_score, 4),
                        "fusion_score": round(best.fusion_score, 6),
                        "rerank_score": (
                            round(best.rerank_score, 6)
                            if best.rerank_score is not None
                            else None
                        ),
                    },
                )
        return sources

    @staticmethod
    def _lexical_query(
        query: str, exact_terms: Sequence[str] | None
    ) -> str:
        parts = [query.strip()]
        seen = {parts[0].casefold()} if parts[0] else set()
        for raw_term in exact_terms or ():
            term = str(raw_term).strip()
            if term and term.casefold() not in seen:
                parts.append(term)
                seen.add(term.casefold())
        return " ".join(item for item in parts if item)


@dataclass(slots=True)
class _ChildCandidate:
    id: str
    parent_id: str
    text: str
    dense_rank: int | None = None
    lexical_rank: int | None = None
    dense_score: float | None = None
    lexical_score: float | None = None
    fusion_score: float = 0.0
    rerank_score: float | None = None


def _candidate_rrf_sort_key(candidate: _ChildCandidate) -> tuple[Any, ...]:
    return (
        -candidate.fusion_score,
        -(candidate.dense_score or 0.0),
        candidate.dense_rank if candidate.dense_rank is not None else 10**9,
        candidate.lexical_rank if candidate.lexical_rank is not None else 10**9,
        candidate.id,
    )


def _rerank_result_values(result: Any) -> tuple[str, float]:
    """Cho phép service thật và test double nhỏ dùng cùng interface."""

    if isinstance(result, RerankResult):
        return result.id, max(0.0, min(1.0, float(result.score)))
    if isinstance(result, dict):
        return str(result["id"]), max(
            0.0, min(1.0, float(result["score"]))
        )
    result_id, score = result
    return str(result_id), max(0.0, min(1.0, float(score)))


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
