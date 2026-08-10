"""Child retrieval, parent grouping và context/citation an toàn."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from html import escape
from typing import Any, Sequence

from .config import Settings
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

    @property
    def resolved_page_end(self) -> int:
        return self.page if self.page_end is None else self.page_end

    def public(self, cited: bool = False) -> dict[str, Any]:
        return {
            "id": self.id,
            "document_id": self.document_id,
            "filename": self.filename,
            "page": self.page,
            "page_end": self.resolved_page_end,
            "parent_id": self.parent_id,
            "snippet": self.text[:500],
            "score": round(self.score, 4),
            "cited": cited,
        }


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

    def retrieve(self, query: str, document_ids: Sequence[str]) -> list[RetrievedSource]:
        dense_results = self.index.search(
            query, k=self.settings.dense_search_k, document_ids=document_ids
        )
        lexical_results = (
            self.storage.lexical_search(
                query, k=self.settings.lexical_search_k, document_ids=document_ids
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
    """XML-escape dữ liệu PDF để nó không thể đóng/mở thẻ prompt."""

    blocks = []
    for source in sources:
        blocks.append(
            f'<source id="[{escape(source.id)}]" '
            f'filename="{escape(source.filename, quote=True)}" '
            f'page_start="{source.page}" page_end="{source.resolved_page_end}">\n'
            f"{escape(source.text)}\n"
            "</source>"
        )
    return "\n\n".join(blocks)
