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

    def public(self, cited: bool = False) -> dict[str, Any]:
        return {
            "id": self.id,
            "document_id": self.document_id,
            "filename": self.filename,
            "page": self.page,
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
        child_results = self.index.search(
            query, k=self.settings.child_search_k, document_ids=document_ids
        )
        parent_scores: dict[str, float] = {}
        best_child_ids: dict[str, str] = {}
        for child, score in child_results:
            parent_id = str(child.metadata.get("parent_id", ""))
            numeric_score = max(0.0, min(1.0, float(score)))
            if parent_id and numeric_score >= self.settings.relevance_threshold:
                if numeric_score > parent_scores.get(parent_id, 0.0):
                    parent_scores[parent_id] = numeric_score
                    best_child_ids[parent_id] = str(child.metadata.get("child_id", ""))

        ranked_ids = sorted(parent_scores, key=parent_scores.get, reverse=True)[
            : self.settings.parent_search_k
        ]
        parents = self.storage.parents_by_ids(ranked_ids)
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
                    parent_id=parent_id,
                    text=parent["text"],
                    score=parent_scores[parent_id],
                )
            )
            if self.logger:
                self.logger.info(
                    "parent retrieved",
                    extra={
                        "event": "parent_retrieved",
                        "document_id": parent["document_id"],
                        "chunk_ids": [best_child_ids.get(parent_id, ""), parent_id],
                        "score": round(parent_scores[parent_id], 4),
                    },
                )
        return sources


def format_context(sources: Sequence[RetrievedSource]) -> str:
    """XML-escape dữ liệu PDF để nó không thể đóng/mở thẻ prompt."""

    blocks = []
    for source in sources:
        blocks.append(
            f'<source id="[{escape(source.id)}]" '
            f'filename="{escape(source.filename, quote=True)}" page="{source.page}">\n'
            f"{escape(source.text)}\n"
            "</source>"
        )
    return "\n\n".join(blocks)
