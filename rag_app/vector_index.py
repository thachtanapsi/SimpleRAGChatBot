"""Chroma persistent index chứa duy nhất child chunks."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import chromadb
from chromadb.config import Settings as ChromaSettings
from langchain_chroma import Chroma


class ChromaIndex:
    def __init__(self, persist_dir: Path, collection_name: str, embeddings: Any):
        self.collection_name = collection_name
        self.embeddings = embeddings
        self.client = chromadb.PersistentClient(
            path=str(persist_dir),
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self.store = self._open(collection_name)

    def _open(self, name: str) -> Chroma:
        return Chroma(
            client=self.client,
            collection_name=name,
            embedding_function=self.embeddings,
            collection_metadata={"hnsw:space": "cosine"},
        )

    def add_children(
        self,
        *,
        ids: Sequence[str],
        texts: Sequence[str],
        metadatas: Sequence[dict[str, Any]],
    ) -> None:
        if ids:
            self.store.add_texts(texts=list(texts), metadatas=list(metadatas), ids=list(ids))

    def delete(self, ids: Sequence[str], collection_name: str | None = None) -> None:
        if not ids:
            return
        name = collection_name or self.collection_name
        existing = {
            item if isinstance(item, str) else item.name
            for item in self.client.list_collections()
        }
        if name in existing:
            self._open(name).delete(ids=list(ids))

    def search(
        self, query: str, *, k: int, document_ids: Sequence[str]
    ) -> list[tuple[Any, float]]:
        if not document_ids:
            return []
        where: dict[str, Any]
        if len(document_ids) == 1:
            where = {"document_id": document_ids[0]}
        else:
            where = {"document_id": {"$in": list(document_ids)}}
        return self.store.similarity_search_with_relevance_scores(
            query=query,
            k=k,
            filter=where,
        )

    def count(self) -> int:
        return self.client.get_collection(self.collection_name).count()

    def healthcheck(self) -> bool:
        return self.client.heartbeat() > 0
