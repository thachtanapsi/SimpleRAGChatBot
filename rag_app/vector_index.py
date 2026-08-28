"""Chroma persistent index chứa duy nhất child chunks."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any, Sequence

import chromadb
from chromadb.config import Settings as ChromaSettings
from langchain_chroma import Chroma


LEGACY_PROVENANCE = "legacy_unknown"
LEGACY_METADATA_BACKFILL_SIZE = 1000


class ChromaIndex:
    def __init__(self, persist_dir: Path, collection_name: str, embeddings: Any):
        self.collection_name = collection_name
        self.embeddings = embeddings
        self.client = chromadb.PersistentClient(
            path=str(persist_dir),
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self.store = self._open(collection_name)
        self._backfill_legacy_provenance()

    def _open(self, name: str) -> Chroma:
        return Chroma(
            client=self.client,
            collection_name=name,
            embedding_function=self.embeddings,
            collection_metadata={"hnsw:space": "cosine"},
        )

    def _backfill_legacy_provenance(self) -> int:
        """Label legacy vector metadata without recomputing embeddings.

        SQLite already migrates old documents to ``legacy_unknown``.  Chroma
        metadata is independent, however, and an equality filter does not
        match a missing key.  Updating the existing metadata in place keeps
        those vectors retrievable by the explicit legacy filter while leaving
        every already-labelled pair unchanged.
        """

        collection = self.client.get_collection(self.collection_name)
        updated = 0
        offset = 0
        while True:
            page = collection.get(
                include=["metadatas"],
                limit=LEGACY_METADATA_BACKFILL_SIZE,
                offset=offset,
            )
            ids = list(page.get("ids") or [])
            metadatas = list(page.get("metadatas") or [])
            if not ids:
                break
            legacy_ids: list[str] = []
            legacy_metadatas: list[dict[str, Any]] = []
            for item_id, raw_metadata in zip(ids, metadatas, strict=True):
                metadata = dict(raw_metadata or {})
                changed = False
                if "research_mode" not in metadata:
                    metadata["research_mode"] = LEGACY_PROVENANCE
                    changed = True
                if "data_provenance" not in metadata:
                    metadata["data_provenance"] = LEGACY_PROVENANCE
                    changed = True
                source_type = str(metadata.get("source_type") or "pdf")
                if "content_kind" not in metadata:
                    metadata["content_kind"] = (
                        "daily_digest_v1"
                        if source_type == "trading_digest"
                        else "pdf"
                    )
                    changed = True
                if "report_schema" not in metadata:
                    metadata["report_schema"] = metadata["content_kind"]
                    changed = True
                if source_type == "trading_digest":
                    if "report_stage" not in metadata:
                        metadata["report_stage"] = "digest"
                        changed = True
                    if "section_status" not in metadata:
                        metadata["section_status"] = "complete"
                        changed = True
                    if "contains_decision_content" not in metadata:
                        metadata["contains_decision_content"] = False
                        changed = True
                if changed:
                    legacy_ids.append(str(item_id))
                    legacy_metadatas.append(metadata)
            if legacy_ids:
                collection.update(ids=legacy_ids, metadatas=legacy_metadatas)
                updated += len(legacy_ids)
            offset += len(ids)
        return updated

    def add_children(
        self,
        *,
        ids: Sequence[str],
        texts: Sequence[str],
        metadatas: Sequence[dict[str, Any]],
    ) -> None:
        if ids:
            self.store.add_texts(texts=list(texts), metadatas=list(metadatas), ids=list(ids))

    def update_provenance(
        self,
        ids: Sequence[str],
        research_mode: str,
        data_provenance: str,
        collection_name: str | None = None,
    ) -> bool:
        """Update vector metadata in place and verify every expected child.

        Chroma metadata is replaced as a whole, so each existing dictionary is
        read and preserved before adding the two provenance fields.  Returning
        ``False`` leaves the durable SQLite outbox unacknowledged for retry.
        """

        if not ids:
            return False
        name = collection_name or self.collection_name
        existing_collections = {
            item if isinstance(item, str) else item.name
            for item in self.client.list_collections()
        }
        if name not in existing_collections:
            return False
        collection = self.client.get_collection(name)
        expected_ids = list(dict.fromkeys(str(item) for item in ids))
        for start in range(0, len(expected_ids), LEGACY_METADATA_BACKFILL_SIZE):
            batch_ids = expected_ids[start : start + LEGACY_METADATA_BACKFILL_SIZE]
            current = collection.get(ids=batch_ids, include=["metadatas"])
            found_ids = [str(item) for item in current.get("ids") or []]
            found_metadata = list(current.get("metadatas") or [])
            if set(found_ids) != set(batch_ids):
                return False
            metadata_by_id = {
                item_id: dict(metadata or {})
                for item_id, metadata in zip(
                    found_ids, found_metadata, strict=True
                )
            }
            replacements = []
            for item_id in batch_ids:
                metadata = metadata_by_id[item_id]
                metadata["research_mode"] = research_mode
                metadata["data_provenance"] = data_provenance
                replacements.append(metadata)
            collection.update(ids=batch_ids, metadatas=replacements)
            verified = collection.get(ids=batch_ids, include=["metadatas"])
            if set(str(item) for item in verified.get("ids") or []) != set(batch_ids):
                return False
            if any(
                (metadata or {}).get("research_mode") != research_mode
                or (metadata or {}).get("data_provenance") != data_provenance
                for metadata in verified.get("metadatas") or []
            ):
                return False
        return True

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
        self,
        query: str,
        *,
        k: int,
        document_ids: Sequence[str] | None = None,
        filters: dict[str, Any] | None = None,
    ) -> list[tuple[Any, float]]:
        if document_ids is not None and not document_ids:
            return []
        conditions: list[dict[str, Any]] = []
        if document_ids is not None:
            if len(document_ids) == 1:
                conditions.append({"document_id": document_ids[0]})
            else:
                conditions.append({"document_id": {"$in": list(document_ids)}})
        configured = filters or {}
        for field, metadata_field in (
            ("source_types", "source_type"),
            ("tickers", "ticker"),
            ("batch_ids", "batch_id"),
            ("digest_sections", "digest_section"),
            ("research_modes", "research_mode"),
            ("data_provenances", "data_provenance"),
            ("content_kinds", "content_kind"),
            ("report_stages", "report_stage"),
        ):
            values = list(dict.fromkeys(configured.get(field) or []))
            if len(values) == 1:
                conditions.append({metadata_field: values[0]})
            elif values:
                conditions.append({metadata_field: {"$in": values}})
        if configured.get("date_from"):
            conditions.append(
                {
                    "analysis_date_ordinal": {
                        "$gte": date.fromisoformat(configured["date_from"]).toordinal()
                    }
                }
            )
        if configured.get("date_to"):
            conditions.append(
                {
                    "analysis_date_ordinal": {
                        "$lte": date.fromisoformat(configured["date_to"]).toordinal()
                    }
                }
            )
        if configured.get("liquidity_rank_min") is not None:
            conditions.append(
                {"liquidity_rank": {"$gte": int(configured["liquidity_rank_min"])}}
            )
        if configured.get("liquidity_rank_max") is not None:
            conditions.append(
                {"liquidity_rank": {"$lte": int(configured["liquidity_rank_max"])}}
            )
        if not conditions:
            # LangChain/Chroma reject an empty filter; no filter means all rows.
            return self.store.similarity_search_with_relevance_scores(query=query, k=k)
        where = conditions[0] if len(conditions) == 1 else {"$and": conditions}
        return self.store.similarity_search_with_relevance_scores(
            query=query,
            k=k,
            filter=where,
        )

    def count(self) -> int:
        return self.client.get_collection(self.collection_name).count()

    def healthcheck(self) -> bool:
        return self.client.heartbeat() > 0
