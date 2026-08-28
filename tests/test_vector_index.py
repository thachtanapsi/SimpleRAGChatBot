from __future__ import annotations

from rag_app.vector_index import ChromaIndex


class FakeEmbeddings:
    def __init__(self):
        self.document_calls = 0

    @staticmethod
    def _vector(text: str):
        lower = text.lower()
        return [float("apple" in lower), float("banana" in lower), 0.1]

    def embed_documents(self, texts):
        self.document_calls += 1
        return [self._vector(text) for text in texts]

    def embed_query(self, text):
        return self._vector(text)


def test_chroma_survives_restart_without_reembedding(settings):
    first_embeddings = FakeEmbeddings()
    first = ChromaIndex(settings.chroma_dir, settings.collection_name, first_embeddings)
    first.add_children(
        ids=["c1", "c2"],
        texts=["apple facts", "banana facts"],
        metadatas=[
            {"document_id": "doc1", "parent_id": "p1"},
            {"document_id": "doc1", "parent_id": "p2"},
        ],
    )
    assert first.count() == 2
    assert first_embeddings.document_calls == 1

    restarted_embeddings = FakeEmbeddings()
    restarted = ChromaIndex(settings.chroma_dir, settings.collection_name, restarted_embeddings)
    assert restarted.count() == 2
    assert restarted_embeddings.document_calls == 0
    results = restarted.search("apple", k=2, document_ids=["doc1"])
    assert results[0][0].metadata["parent_id"] == "p1"

    restarted.delete(["c1", "c2"])
    assert restarted.count() == 0


def test_chroma_uses_digest_metadata_filters_without_document_id_list(settings):
    index = ChromaIndex(settings.chroma_dir, "digest_filter_test", FakeEmbeddings())
    index.add_children(
        ids=["hpg-risk", "vcb-market"],
        texts=["apple liquidity risk", "apple market strength"],
        metadatas=[
            {
                "document_id": "digest-hpg",
                "parent_id": "parent-hpg",
                "source_type": "trading_digest",
                "ticker": "HPG",
                "analysis_date_ordinal": 739484,
                "liquidity_rank": 3,
                "digest_section": "risks_unknowns",
                "research_mode": "historical",
                "data_provenance": "historical_replay",
            },
            {
                "document_id": "digest-vcb",
                "parent_id": "parent-vcb",
                "source_type": "trading_digest",
                "ticker": "VCB",
                "analysis_date_ordinal": 739484,
                "liquidity_rank": 8,
                "digest_section": "market",
                "research_mode": "eod",
                "data_provenance": "eod_cutoff",
            },
        ],
    )

    results = index.search(
        "apple",
        k=5,
        document_ids=None,
        filters={
            "source_types": ["trading_digest"],
            "tickers": ["HPG"],
            "liquidity_rank_max": 5,
            "digest_sections": ["risks_unknowns"],
            "research_modes": ["historical"],
            "data_provenances": ["historical_replay"],
        },
    )

    assert [item.metadata["ticker"] for item, _score in results] == ["HPG"]


def test_chroma_backfills_missing_legacy_provenance_without_reembedding(settings):
    first_embeddings = FakeEmbeddings()
    first = ChromaIndex(settings.chroma_dir, "legacy_provenance", first_embeddings)
    first.add_children(
        ids=["legacy", "historical"],
        texts=["apple legacy digest", "apple historical digest"],
        metadatas=[
            {
                "document_id": "legacy-document",
                "parent_id": "legacy-parent",
                "source_type": "trading_digest",
            },
            {
                "document_id": "historical-document",
                "parent_id": "historical-parent",
                "source_type": "trading_digest",
                "research_mode": "historical",
                "data_provenance": "historical_replay",
            },
        ],
    )

    restarted_embeddings = FakeEmbeddings()
    restarted = ChromaIndex(
        settings.chroma_dir, "legacy_provenance", restarted_embeddings
    )
    results = restarted.search(
        "apple",
        k=5,
        filters={
            "research_modes": ["legacy_unknown"],
            "data_provenances": ["legacy_unknown"],
            "content_kinds": ["daily_digest_v1"],
            "report_stages": ["digest"],
        },
    )

    assert restarted_embeddings.document_calls == 0
    assert [item.metadata["document_id"] for item, _score in results] == [
        "legacy-document"
    ]
    assert results[0][0].metadata["research_mode"] == "legacy_unknown"
    assert results[0][0].metadata["data_provenance"] == "legacy_unknown"
    assert results[0][0].metadata["content_kind"] == "daily_digest_v1"
    assert results[0][0].metadata["report_schema"] == "daily_digest_v1"
    assert results[0][0].metadata["report_stage"] == "digest"
    assert results[0][0].metadata["section_status"] == "complete"
    historical = restarted.client.get_collection("legacy_provenance").get(
        ids=["historical"], include=["metadatas"]
    )
    assert historical["metadatas"][0]["research_mode"] == "historical"
    assert historical["metadatas"][0]["data_provenance"] == "historical_replay"
    assert historical["metadatas"][0]["content_kind"] == "daily_digest_v1"

    assert restarted.update_provenance(
        ["legacy"], "live", "request_cutoff", "legacy_provenance"
    )
    relabelled = restarted.search(
        "apple",
        k=5,
        filters={
            "research_modes": ["live"],
            "data_provenances": ["request_cutoff"],
        },
    )
    assert restarted_embeddings.document_calls == 0
    assert [item.metadata["document_id"] for item, _score in relabelled] == [
        "legacy-document"
    ]


def test_chroma_filters_full_report_content_kind_and_stage(settings):
    index = ChromaIndex(settings.chroma_dir, "full_report_filters", FakeEmbeddings())
    index.add_children(
        ids=["daily-risk", "full-decision"],
        texts=["apple daily risk", "apple portfolio decision"],
        metadatas=[
            {
                "document_id": "digest-hpg",
                "parent_id": "daily-parent",
                "source_type": "trading_digest",
                "content_kind": "daily_digest_v1",
                "report_stage": "digest",
                "digest_section": "risks_unknowns",
            },
            {
                "document_id": "digest-hpg",
                "parent_id": "full-parent",
                "source_type": "trading_digest",
                "content_kind": "full_report_v1",
                "report_stage": "risk",
                "digest_section": "portfolio.decision",
            },
        ],
    )

    results = index.search(
        "apple",
        k=5,
        filters={
            "content_kinds": ["full_report_v1"],
            "report_stages": ["risk"],
            "digest_sections": ["portfolio.decision"],
        },
    )

    assert [item.metadata["parent_id"] for item, _score in results] == [
        "full-parent"
    ]
