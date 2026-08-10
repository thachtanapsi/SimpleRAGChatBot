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

