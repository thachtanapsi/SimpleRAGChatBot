from __future__ import annotations

import json
from dataclasses import replace
from typing import Any

import pytest

import rag_app.runtime as runtime_module
from rag_app.digests import build_digest_chunks
from rag_app.graph_tool import GraphRetrievalTool
from rag_app.orchestration import (
    AdaptiveEvidenceGrader,
    AdaptiveQueryPlanner,
    AgenticOrchestrator,
    DeterministicAnswerRequest,
    EvidenceView,
    HardFilters,
    QueryPlan,
    RequestScope,
    SoftFilters,
    ToolRequest,
)
from rag_app.reranking import (
    RerankResult,
    RerankerUnavailableError,
)
from rag_app.runtime import Runtime
from rag_app.storage import SQLiteStorage
from rag_app.structured_tool import StructuredDecisionTool


def _scope(
    document_ids: list[str],
    *,
    filters: HardFilters | None = None,
    explicit: bool = False,
) -> RequestScope:
    return RequestScope(
        allowed_document_ids=tuple(document_ids),
        document_ids_explicit=explicit,
        filters=filters or HardFilters(),
    )


def _tool_request(
    document_ids: list[str],
    *,
    latest_only: bool = False,
    filters: HardFilters | None = None,
) -> ToolRequest:
    return ToolRequest(
        query="quyết định HPG mới nhất",
        original_query="quyết định HPG mới nhất",
        scope=_scope(document_ids, filters=filters),
        limit=20,
        round=1,
        intent="decision",
        exact_terms=["HPG"],
        soft_filters=SoftFilters(
            ticker=None,
            section=None,
            date_from=None,
            date_to=None,
            latest_only=latest_only,
        ),
        required_facets=["rating", "price_target"],
    )


def _decision(*, status: str = "complete", target: int = 77000) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "status": status,
        "warnings": [] if status == "complete" else ["trading_reasoning_missing"],
        "portfolio": {
            "rating": "Overweight",
            "executive_summary": "Tích lũy theo ba đợt.",
            "investment_thesis": "Tăng trưởng lợi nhuận hỗ trợ định giá.",
            "time_horizon": "3-6 months",
            "price_target_status": "available",
            "price_target": target,
            "price_target_currency": "VND",
            "price_target_rationale": "Dựa trên tăng trưởng lợi nhuận.",
            "price_target_unavailable_reason": None,
        },
        "trading": {
            "action": "Buy",
            "reasoning": (
                "Định giá và động lượng hỗ trợ."
                if status == "complete"
                else None
            ),
            "entry_price": 70700,
            "stop_loss": 66500,
            "position_sizing": "3-5% danh mục",
        },
    }


def _insert_structured_document(
    storage: SQLiteStorage,
    *,
    document_id: str,
    ticker: str,
    analysis_date: str,
    status: str = "complete",
    target: int = 77000,
) -> str:
    decision = _decision(status=status, target=target)
    decision_json = json.dumps(
        decision,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    parent_id = f"decision-{document_id}"
    cutoff = f"{analysis_date}T16:45:00+07:00"
    with storage.connect() as db:
        db.execute(
            """
            INSERT INTO documents(
                id, filename, sha256, stored_path, size_bytes, page_count,
                status, index_fingerprint, collection_name, source_type,
                source_key, content_sha256, ticker, analysis_date,
                analysis_cutoff, research_mode, data_provenance, content_kind,
                report_schema, full_report_hash, pipeline_version,
                execution_generation, contains_decision_content,
                created_at, updated_at
            ) VALUES (
                ?, ?, ?, '', ?, 0, 'ready', 'index-v1', 'test',
                'trading_digest', ?, ?, ?, ?, ?, 'eod', 'eod_cutoff',
                'full_report_v1', 'full_report_v1', ?, 'pipeline-v1', 1, 1,
                ?, ?
            )
            """,
            (
                document_id,
                f"{ticker}-{analysis_date}.json",
                f"sha-{document_id}",
                len(decision_json.encode("utf-8")),
                f"source-{document_id}",
                f"content-{document_id}",
                ticker,
                analysis_date,
                cutoff,
                f"full-{document_id}",
                cutoff,
                cutoff,
            ),
        )
        db.execute(
            """
            INSERT INTO parents(
                id, document_id, page, page_end, kind, digest_section,
                report_stage, section_status, start_index, text
            ) VALUES (?, ?, 0, 0, 'page', 'decision.structured',
                      'decision', ?, 0, ?)
            """,
            (parent_id, document_id, status, decision_json),
        )
        db.execute(
            """
            INSERT INTO full_report_payloads(
                document_id, canonical_json, canonical_sha256,
                decision_json, decision_sha256, decision_status,
                projection_version, decision_index_version,
                created_at, updated_at
            ) VALUES (?, '{}', ?, ?, ?, ?, 2, 2, ?, ?)
            """,
            (
                document_id,
                f"canonical-{document_id}",
                decision_json,
                f"decision-{document_id}",
                status,
                cutoff,
                cutoff,
            ),
        )
    return parent_id


@pytest.fixture
def advanced_storage(settings) -> SQLiteStorage:
    storage = SQLiteStorage(settings.sqlite_path)
    storage.initialise()
    return storage


@pytest.mark.asyncio
async def test_structured_tool_latest_is_narrowed_inside_caller_scope(
    advanced_storage: SQLiteStorage,
):
    _insert_structured_document(
        advanced_storage,
        document_id="old-hpg",
        ticker="HPG",
        analysis_date="2026-08-20",
        target=75000,
    )
    _insert_structured_document(
        advanced_storage,
        document_id="new-hpg",
        ticker="HPG",
        analysis_date="2026-08-21",
        target=77000,
    )
    tool = StructuredDecisionTool(advanced_storage)

    latest = await tool.retrieve(
        _tool_request(["old-hpg", "new-hpg"], latest_only=True)
    )
    explicitly_old = await tool.retrieve(
        _tool_request(["old-hpg"], latest_only=True)
    )

    assert [item.document_id for item in latest] == ["new-hpg"]
    assert [item.document_id for item in explicitly_old] == ["old-hpg"]
    assert all(item.score == 0.0 for item in (*latest, *explicitly_old))


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("status", "expected_qualification"),
    [("complete", None), ("partial", "chưa hoàn chỉnh")],
)
async def test_structured_tool_renders_complete_and_qualified_partial_decisions(
    advanced_storage: SQLiteStorage,
    status: str,
    expected_qualification: str | None,
):
    document_id = f"hpg-{status}"
    _insert_structured_document(
        advanced_storage,
        document_id=document_id,
        ticker="HPG",
        analysis_date="2026-08-21",
        status=status,
    )
    tool = StructuredDecisionTool(advanced_storage)
    sources = await tool.retrieve(_tool_request([document_id]))
    assert len(sources) == 1
    source = sources[0]
    request = DeterministicAnswerRequest(
        question="Quyết định đầu tư HPG là gì?",
        scope=_scope([document_id]),
        plan=QueryPlan(
            standalone_query="Quyết định đầu tư HPG",
            intent="decision",
            routes=["structured"],
            subqueries=[],
            exact_terms=["HPG"],
            ticker="HPG",
            section="decision.structured",
            date_from=None,
            date_to=None,
            latest_only=True,
            required_facets=["rating", "price_target"],
        ),
        evidence=[
            EvidenceView(
                id="1",
                document_id=source.document_id,
                text=source.text,
                score=source.score,
            )
        ],
    )

    rendered = await tool.deterministic_answer(request)

    assert rendered is not None
    assert rendered.status == status
    assert "Overweight [1]" in rendered.generated.answer
    assert "77000 VND [1]" in rendered.generated.answer
    target_claim = next(
        claim for claim in rendered.generated.claims if claim.numeric == "77000"
    )
    assert target_claim.unit == "VND"
    if expected_qualification is None:
        assert rendered.qualification is None
    else:
        assert expected_qualification in str(rendered.qualification)


@pytest.mark.asyncio
async def test_structured_tool_uses_real_chunked_decision_parents(
    advanced_storage: SQLiteStorage,
    settings,
):
    document_id = "hpg-chunked"
    _insert_structured_document(
        advanced_storage,
        document_id=document_id,
        ticker="HPG",
        analysis_date="2026-08-21",
    )
    document = advanced_storage.get_document(document_id)
    payload = advanced_storage.get_full_report_payload(document_id)
    chunk_settings = replace(
        settings,
        parent_chunk_size=300,
        parent_chunk_overlap=100,
    )
    parents, _, _, _ = build_digest_chunks(
        document=document,
        sections=[
            {
                "section": "decision.structured",
                "position": 0,
                "report_stage": "decision",
                "section_status": "complete",
                "text": payload["decision_json"],
            }
        ],
        settings=chunk_settings,
    )
    assert 1 < len(parents) <= 5
    with advanced_storage.connect() as db:
        db.execute("DELETE FROM parents WHERE document_id=?", (document_id,))
        db.executemany(
            """
            INSERT INTO parents(
                id, document_id, page, page_end, kind, digest_section,
                report_stage, section_status, start_index, text
            ) VALUES (
                :id, :document_id, :page, :page_end, :kind, :digest_section,
                :report_stage, :section_status, :start_index, :text
            )
            """,
            parents,
        )

    tool = StructuredDecisionTool(advanced_storage)
    sources = await tool.retrieve(_tool_request([document_id]))
    assert len(sources) == len(parents)
    assert all(item.text != payload["decision_json"] for item in sources)
    rendered = await tool.deterministic_answer(
        DeterministicAnswerRequest(
            question="Quyết định HPG?",
            scope=_scope([document_id]),
            plan=QueryPlan(
                standalone_query="Quyết định HPG",
                intent="decision",
                routes=["structured"],
                subqueries=[],
                exact_terms=["HPG"],
                ticker="HPG",
                section="decision.structured",
                date_from=None,
                date_to=None,
                latest_only=True,
                required_facets=["rating", "price_target"],
            ),
            evidence=[
                EvidenceView(
                    id=item.id,
                    document_id=item.document_id,
                    text=item.text,
                    score=item.score,
                )
                for item in sources
            ],
        )
    )

    assert rendered is not None
    assert rendered.status == "complete"
    assert "Overweight [" in rendered.generated.answer
    assert "77000 VND [" in rendered.generated.answer


@pytest.mark.asyncio
async def test_structured_deterministic_answer_rejects_evidence_outside_scope(
    advanced_storage: SQLiteStorage,
):
    _insert_structured_document(
        advanced_storage,
        document_id="outside",
        ticker="HPG",
        analysis_date="2026-08-21",
    )
    source = (await StructuredDecisionTool(advanced_storage).retrieve(
        _tool_request(["outside"])
    ))[0]
    request = DeterministicAnswerRequest(
        question="Quyết định HPG?",
        scope=_scope(["inside"]),
        plan=QueryPlan(
            standalone_query="Quyết định HPG",
            intent="decision",
            routes=["structured"],
            subqueries=[],
            exact_terms=["HPG"],
            ticker="HPG",
            section=None,
            date_from=None,
            date_to=None,
            latest_only=True,
            required_facets=[],
        ),
        evidence=[
            EvidenceView(
                id="1",
                document_id=source.document_id,
                text=source.text,
                score=0.0,
            )
        ],
    )

    assert await StructuredDecisionTool(advanced_storage).deterministic_answer(request) is None


def _graph_row(parent_id: str, document_id: str, text: str) -> dict[str, Any]:
    return {
        "parent_id": parent_id,
        "document_id": document_id,
        "filename": f"{document_id}.json",
        "page": 0,
        "page_end": 0,
        "kind": "page",
        "text": text,
        "source_type": "trading_digest",
        "ticker": "HPG",
        "analysis_date": "2026-08-21",
        "analysis_cutoff": "2026-08-21T16:45:00+07:00",
        "digest_section": "risks_unknowns",
        "research_mode": "eod",
        "data_provenance": "eod_cutoff",
        "content_kind": "daily_digest_v1",
        "report_schema": "daily_digest_v1",
        "section_status": "complete",
        "score": 0.0,
        "graph_score": 0.75,
    }


class _GraphStorage:
    def __init__(self):
        self.latest_calls: list[list[str]] = []

    def latest_ready_document_ids(self, values):
        self.latest_calls.append(list(values))
        return [value for value in values if value.startswith("new")]


class _GraphRetriever:
    def __init__(self, rows: list[dict[str, Any]]):
        self.storage = _GraphStorage()
        self.rows = rows
        self.calls: list[dict[str, Any]] = []

    def search(self, query: str, **kwargs):
        self.calls.append({"query": query, **kwargs})
        allowed = set(kwargs["document_ids"])
        return [row for row in self.rows if row["document_id"] in allowed]


class _GraphReranker:
    def __init__(self):
        self.calls: list[dict[str, Any]] = []

    def rerank(self, query, candidates, *, top_k, min_score):
        self.calls.append(
            {
                "query": query,
                "candidate_ids": [item.id for item in candidates],
                "top_k": top_k,
                "min_score": min_score,
            }
        )
        return [RerankResult(id=candidates[-1].id, score=0.91)]


@pytest.mark.asyncio
async def test_graph_tool_applies_scoped_latest_then_reranks_raw_parent_evidence():
    retriever = _GraphRetriever(
        [
            _graph_row("p-old", "old-hpg", "old"),
            _graph_row("p-new-a", "new-hpg-a", "first"),
            _graph_row("p-new-b", "new-hpg-b", "second"),
        ]
    )
    reranker = _GraphReranker()
    tool = GraphRetrievalTool(
        retriever,
        reranker=reranker,
        child_k=2,
        min_score=0.6,
    )

    sources = await tool.retrieve(
        _tool_request(["old-hpg", "new-hpg-a", "new-hpg-b"], latest_only=True)
    )

    assert retriever.storage.latest_calls == [
        ["old-hpg", "new-hpg-a", "new-hpg-b"]
    ]
    assert retriever.calls == [
        {
            "query": "quyết định HPG mới nhất",
            "k": 20,
            "document_ids": ["new-hpg-a", "new-hpg-b"],
            "filters": None,
        }
    ]
    assert reranker.calls == [
        {
            "query": "quyết định HPG mới nhất",
            "candidate_ids": ["p-new-a", "p-new-b"],
            "top_k": 2,
            "min_score": 0.6,
        }
    ]
    assert [(item.parent_id, item.text, item.score) for item in sources] == [
        ("p-new-b", "second", 0.0)
    ]
    diagnostics = sources[0].public(explain=True)["explain"]["retrieval"]
    assert diagnostics["graph_score"] == 0.75
    assert diagnostics["rerank_score"] == 0.91


@pytest.mark.asyncio
async def test_graph_tool_propagates_fail_closed_reranker_error():
    retriever = _GraphRetriever([_graph_row("p-new", "new-hpg", "evidence")])

    class UnavailableReranker:
        def rerank(self, *_args, **_kwargs):
            raise RerankerUnavailableError("missing local cache")

    tool = GraphRetrievalTool(retriever, reranker=UnavailableReranker())
    with pytest.raises(RerankerUnavailableError):
        await tool.retrieve(_tool_request(["new-hpg"]))


class _Logger:
    def info(self, *_args, **_kwargs):
        pass

    def warning(self, *_args, **_kwargs):
        pass


class _EmbeddingService:
    def __init__(self, *_args, **_kwargs):
        self.revision = "embedding-commit"
        self.device = "cpu"
        self.dimension = 1024
        self.truncated_texts = 0


class _Index:
    def __init__(self, *_args, **_kwargs):
        pass


class _Reranker:
    def __init__(self, *_args, **_kwargs):
        self.available = True


class _LLM:
    def __init__(self, **kwargs):
        self.model = kwargs["model"]
        self.kwargs = kwargs

    def with_structured_output(self, *_args, **_kwargs):
        return self


class _GraphService:
    def __init__(self, storage, **kwargs):
        self.storage = storage
        self.kwargs = kwargs
        self.reconciled = False

    def reconcile(self, *, apply):
        self.reconciled = apply

    async def stop(self):
        pass


class _RuntimeGraphRetriever:
    def __init__(self, storage, **kwargs):
        self.storage = storage
        self.kwargs = kwargs


class _Ingestion:
    def __init__(self, *_args, **_kwargs):
        pass

    def reconcile_vector_metadata_updates(self):
        pass

    def seed_papers_once(self):
        pass


class _Worker:
    def __init__(self, *_args, **_kwargs):
        pass

    async def stop(self):
        pass


@pytest.mark.asyncio
async def test_runtime_wires_bounded_three_tool_agentic_orchestrator(
    settings, monkeypatch
):
    configured = replace(
        settings,
        hybrid_search=True,
        rerank_enabled=True,
        agentic_enabled=True,
        self_check_enabled=True,
        graph_build_enabled=True,
        graph_enabled=True,
        answer_num_predict=2048,
        advanced_answer_num_predict=1024,
    )
    configured.validate()
    monkeypatch.setattr(runtime_module, "configure_logging", lambda *_args: _Logger())
    monkeypatch.setattr(runtime_module, "EmbeddingService", _EmbeddingService)
    monkeypatch.setattr(runtime_module, "ChromaIndex", _Index)
    monkeypatch.setattr(runtime_module, "RerankerService", _Reranker)
    monkeypatch.setattr(runtime_module, "ChatOllama", _LLM)
    monkeypatch.setattr(runtime_module, "GraphService", _GraphService)
    monkeypatch.setattr(runtime_module, "GraphRetriever", _RuntimeGraphRetriever)
    monkeypatch.setattr(runtime_module, "IngestionService", _Ingestion)
    monkeypatch.setattr(runtime_module, "IngestionWorker", _Worker)

    async def check_ollama(runtime):
        runtime.ollama_ready = True
        runtime.review_model_ready = True
        runtime.graph_extractor_model_ready = True
        runtime.ollama_model_digests = {
            configured.chat_model: "chat-digest",
            configured.graph_extractor_model: "graph-digest",
        }
        return True

    monkeypatch.setattr(Runtime, "check_ollama", check_ollama)
    runtime = Runtime(configured)

    await runtime.start(start_worker=False)
    try:
        orchestrator = runtime.chat.orchestrator
        assert isinstance(orchestrator, AgenticOrchestrator)
        assert set(orchestrator.tools) == {"hybrid", "structured", "graph"}
        assert orchestrator.allowed_tools == ("hybrid", "structured", "graph")
        assert isinstance(orchestrator.planner, AdaptiveQueryPlanner)
        assert isinstance(orchestrator.grader, AdaptiveEvidenceGrader)
        assert orchestrator.planner.delegate.llm.model == configured.review_model
        assert orchestrator.grader.delegate.llm.model == configured.review_model
        assert orchestrator.grader.delegate.llm.kwargs["num_predict"] == min(
            configured.answer_num_predict, 256
        )
        assert runtime.chat.llm.kwargs["num_predict"] == configured.answer_num_predict
        assert orchestrator.planner.delegate.llm.kwargs["num_predict"] == min(
            configured.answer_num_predict, 256
        )
        assert runtime.chat.advanced_answer_llm.kwargs["num_predict"] == 1024
        assert runtime.chat.advanced_answer_num_predict == 1024
        assert (
            runtime.graph_service.kwargs["extractor"].structured_llm.kwargs[
                "num_predict"
            ]
            == min(configured.answer_num_predict, 1536)
        )
        graph_tool = orchestrator.tools["graph"]
        assert graph_tool.reranker is runtime.reranker
        assert graph_tool.retriever is runtime.graph_retriever
        assert runtime.graph_service.reconciled is True
    finally:
        await runtime.stop()
