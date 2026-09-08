from __future__ import annotations

import asyncio
from typing import Any

import pytest
from langchain_core.messages import AIMessage

from rag_app.errors import (
    AgentTimeoutError,
    RetrievalToolsUnavailableError,
    ReviewModelUnavailableError,
)
from rag_app.orchestration import (
    AdaptiveEvidenceGrader,
    AdaptiveQueryPlanner,
    AgenticOrchestrator,
    ConversationTurn,
    DeterministicDecision,
    EvidenceGrade,
    EvidenceView,
    GradingRequest,
    HybridRetrievalTool,
    LLMEvidenceGrader,
    LLMQueryPlanner,
    OLLAMA_EVIDENCE_GRADE_SCHEMA,
    OLLAMA_QUERY_PLAN_SCHEMA,
    PlanningRequest,
    QueryPlan,
    RequestScope,
    SoftFilters,
    ToolRequest,
)
from rag_app.reranking import RerankerUnavailableError
from rag_app.retrieval import RetrievedSource
from rag_app.self_check import (
    GeneratedAnswer,
    VerificationIssue,
    VerificationReport,
    audit_generated_answer,
)


def assert_ollama_schema_is_flat(value: Any) -> None:
    if isinstance(value, dict):
        assert not ({"$defs", "$ref", "anyOf"} & value.keys())
        for child in value.values():
            assert_ollama_schema_is_flat(child)
    elif isinstance(value, list):
        for child in value:
            assert_ollama_schema_is_flat(child)


class StructuredReviewModel:
    def __init__(self, response: Any):
        self.response = response
        self.schema: dict[str, Any] | None = None
        self.method: str | None = None
        self.include_raw: bool | None = None
        self.messages: list[Any] | None = None

    def with_structured_output(
        self,
        schema: dict[str, Any],
        *,
        method: str,
        include_raw: bool,
    ) -> "StructuredReviewModel":
        self.schema = schema
        self.method = method
        self.include_raw = include_raw
        return self

    async def ainvoke(self, messages: list[Any]) -> Any:
        self.messages = messages
        return self.response


def planning_request() -> PlanningRequest:
    return PlanningRequest(
        question="FPT tăng trưởng thế nào?",
        history=[],
        allowed_tools=["hybrid"],
        correction_round=0,
        prior_grade=None,
    )


def grading_request() -> GradingRequest:
    return GradingRequest(
        question="FPT tăng trưởng thế nào?",
        intent="lookup",
        required_facets=["growth"],
        evidence=[
            EvidenceView(
                id="1",
                document_id="doc1",
                text="FPT tăng trưởng 20% trong Q2/2026.",
                score=0.9,
            )
        ],
        round=1,
    )


def plan(**updates: object) -> QueryPlan:
    payload: dict[str, object] = {
        "standalone_query": "FPT tăng trưởng Q2/2026",
        "intent": "lookup",
        "routes": ["hybrid"],
        "subqueries": [],
        "exact_terms": ["FPT", "Q2/2026"],
        "ticker": "FPT",
        "section": None,
        "date_from": None,
        "date_to": None,
        "latest_only": True,
        "required_facets": ["growth"],
    }
    payload.update(updates)
    return QueryPlan.model_validate(payload, strict=True)


def source(
    source_id: str = "old",
    *,
    document_id: str = "doc1",
    parent_id: str | None = None,
    text: str = "FPT tăng trưởng 20% trong Q2/2026.",
) -> RetrievedSource:
    return RetrievedSource(
        id=source_id,
        document_id=document_id,
        filename="report.pdf",
        page=1,
        parent_id=parent_id or f"parent-{source_id}",
        text=text,
        score=0.9,
    )


def scope(filters=None) -> RequestScope:
    return RequestScope.from_request(
        ["doc1", "doc2"],
        document_ids_explicit=True,
        filters=filters,
    )


def answer(citations: list[str] | None = None) -> GeneratedAnswer:
    citation_ids = citations or ["1"]
    labels = ", ".join(citation_ids)
    return GeneratedAnswer(
        answer=f"FPT tăng trưởng 20% trong Q2/2026 [{labels}].",
        claims=[
            {
                "text": "FPT tăng trưởng 20% trong Q2/2026.",
                "citation_ids": citation_ids,
                "ticker": "FPT",
                "numeric": "20%",
                "unit": "%",
                "period": "Q2/2026",
                "date": None,
            }
        ],
    )


async def generate(_question, _sources, _feedback):
    return answer()


async def verify(_question, generated, sources):
    return audit_generated_answer(generated, sources)


class RecordingTool:
    name = "hybrid"

    def __init__(self, outputs=None, error: Exception | None = None):
        self.outputs = outputs if outputs is not None else [source()]
        self.error = error
        self.requests: list[ToolRequest] = []

    async def retrieve(self, request: ToolRequest):
        self.requests.append(request)
        if self.error:
            raise self.error
        return self.outputs


class MalformedPlanner:
    async def plan(self, _request):
        return {"routes": ["graph"], "scope": {"document_ids": ["other"]}}


@pytest.mark.parametrize(
    "schema",
    [OLLAMA_QUERY_PLAN_SCHEMA, OLLAMA_EVIDENCE_GRADE_SCHEMA],
)
def test_review_ollama_schemas_are_flat(schema):
    assert_ollama_schema_is_flat(schema)


@pytest.mark.asyncio
async def test_planner_uses_raw_json_schema_wrapper_and_parses_valid_payload():
    expected = plan()
    model = StructuredReviewModel(
        {
            "raw": AIMessage(
                content="",
                response_metadata={"done_reason": "stop"},
            ),
            "parsed": expected.model_dump(mode="json", exclude_none=True),
            "parsing_error": None,
        }
    )

    result = await LLMQueryPlanner(model).plan(planning_request())

    assert result == expected
    assert model.schema == OLLAMA_QUERY_PLAN_SCHEMA
    assert model.method == "json_schema"
    assert model.include_raw is True
    assert model.messages is not None


@pytest.mark.asyncio
async def test_planner_truncation_takes_precedence_over_parsing_error():
    model = StructuredReviewModel(
        {
            "raw": AIMessage(
                content="{",
                response_metadata={"done_reason": "length"},
            ),
            "parsed": None,
            "parsing_error": ValueError("invalid JSON"),
        }
    )

    with pytest.raises(ReviewModelUnavailableError):
        await LLMQueryPlanner(model).plan(planning_request())


@pytest.mark.asyncio
async def test_grader_parsing_error_is_review_model_unavailable():
    model = StructuredReviewModel(
        {
            "raw": AIMessage(
                content="{}",
                response_metadata={"done_reason": "stop"},
            ),
            "parsed": None,
            "parsing_error": ValueError("missing required fields"),
        }
    )
    grader = LLMEvidenceGrader(model)

    with pytest.raises(ReviewModelUnavailableError):
        await grader.grade(grading_request())

    assert model.schema == OLLAMA_EVIDENCE_GRADE_SCHEMA
    assert model.method == "json_schema"
    assert model.include_raw is True


class UnexpectedPlanner:
    def __init__(self):
        self.calls = 0

    async def plan(self, _request):
        self.calls += 1
        raise AssertionError("exact known ticker should not call the LLM planner")

    async def correct(self, _request):
        self.calls += 1
        return plan()


@pytest.mark.asyncio
async def test_adaptive_planner_uses_exact_ready_ticker_without_model_call():
    delegate = UnexpectedPlanner()
    planner = AdaptiveQueryPlanner(delegate, lambda: ["FPT", "HPG"])

    result = await planner.plan(
        PlanningRequest(
            question="HPG chịu tác động bởi những rủi ro nào mới nhất?",
            history=[],
            allowed_tools=["hybrid", "structured", "graph"],
        )
    )

    assert result.standalone_query == "HPG chịu tác động bởi những rủi ro nào mới nhất?"
    assert result.intent == "lookup"
    assert result.routes == ["hybrid"]
    assert result.subqueries == []
    assert result.ticker == "HPG"
    assert result.section == "risks_unknowns"
    assert result.latest_only is True
    assert result.exact_terms == ["HPG", "rủi ro"]
    assert result.required_facets == ["rủi ro"]
    assert delegate.calls == 0

    corrected = await planner.correct(
        PlanningRequest(
            question="HPG chịu tác động bởi những rủi ro nào mới nhất?",
            history=[],
            allowed_tools=["hybrid", "structured", "graph"],
            correction_round=1,
            prior_grade="insufficient",
        )
    )
    assert corrected.standalone_query == "HPG rủi ro"
    assert corrected.routes == ["hybrid"]
    assert corrected.ticker == "HPG"
    assert corrected.section == "risks_unknowns"
    assert delegate.calls == 0


@pytest.mark.parametrize(
    ("question", "ticker"),
    [
        ("thông tin giá POW", "POW"),
        ("Giá cổ phiếu HPG hiện tại?", "HPG"),
        ("giá FPT là bao nhiêu?", "FPT"),
        ("POW giá bao nhiêu?", "POW"),
    ],
)
@pytest.mark.asyncio
async def test_simple_price_lookup_uses_one_query_despite_unrelated_history(question, ticker):
    delegate = UnexpectedPlanner()
    planner = AdaptiveQueryPlanner(delegate, lambda: ["POW", "HPG", "FPT"])

    result = await planner.plan(
        PlanningRequest(
            question=question,
            history=[
                ConversationTurn(role="user", content="tóm tắt vẻ đẹp phú yên"),
                ConversationTurn(role="assistant", content="Phú Yên có nhiều cảnh đẹp [1]."),
            ],
            allowed_tools=["hybrid", "structured", "graph"],
        )
    )

    assert result.standalone_query == question
    assert result.intent == "lookup"
    assert result.routes == ["hybrid"]
    assert result.subqueries == []
    assert result.ticker == ticker
    assert result.section is None
    assert result.latest_only is True
    assert result.required_facets == ["giá"]
    assert result.exact_terms == [ticker, "giá"]
    assert delegate.calls == 0


@pytest.mark.parametrize(
    "question",
    [
        "giá POW ngày 26/08/2026",
        "giá POW trong tháng trước",
        "dự báo giá POW tuần tới",
        "so sánh giá POW và FPT",
        "giá POW và VNM",
        "đánh giá POW",
        "giá của mã đó",
        "thông tin giá UNKNOWN",
    ],
)
@pytest.mark.asyncio
async def test_price_shortcut_delegates_questions_needing_more_planning(question):
    expected = plan(standalone_query=question)
    model = StructuredReviewModel(expected.model_dump(mode="json"))
    planner = AdaptiveQueryPlanner(LLMQueryPlanner(model), lambda: ["POW", "FPT"])

    result = await planner.plan(
        PlanningRequest(question=question, history=[], allowed_tools=["hybrid"])
    )

    assert model.messages is not None
    assert result == expected


@pytest.mark.asyncio
async def test_price_lookup_can_use_semantic_correction_when_evidence_is_missing():
    delegate = UnexpectedPlanner()
    planner = AdaptiveQueryPlanner(delegate, lambda: ["POW"])

    result = await planner.correct(
        PlanningRequest(
            question="thông tin giá POW",
            history=[],
            allowed_tools=["hybrid"],
            correction_round=1,
            prior_grade="insufficient",
        )
    )

    assert result == plan()
    assert delegate.calls == 1


@pytest.mark.asyncio
async def test_price_lookup_keeps_budget_for_grading_generation_and_verification():
    # The live planner produced these equivalent searches for this question
    # after a Phu Yen exchange. Each search reranks the same corpus on CPU.
    model = StructuredReviewModel(
        plan(
            standalone_query="giá POW",
            subqueries=["điều tra giá hiện tại của POW"],
            ticker=None,
            exact_terms=["POW"],
            required_facets=[],
        ).model_dump(mode="json")
    )
    price_text = "POW đóng cửa ở mức 13,300 VND ngày 26/08/2026."

    class PriceTool(RecordingTool):
        async def retrieve(self, request):
            await asyncio.sleep(0.55)
            return await super().retrieve(request)

    class PriceGrader:
        async def grade(self, request):
            await asyncio.sleep(0.2)
            assert request.evidence[0].text == price_text
            return EvidenceGrade(verdict="sufficient", reason="price_found", suggested_queries=[])

    async def generate_price(_question, _sources, _feedback):
        await asyncio.sleep(0.55)
        return GeneratedAnswer(
            answer=f"{price_text} [1]",
            claims=[{"text": price_text, "citation_ids": ["1"]}],
        )

    tool = PriceTool(outputs=[source(text=price_text)])
    selected_scope = RequestScope.from_request(["doc1"], document_ids_explicit=True)
    result = await AgenticOrchestrator(
        [tool],
        planner=AdaptiveQueryPlanner(LLMQueryPlanner(model), lambda: ["POW"]),
        grader=PriceGrader(),
        timeout_seconds=1.5,
    ).run(
        question="thông tin giá POW",
        scope=selected_scope,
        history=[{"role": "user", "content": "tóm tắt vẻ đẹp phú yên"}],
        generate=generate_price,
        verify=verify,
        abstention="insufficient",
    )

    assert result.abstained is False
    assert result.answer == f"{price_text} [1]"
    assert len(tool.requests) == 1
    assert tool.requests[0].scope == selected_scope
    assert tool.requests[0].soft_filters.latest_only is False
    assert [event.stage for event in result.events] == [
        "validate", "plan", "retrieve", "grade", "generate", "verify", "finalize"
    ]


@pytest.mark.asyncio
async def test_adaptive_grader_checks_simple_facet_without_model_call():
    class UnexpectedGrader:
        async def grade(self, _request):
            raise AssertionError("simple facet coverage must be deterministic")

    result = await AdaptiveEvidenceGrader(UnexpectedGrader()).grade(
        GradingRequest(
            question="HPG có rủi ro nào?",
            intent="lookup",
            ticker="HPG",
            required_facets=["rủi ro"],
            evidence=[
                EvidenceView(
                    id="1",
                    document_id="doc1",
                    text="Rủi ro chính của HPG là biến động tỷ giá.",
                    score=0.0,
                    source_type="trading_digest",
                    ticker="HPG",
                    content_kind="daily_digest_v1",
                    digest_section="risks_unknowns",
                    section_status="complete",
                    rerank_score=0.91,
                )
            ],
            round=1,
            latest_only=True,
        )
    )

    assert result.verdict == "sufficient"
    assert result.reason == "verified_latest_risk_section"


@pytest.mark.asyncio
async def test_adaptive_grader_delegates_when_risk_evidence_ticker_mismatches():
    class RecordingGrader:
        def __init__(self):
            self.calls = 0

        async def grade(self, _request):
            self.calls += 1
            return EvidenceGrade(
                verdict="insufficient",
                reason="ticker_mismatch",
                suggested_queries=[],
            )

    delegate = RecordingGrader()
    result = await AdaptiveEvidenceGrader(delegate).grade(
        GradingRequest(
            question="HPG có rủi ro nào?",
            intent="lookup",
            ticker="HPG",
            required_facets=["rủi ro"],
            evidence=[
                EvidenceView(
                    id="1",
                    document_id="doc1",
                    text="Rủi ro chính của FPT là tỷ giá.",
                    score=0.0,
                    source_type="trading_digest",
                    ticker="FPT",
                    content_kind="daily_digest_v1",
                    digest_section="risks_unknowns",
                    section_status="complete",
                    rerank_score=0.91,
                )
            ],
            round=1,
            latest_only=True,
        )
    )

    assert result.verdict == "insufficient"
    assert delegate.calls == 1


@pytest.mark.asyncio
async def test_adaptive_grader_rejects_missing_latest_risk_after_correction():
    class UnexpectedGrader:
        async def grade(self, _request):
            raise AssertionError("bounded correction must end deterministically")

    result = await AdaptiveEvidenceGrader(UnexpectedGrader()).grade(
        GradingRequest(
            question="HPG có rủi ro nào mới nhất?",
            intent="lookup",
            ticker="HPG",
            required_facets=["rủi ro"],
            evidence=[
                EvidenceView(
                    id="1",
                    document_id="doc1",
                    text="Tâm lý thị trường ở mức trung tính.",
                    score=0.0,
                    source_type="trading_digest",
                    ticker="HPG",
                    content_kind="daily_digest_v1",
                    digest_section="sentiment",
                    section_status="complete",
                    rerank_score=0.91,
                )
            ],
            round=2,
            latest_only=True,
        )
    )

    assert result.verdict == "insufficient"
    assert result.reason == "latest_risk_evidence_not_supported"


@pytest.mark.asyncio
async def test_adaptive_decision_grader_delegates_when_required_facet_is_missing():
    class RecordingGrader:
        def __init__(self):
            self.calls = 0

        async def grade(self, _request):
            self.calls += 1
            return EvidenceGrade(
                verdict="insufficient",
                reason="missing_risk_facet",
                suggested_queries=[],
            )

    delegate = RecordingGrader()
    result = await AdaptiveEvidenceGrader(delegate).grade(
        GradingRequest(
            question="Khuyến nghị và rủi ro cho HPG?",
            intent="decision",
            ticker="HPG",
            required_facets=["khuyến nghị", "rủi ro"],
            evidence=[
                EvidenceView(
                    id="1",
                    document_id="doc1",
                    text="Khuyến nghị: MUA.",
                    score=0.0,
                    source_type="trading_digest",
                    ticker="HPG",
                    content_kind="full_report_v1",
                    digest_section="decision.structured",
                    section_status="complete",
                    rerank_score=0.91,
                )
            ],
            round=1,
            latest_only=True,
        )
    )

    assert result.verdict == "insufficient"
    assert delegate.calls == 1


@pytest.mark.asyncio
async def test_malformed_planner_falls_back_without_widening_hard_scope():
    tool = RecordingTool(
        outputs=[source(), source("leak", document_id="not-allowed")]
    )
    orchestrator = AgenticOrchestrator(
        [tool], planner=MalformedPlanner(), max_retrieval_rounds=1
    )

    result = await orchestrator.run(
        question="FPT tăng bao nhiêu?",
        scope=scope({"tickers": ["FPT"], "batch_ids": ["batch-1"]}),
        generate=generate,
        verify=verify,
        abstention="insufficient",
    )

    assert result.abstained is False
    assert [item.document_id for item in result.sources] == ["doc1"]
    assert tool.requests[0].scope.allowed_document_ids == ("doc1", "doc2")
    assert tool.requests[0].scope.filters.batch_ids == ("batch-1",)
    assert result.events[1].outcome == "fallback"


class CorrectivePlanner:
    async def plan(self, _request):
        return plan()

    async def correct(self, _request):
        return plan(
            standalone_query="tăng trưởng",
            ticker=None,
            latest_only=False,
            exact_terms=[],
        )


class TwoRoundGrader:
    def __init__(self):
        self.calls = 0

    async def grade(self, _request):
        self.calls += 1
        if self.calls == 1:
            return EvidenceGrade(
                verdict="insufficient",
                reason="missing_facet",
                suggested_queries=["tăng trưởng FPT kỳ trước"],
            )
        return EvidenceGrade(
            verdict="sufficient",
            reason="facet_covered",
            suggested_queries=[],
        )


@pytest.mark.asyncio
async def test_corrective_round_drops_only_soft_hints_and_keeps_hard_scope():
    tool = RecordingTool()
    orchestrator = AgenticOrchestrator(
        [tool],
        planner=CorrectivePlanner(),
        grader=TwoRoundGrader(),
        max_retrieval_rounds=9,
        max_subqueries=9,
        max_tool_calls=9,
    )

    result = await orchestrator.run(
        question="FPT tăng bao nhiêu?",
        scope=RequestScope.from_request(
            ["doc1", "doc2"],
            document_ids_explicit=False,
            filters={
                "tickers": ["FPT", "HPG"],
                "content_kinds": ["full_report_v1"],
            },
        ),
        generate=generate,
        verify=verify,
        abstention="insufficient",
    )

    assert result.explanation.corrective_rounds == 1
    assert len(tool.requests) == 2
    first = tool.requests[0].effective_retrieval_filters()
    second = tool.requests[1].effective_retrieval_filters()
    assert first == {
        "tickers": ["FPT"],
        "content_kinds": ["full_report_v1"],
        "latest_only": True,
    }
    assert second == {
        "tickers": ["FPT", "HPG"],
        "content_kinds": ["full_report_v1"],
    }
    assert tool.requests[0].scope is tool.requests[1].scope


def test_corrective_latest_risk_preserves_explicit_ticker_and_freshness():
    request = ToolRequest(
        query="HPG rủi ro",
        original_query="HPG có rủi ro nào mới nhất?",
        scope=RequestScope.from_request(
            ["doc1", "doc2"], document_ids_explicit=False
        ),
        limit=5,
        round=2,
        intent="lookup",
        exact_terms=["HPG", "rủi ro"],
        soft_filters=SoftFilters(
            ticker="HPG",
            section="risks_unknowns",
            latest_only=True,
        ),
        required_facets=["rủi ro"],
    )

    assert request.effective_retrieval_filters() == {
        "tickers": ["HPG"],
        "latest_only": True,
    }


class FullFirstRoundTool:
    name = "hybrid"

    async def retrieve(self, request: ToolRequest):
        if request.round == 1:
            return [source(f"old-{index}") for index in range(5)]
        return [source("corrected")]


class CorrectedFacetGrader:
    async def grade(self, request):
        if any("corrective evidence" in item.text for item in request.evidence):
            return EvidenceGrade(
                verdict="sufficient",
                reason="facet_covered",
                suggested_queries=[],
            )
        return EvidenceGrade(
            verdict="insufficient",
            reason="missing_facet",
            suggested_queries=["corrective evidence"],
        )


@pytest.mark.asyncio
async def test_corrective_evidence_can_enter_a_full_five_parent_context():
    tool = FullFirstRoundTool()

    async def retrieve_with_distinct_text(request: ToolRequest):
        items = await FullFirstRoundTool.retrieve(tool, request)
        if request.round == 2:
            return [
                source(
                    "corrected",
                    text="corrective evidence: FPT tăng trưởng 20% trong Q2/2026.",
                )
            ]
        return items

    tool.retrieve = retrieve_with_distinct_text
    orchestrator = AgenticOrchestrator(
        [tool],
        planner=CorrectivePlanner(),
        grader=CorrectedFacetGrader(),
        max_retrieval_rounds=2,
        max_tool_calls=2,
    )

    result = await orchestrator.run(
        question="FPT tăng bao nhiêu?",
        scope=scope(),
        generate=generate,
        verify=verify,
        abstention="insufficient",
    )

    assert result.abstained is False
    assert len(result.sources) == 5
    assert result.sources[0].parent_id == "parent-corrected"


@pytest.mark.asyncio
async def test_all_retrieval_tools_failure_is_domain_503():
    tool = RecordingTool(error=ConnectionError("index unavailable"))
    orchestrator = AgenticOrchestrator([tool], max_retrieval_rounds=2)

    with pytest.raises(RetrievalToolsUnavailableError):
        await orchestrator.run(
            question="Question",
            scope=scope(),
            generate=generate,
            verify=verify,
            abstention="insufficient",
        )

    assert len(tool.requests) == 2


class RouteTool(RecordingTool):
    def __init__(self, name, **kwargs):
        super().__init__(**kwargs)
        self.name = name


@pytest.mark.asyncio
async def test_graph_failure_falls_back_to_hybrid_within_tool_budget():
    graph = RouteTool("graph", error=ConnectionError("graph unavailable"))
    hybrid = RouteTool("hybrid", outputs=[source()])
    orchestrator = AgenticOrchestrator(
        [graph, hybrid],
        allowed_tools=["hybrid", "graph"],
        planner=None,
        max_retrieval_rounds=1,
        max_tool_calls=4,
    )

    result = await orchestrator.run(
        question="FPT liên quan tới rủi ro nào?",
        scope=scope(),
        initial_plan=plan(
            intent="relationship",
            routes=["graph"],
            subqueries=["quan hệ", "rủi ro", "đối tác"],
            latest_only=False,
        ),
        generate=generate,
        verify=verify,
        abstention="insufficient",
    )

    assert result.abstained is False
    assert result.explanation.tools == ["graph", "hybrid"]
    assert len(graph.requests) == 3
    assert len(hybrid.requests) == 1


class RejectingVerifier:
    def __init__(self):
        self.calls = 0

    async def __call__(self, _question, _generated, _sources):
        self.calls += 1
        return VerificationReport(
            passed=False,
            issues=[
                VerificationIssue(
                    code="unsupported_claim", detail="not fully supported"
                )
            ],
            supported_citation_ids=[],
        )


@pytest.mark.asyncio
async def test_second_verification_failure_abstains_after_one_regeneration():
    tool = RecordingTool()
    verifier = RejectingVerifier()
    attempts: list[str | None] = []

    async def retrying_generator(_question, _sources, feedback):
        attempts.append(feedback)
        return answer()

    result = await AgenticOrchestrator([tool], max_answer_attempts=8).run(
        question="Question",
        scope=scope(),
        generate=retrying_generator,
        verify=verifier,
        abstention="insufficient",
    )

    assert result.abstained is True
    assert result.answer == "insufficient"
    assert result.sources == ()
    assert len(attempts) == 2
    assert attempts[0] is None and "not fully supported" in attempts[1]
    assert verifier.calls == 2
    assert result.explanation.verification_status == "abstained"


class StructuredDecisionTool:
    name = "structured"

    def __init__(self):
        self.decision_calls = 0

    async def retrieve(self, _request):
        return [source()]

    async def deterministic_answer(self, _request):
        self.decision_calls += 1
        return DeterministicDecision(
            generated=answer(),
            status="partial",
            qualification="Dữ liệu quyết định mới chỉ hoàn tất một phần.",
        )


class DecisionPlanner:
    async def plan(self, _request):
        return plan(intent="decision", routes=["structured"])


@pytest.mark.asyncio
async def test_structured_decision_bypasses_generator_and_semantic_verifier():
    tool = StructuredDecisionTool()

    async def forbidden(*_args):
        raise AssertionError("LLM path must be bypassed")

    result = await AgenticOrchestrator(
        [tool], planner=DecisionPlanner(), allowed_tools=["structured"]
    ).run(
        question="Quyết định cho FPT?",
        scope=scope(),
        generate=forbidden,
        verify=forbidden,
        abstention="insufficient",
    )

    assert result.abstained is False
    assert result.answer.startswith("Dữ liệu quyết định mới chỉ hoàn tất một phần.")
    assert result.explanation.strategy == "structured"
    assert result.explanation.verification_status == "qualified"
    assert tool.decision_calls == 1


class ConflictGrader:
    async def grade(self, _request):
        return EvidenceGrade(
            verdict="ambiguous",
            reason="source_conflict",
            suggested_queries=[],
            conflict_presented=True,
            conflict_source_ids=["1", "2"],
        )


@pytest.mark.asyncio
async def test_explicit_conflict_requires_citation_for_each_side():
    tool = RecordingTool(
        outputs=[
            source("a", text="FPT tăng trưởng 20% trong Q2/2026."),
            source("b", document_id="doc2", text="FPT tăng trưởng 20% trong Q2/2026."),
        ]
    )
    attempts = 0

    async def conflict_generator(_question, _sources, _feedback):
        nonlocal attempts
        attempts += 1
        return answer(["1"] if attempts == 1 else ["1", "2"])

    result = await AgenticOrchestrator(
        [tool], grader=ConflictGrader(), max_answer_attempts=2
    ).run(
        question="Hai nguồn nói gì?",
        scope=scope(),
        generate=conflict_generator,
        verify=verify,
        abstention="insufficient",
    )

    assert attempts == 2
    assert result.abstained is False
    assert result.explanation.confidence == "medium"
    assert result.explanation.verification_status == "qualified"


@pytest.mark.asyncio
async def test_tool_timeout_is_bounded_and_reported_as_all_tools_failure():
    class SlowTool:
        name = "hybrid"

        async def retrieve(self, _request):
            await asyncio.sleep(0.2)
            return [source()]

    orchestrator = AgenticOrchestrator(
        [SlowTool()],
        max_retrieval_rounds=1,
        timeout_seconds=1,
        tool_timeout_seconds=0.01,
    )
    with pytest.raises(RetrievalToolsUnavailableError):
        await orchestrator.run(
            question="Question",
            scope=scope(),
            generate=generate,
            verify=verify,
            abstention="insufficient",
        )


@pytest.mark.asyncio
async def test_hybrid_latest_only_narrows_ids_and_preserves_query_channels():
    class LatestStorage:
        documents = {
            "pdf": {"source_type": "pdf"},
            "old": {
                "source_type": "trading_digest",
                "ticker": "FPT",
                "analysis_date": "2026-08-26",
                "analysis_cutoff": "2026-08-26T15:00:00+07:00",
            },
            "new-digest": {
                "source_type": "trading_digest",
                "ticker": "FPT",
                "analysis_date": "2026-08-27",
                "analysis_cutoff": "2026-08-27T15:00:00+07:00",
            },
            "new-full": {
                "source_type": "trading_digest",
                "ticker": "FPT",
                "analysis_date": "2026-08-27",
                "analysis_cutoff": "2026-08-27T15:00:00+07:00",
            },
        }

        def get_document(self, document_id):
            return self.documents[document_id]

    class ChannelRetriever:
        def __init__(self):
            self.storage = LatestStorage()
            self.call = None

        def retrieve(
            self,
            query,
            document_ids,
            *,
            filters=None,
            restrict_document_ids=True,
            semantic_query=None,
            lexical_query=None,
            exact_terms=None,
        ):
            self.call = {
                "query": query,
                "document_ids": document_ids,
                "filters": filters,
                "restrict_document_ids": restrict_document_ids,
                "semantic_query": semantic_query,
                "lexical_query": lexical_query,
                "exact_terms": exact_terms,
            }
            return []

    retriever = ChannelRetriever()
    request = ToolRequest(
        query="semantic rewrite",
        original_query="FPT exact 20%",
        scope=RequestScope.from_request(
            ["pdf", "old", "new-digest", "new-full"],
            document_ids_explicit=False,
        ),
        limit=10,
        round=1,
        intent="lookup",
        exact_terms=["FPT", "20%"],
        soft_filters=SoftFilters(latest_only=True),
        required_facets=[],
    )

    await HybridRetrievalTool(retriever).retrieve(request)

    assert retriever.call == {
        "query": "semantic rewrite",
        "document_ids": ["pdf", "new-digest", "new-full"],
        "filters": None,
        "restrict_document_ids": True,
        "semantic_query": "semantic rewrite",
        "lexical_query": "FPT exact 20%",
        "exact_terms": ["FPT", "20%"],
    }


@pytest.mark.asyncio
async def test_reranker_unavailable_keeps_specific_domain_error():
    tool = RecordingTool(error=RerankerUnavailableError("offline"))
    orchestrator = AgenticOrchestrator([tool], max_retrieval_rounds=1)

    with pytest.raises(RerankerUnavailableError) as caught:
        await orchestrator.run(
            question="Question",
            scope=scope(),
            generate=generate,
            verify=verify,
            abstention="insufficient",
        )

    assert caught.value.error_code == "rag_reranker_unavailable"


@pytest.mark.asyncio
async def test_explicit_document_scope_disables_planner_latest_hint():
    tool = RecordingTool()

    class LatestPlanner:
        async def plan(self, _request):
            return plan(latest_only=True)

    await AgenticOrchestrator(
        [tool], planner=LatestPlanner(), max_retrieval_rounds=1
    ).run(
        question="FPT hiện tại?",
        scope=scope(),
        generate=generate,
        verify=verify,
        abstention="insufficient",
    )

    assert tool.requests[0].soft_filters.latest_only is False


@pytest.mark.asyncio
async def test_explicit_historical_question_disables_planner_latest_hint():
    tool = RecordingTool()

    class IncorrectLatestPlanner:
        async def plan(self, _request):
            return plan(latest_only=True)

    await AgenticOrchestrator(
        [tool], planner=IncorrectLatestPlanner(), max_retrieval_rounds=1
    ).run(
        question="FPT trong Q2/2024 ra sao?",
        scope=RequestScope.from_request(
            ["doc1", "doc2"], document_ids_explicit=False
        ),
        generate=generate,
        verify=verify,
        abstention="insufficient",
    )

    assert tool.requests[0].soft_filters.latest_only is False


@pytest.mark.asyncio
async def test_review_transport_failure_is_503_not_business_abstention():
    class BrokenReviewModel:
        async def ainvoke(self, _messages):
            raise ConnectionError("local model stopped")

    tool = RecordingTool()
    orchestrator = AgenticOrchestrator(
        [tool],
        planner=LLMQueryPlanner(BrokenReviewModel()),
        max_retrieval_rounds=1,
    )
    with pytest.raises(ReviewModelUnavailableError) as caught:
        await orchestrator.run(
            question="Question",
            scope=scope(),
            generate=generate,
            verify=verify,
            abstention="insufficient",
        )
    assert caught.value.status_code == 503
    assert tool.requests == []

    grader_orchestrator = AgenticOrchestrator(
        [tool],
        grader=LLMEvidenceGrader(BrokenReviewModel()),
        max_retrieval_rounds=1,
    )
    with pytest.raises(ReviewModelUnavailableError):
        await grader_orchestrator.run(
            question="Question",
            scope=scope(),
            generate=generate,
            verify=verify,
            abstention="insufficient",
        )

    class MalformedReviewModel:
        async def ainvoke(self, _messages):
            return "{}"

    with pytest.raises(ReviewModelUnavailableError):
        await AgenticOrchestrator(
            [tool],
            grader=LLMEvidenceGrader(MalformedReviewModel()),
            max_retrieval_rounds=1,
        ).run(
            question="Question",
            scope=scope(),
            generate=generate,
            verify=verify,
            abstention="insufficient",
        )


@pytest.mark.asyncio
async def test_subquery_results_are_interleaved_for_facet_coverage():
    class FacetTool:
        name = "hybrid"

        async def retrieve(self, request):
            slug = request.query.replace(" ", "-")
            document_id = {
                "primary": "doc1",
                "facet-a": "doc2",
                "facet-b": "doc3",
            }[request.query]
            return [
                source(
                    f"{slug}-{rank}",
                    document_id=document_id,
                    parent_id=f"{slug}-{rank}",
                )
                for rank in range(3)
            ]

    planned = plan(
        standalone_query="primary",
        subqueries=["facet-a", "facet-b"],
        required_facets=["a", "b"],
        latest_only=False,
    )
    result = await AgenticOrchestrator(
        [FacetTool()], max_tool_calls=4, max_subqueries=3
    ).run(
        question="Question",
        scope=RequestScope.from_request(
            ["doc1", "doc2", "doc3"], document_ids_explicit=True
        ),
        initial_plan=planned,
        generate=generate,
        verify=verify,
        abstention="insufficient",
    )

    assert len(result.sources) == 5
    assert {item.document_id for item in result.sources[:3]} == {
        "doc1",
        "doc2",
        "doc3",
    }


@pytest.mark.asyncio
async def test_sql_planner_queries_are_removed_before_retrieval():
    tool = RecordingTool()

    class SQLPlanner:
        async def plan(self, _request):
            return plan(
                standalone_query="SELECT risk FROM reports WHERE ticker='FPT'",
                subqueries=["SELECT * FROM entities", "rủi ro FPT"],
                exact_terms=["FPT", "SELECT secret FROM documents"],
                required_facets=["risk_factor", "rủi ro"],
            )

    await AgenticOrchestrator(
        [tool], planner=SQLPlanner(), max_retrieval_rounds=1
    ).run(
        question="FPT có rủi ro nào?",
        scope=scope(),
        generate=generate,
        verify=verify,
        abstention="insufficient",
    )

    assert [request.query for request in tool.requests] == [
        "FPT có rủi ro nào?",
        "rủi ro FPT",
    ]
    assert tool.requests[0].required_facets == ["rủi ro"]
    assert tool.requests[0].exact_terms == ["FPT"]


@pytest.mark.asyncio
async def test_sql_grader_suggestions_are_removed_from_corrective_fallback():
    class PlannerWithoutCorrect:
        async def plan(self, _request):
            return plan()

    orchestrator = AgenticOrchestrator(
        [RecordingTool()], planner=PlannerWithoutCorrect(), max_retrieval_rounds=2
    )
    corrected, used_fallback = await orchestrator._correct_plan(
        "FPT có rủi ro nào?",
        [],
        ["hybrid"],
        EvidenceGrade(
            verdict="insufficient",
            reason="missing",
            suggested_queries=[
                "SELECT secret FROM documents WHERE ticker='FPT'",
                "rủi ro FPT",
            ],
        ),
        plan(required_facets=["rủi ro"]),
    )

    assert used_fallback is True
    assert corrected.standalone_query == "FPT có rủi ro nào?"
    assert corrected.subqueries == ["rủi ro FPT"]


def test_default_tool_timeout_reserves_budget_for_later_stages():
    orchestrator = AgenticOrchestrator(
        [RecordingTool()], timeout_seconds=120, max_tool_calls=4
    )

    assert orchestrator.tool_timeout_seconds == 60


@pytest.mark.asyncio
async def test_local_retrieval_tool_calls_run_sequentially():
    class SerialTool:
        name = "hybrid"

        def __init__(self):
            self.active = 0
            self.max_active = 0
            self.calls = 0

        async def retrieve(self, _request):
            self.calls += 1
            self.active += 1
            self.max_active = max(self.max_active, self.active)
            try:
                await asyncio.sleep(0.01)
                return [source(str(self.calls), parent_id=f"p-{self.calls}")]
            finally:
                self.active -= 1

    tool = SerialTool()
    await AgenticOrchestrator(
        [tool], max_retrieval_rounds=1, max_tool_calls=4
    ).run(
        question="Question",
        scope=scope(),
        initial_plan=plan(subqueries=["facet one", "facet two"]),
        generate=generate,
        verify=verify,
        abstention="insufficient",
    )

    assert tool.calls == 3
    assert tool.max_active == 1


@pytest.mark.asyncio
async def test_agent_timeout_does_not_start_queued_subqueries():
    class SlowTool:
        name = "hybrid"

        def __init__(self):
            self.calls = 0

        async def retrieve(self, _request):
            self.calls += 1
            await asyncio.sleep(10)
            return [source()]

    tool = SlowTool()
    with pytest.raises(AgentTimeoutError):
        await AgenticOrchestrator(
            [tool],
            timeout_seconds=0.05,
            max_retrieval_rounds=1,
            max_tool_calls=4,
        ).run(
            question="Question",
            scope=scope(),
            initial_plan=plan(subqueries=["facet one", "facet two"]),
            generate=generate,
            verify=verify,
            abstention="insufficient",
        )

    assert tool.calls == 1


@pytest.mark.asyncio
async def test_tool_timeout_does_not_start_later_subqueries_for_same_tool():
    class SlowTool:
        name = "hybrid"

        def __init__(self):
            self.calls = 0

        async def retrieve(self, _request):
            self.calls += 1
            await asyncio.sleep(0.2)
            return [source()]

    tool = SlowTool()
    with pytest.raises(RetrievalToolsUnavailableError):
        await AgenticOrchestrator(
            [tool],
            timeout_seconds=1,
            tool_timeout_seconds=0.01,
            max_retrieval_rounds=1,
            max_tool_calls=4,
        ).run(
            question="Question",
            scope=scope(),
            initial_plan=plan(subqueries=["facet one", "facet two"]),
            generate=generate,
            verify=verify,
            abstention="insufficient",
        )

    assert tool.calls == 1
