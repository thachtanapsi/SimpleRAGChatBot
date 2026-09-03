from __future__ import annotations

import asyncio
from typing import Any

import pytest
from langchain_core.messages import AIMessage

from rag_app.errors import SelfCheckUnavailableError
from rag_app.retrieval import RetrievedSource
from rag_app.self_check import (
    GeneratedAnswer,
    GeneratedClaim,
    InvalidGeneratedAnswerError,
    LLMClaimVerifier,
    OLLAMA_GENERATED_ANSWER_SCHEMA,
    OLLAMA_VERIFICATION_REPORT_SCHEMA,
    SelfCheckService,
    SourceEvidence,
    VerificationReport,
    VerificationRequest,
    audit_generated_answer,
    parse_generated_answer,
    parse_verification_report,
)


def assert_ollama_schema_is_flat(value: Any) -> None:
    if isinstance(value, dict):
        assert not ({"$defs", "$ref", "anyOf"} & value.keys())
        for child in value.values():
            assert_ollama_schema_is_flat(child)
    elif isinstance(value, list):
        for child in value:
            assert_ollama_schema_is_flat(child)


class StructuredVerifierModel:
    def __init__(self, response: Any):
        self.response = response
        self.schema: dict[str, Any] | None = None
        self.method: str | None = None
        self.include_raw: bool | None = None

    def with_structured_output(
        self,
        schema: dict[str, Any],
        *,
        method: str,
        include_raw: bool,
    ) -> "StructuredVerifierModel":
        self.schema = schema
        self.method = method
        self.include_raw = include_raw
        return self

    async def ainvoke(self, _messages: list[Any]) -> Any:
        return self.response


def source(source_id: str = "1", text: str | None = None) -> RetrievedSource:
    return RetrievedSource(
        id=source_id,
        document_id=f"doc-{source_id}",
        filename="report.pdf",
        page=1,
        parent_id=f"parent-{source_id}",
        text=text
        or "FPT đạt tăng trưởng 20% trong Q2/2026, công bố ngày 27/08/2026.",
        score=0.9,
    )


def generated(**updates: object) -> GeneratedAnswer:
    payload: dict[str, object] = {
        "answer": "FPT tăng trưởng 20% trong Q2/2026 [1].",
        "claims": [
            {
                "text": "FPT tăng trưởng 20% trong Q2/2026.",
                "citation_ids": ["1"],
                "ticker": "FPT",
                "numeric": "20%",
                "unit": "%",
                "period": "Q2/2026",
                "date": None,
            }
        ],
    }
    payload.update(updates)
    return GeneratedAnswer.model_validate(payload, strict=True)


def verification_request() -> VerificationRequest:
    return VerificationRequest(
        question="FPT tăng bao nhiêu?",
        generated=generated(),
        sources=[
            SourceEvidence(
                id="1",
                document_id="doc-1",
                text="FPT đạt tăng trưởng 20% trong Q2/2026.",
            )
        ],
    )


@pytest.mark.parametrize(
    "schema",
    [OLLAMA_GENERATED_ANSWER_SCHEMA, OLLAMA_VERIFICATION_REPORT_SCHEMA],
)
def test_self_check_ollama_schemas_are_flat(schema):
    assert_ollama_schema_is_flat(schema)


@pytest.mark.asyncio
async def test_claim_verifier_uses_raw_json_schema_wrapper_and_parses_payload():
    model = StructuredVerifierModel(
        {
            "raw": AIMessage(
                content="",
                response_metadata={"done_reason": "stop"},
            ),
            "parsed": {
                "passed": True,
                "issues": [],
                "supported_citation_ids": ["1"],
            },
            "parsing_error": None,
        }
    )

    result = await LLMClaimVerifier(model).verify(verification_request())

    assert result.passed is True
    assert result.supported_citation_ids == ["1"]
    assert model.schema == OLLAMA_VERIFICATION_REPORT_SCHEMA
    assert model.method == "json_schema"
    assert model.include_raw is True


@pytest.mark.parametrize(
    ("done_reason", "parsing_error", "reason_code"),
    [
        ("length", None, "self_check_response_truncated"),
        (
            "stop",
            ValueError("invalid verification JSON"),
            "self_check_response_parse_failed",
        ),
    ],
)
@pytest.mark.asyncio
async def test_claim_verifier_rejects_truncation_and_parsing_errors(
    done_reason, parsing_error, reason_code
):
    model = StructuredVerifierModel(
        {
            "raw": AIMessage(
                content="{}",
                response_metadata={"done_reason": done_reason},
            ),
            "parsed": None,
            "parsing_error": parsing_error,
        }
    )

    with pytest.raises(SelfCheckUnavailableError) as caught:
        await LLMClaimVerifier(model).verify(verification_request())

    assert caught.value.reason_code == reason_code


def test_invalid_verification_report_has_safe_contract_reason():
    with pytest.raises(SelfCheckUnavailableError) as caught:
        parse_verification_report(
            {
                "passed": False,
                "issues": [],
                "supported_citation_ids": ["1"],
            }
        )

    assert caught.value.reason_code == "self_check_response_contract_invalid"
    assert "cấu trúc" in str(caught.value)


def test_deterministic_audit_checks_citation_and_financial_dimensions():
    report = audit_generated_answer(generated(), [source()])

    assert report.passed is True
    assert report.supported_citation_ids == ["1"]


def test_deterministic_audit_rejects_unsupported_number_ticker_unit_period_and_date():
    answer = GeneratedAnswer(
        answer="ACB tăng 99 USD trong Q4/2030 ngày 01/01/2031 [1].",
        claims=[
            GeneratedClaim(
                text="ACB tăng 99 USD trong Q4/2030 ngày 01/01/2031.",
                citation_ids=["1"],
                ticker="ACB",
                numeric="99",
                unit="USD",
                period="Q4/2030",
                date="01/01/2031",
            )
        ],
    )

    report = audit_generated_answer(answer, [source()])
    codes = {issue.code for issue in report.issues}

    assert report.passed is False
    assert {
        "unsupported_number",
        "unsupported_ticker",
        "unsupported_unit",
        "unsupported_period",
        "unsupported_date",
    }.issubset(codes)


def test_numeric_audit_does_not_collapse_decimal_or_sign_punctuation():
    answer = GeneratedAnswer(
        answer="Biên lợi nhuận là +1.23% [1].",
        claims=[
            GeneratedClaim(
                text="Biên lợi nhuận là +1.23%.",
                citation_ids=["1"],
                numeric="+1.23%",
                unit="%",
            )
        ],
    )

    report = audit_generated_answer(
        answer,
        [source(text="Biên lợi nhuận là 123%.")],
    )

    assert report.passed is False
    assert "unsupported_number" in {issue.code for issue in report.issues}


def test_ticker_and_unit_audit_require_phrase_boundaries():
    answer = GeneratedAnswer(
        answer="ACB dùng đơn vị M [1].",
        claims=[
            GeneratedClaim(
                text="ACB dùng đơn vị M.",
                citation_ids=["1"],
                ticker="ACB",
                unit="M",
            )
        ],
    )

    report = audit_generated_answer(
        answer,
        [source(text="XACB được nhắc trong market report.")],
    )

    assert report.passed is False
    assert {"unsupported_ticker", "unsupported_unit"}.issubset(
        {issue.code for issue in report.issues}
    )


def test_strict_generator_schema_rejects_extra_fields_and_type_coercion():
    with pytest.raises(InvalidGeneratedAnswerError):
        parse_generated_answer(
            {
                "answer": "Fact [1]",
                "claims": [{"text": "Fact", "citation_ids": [1]}],
                "reasoning": "hidden",
            }
        )


class RejectingVerifier:
    async def verify(self, _request):
        return VerificationReport(
            passed=False,
            issues=[
                {
                    "code": "unsupported_claim",
                    "detail": "claim is not entailed",
                    "claim_index": 0,
                    "citation_id": "1",
                }
            ],
            supported_citation_ids=[],
        )


class BrokenVerifier:
    async def verify(self, _request):
        raise ConnectionError("review model unavailable")


class SlowVerifier:
    async def verify(self, _request):
        await asyncio.Event().wait()


class OutOfScopeVerifier:
    async def verify(self, _request):
        return VerificationReport(
            passed=True,
            issues=[],
            supported_citation_ids=["2"],
        )


class IncompletePassingVerifier:
    async def verify(self, _request):
        return VerificationReport(
            passed=True,
            issues=[],
            supported_citation_ids=[],
        )


@pytest.mark.asyncio
async def test_semantic_verifier_runs_only_after_deterministic_audit_passes():
    report = await SelfCheckService(RejectingVerifier()).verify(
        "FPT tăng bao nhiêu?", generated(), [source()]
    )

    assert report.passed is False
    assert report.issues[0].code == "unsupported_claim"


@pytest.mark.asyncio
async def test_verifier_technical_failure_is_domain_503():
    with pytest.raises(SelfCheckUnavailableError) as caught:
        await SelfCheckService(BrokenVerifier(), timeout_seconds=0.5).verify(
            "FPT tăng bao nhiêu?", generated(), [source()]
        )

    assert caught.value.reason_code == "self_check_execution_failed"


@pytest.mark.asyncio
async def test_verifier_timeout_has_safe_reason():
    with pytest.raises(SelfCheckUnavailableError) as caught:
        await SelfCheckService(SlowVerifier(), timeout_seconds=0.01).verify(
            "FPT tăng bao nhiêu?", generated(), [source()]
        )

    assert caught.value.reason_code == "self_check_timeout"


@pytest.mark.asyncio
async def test_verifier_citation_outside_request_scope_has_safe_reason():
    with pytest.raises(SelfCheckUnavailableError) as caught:
        await SelfCheckService(OutOfScopeVerifier()).verify(
            "FPT tăng bao nhiêu?", generated(), [source()]
        )

    assert caught.value.reason_code == "self_check_source_scope_invalid"


@pytest.mark.asyncio
async def test_passing_verifier_must_support_every_claim_citation():
    report = await SelfCheckService(IncompletePassingVerifier()).verify(
        "FPT tăng bao nhiêu?", generated(), [source()]
    )

    assert report.passed is False
    assert report.issues[0].code == "verifier_rejected"
