from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from rag_app.graph_extraction import (
    ExtractedAssertion,
    LocalGraphExtractor,
    OLLAMA_GRAPH_EXTRACTION_SCHEMA,
    graph_extractor_fingerprint,
)


class FakeStructuredLLM:
    def __init__(self, result=None):
        self.messages = None
        self.result = result

    async def ainvoke(self, messages):
        self.messages = messages
        return self.result if self.result is not None else {
            "assertions": [
                {
                    "predicate": "HAS_RISK",
                    "object_entity": {
                        "entity_type": "risk",
                        "canonical_key": "jpy",
                        "canonical_name": "Rủi ro JPY",
                    },
                    "quote": "FPT chịu rủi ro JPY",
                    "confidence": 0.9,
                }
            ]
        }


class FakeLLM:
    def __init__(self, result=None):
        self.structured = FakeStructuredLLM(result)
        self.schema = None
        self.method = None

    def with_structured_output(self, schema, *, method):
        self.schema = schema
        self.method = method
        return self.structured


@pytest.mark.asyncio
async def test_local_graph_extractor_uses_schema_and_returns_strict_mappings():
    llm = FakeLLM()
    extractor = LocalGraphExtractor(llm)
    result = await extractor(
        SimpleNamespace(
            document={"ticker": "FPT", "analysis_date": "2026-08-28"},
            parent={
                "text": "FPT chịu rủi ro JPY",
                "digest_section": "risks_unknowns",
            },
            claims=(
                {
                    "text": "FPT chịu rủi ro JPY",
                    "confidence": 0.9,
                    "evidence_ids": ["opaque:must-not-leak"],
                },
            ),
            default_subject={
                "entity_type": "security",
                "canonical_key": "FPT",
                "canonical_name": "FPT",
            },
        )
    )

    assert llm.method == "json_schema"
    assert llm.schema == OLLAMA_GRAPH_EXTRACTION_SCHEMA
    assert result[0]["predicate"] == "HAS_RISK"
    assert result[0]["quote"] == "FPT chịu rủi ro JPY"
    assert "<evidence>FPT chịu rủi ro JPY</evidence>" in llm.structured.messages[1].content
    assert "opaque:must-not-leak" not in llm.structured.messages[1].content


def test_ollama_graph_schema_is_flat_and_grammar_compatible():
    def walk(value):
        if isinstance(value, dict):
            assert not ({"$defs", "$ref", "anyOf", "allOf"} & value.keys())
            for item in value.values():
                walk(item)
        elif isinstance(value, list):
            for item in value:
                walk(item)

    walk(OLLAMA_GRAPH_EXTRACTION_SCHEMA)
    assertions = OLLAMA_GRAPH_EXTRACTION_SCHEMA["properties"]["assertions"]["items"]
    entity_branch, literal_branch = assertions["oneOf"]
    assert "object_entity" in entity_branch["required"]
    assert "object_literal" not in entity_branch["properties"]
    assert "object_literal" in literal_branch["required"]
    assert literal_branch["properties"]["object_literal"] == {"type": "string"}
    assert "object_entity" not in literal_branch["properties"]


@pytest.mark.asyncio
async def test_local_graph_extractor_drops_invalid_predicate_type_and_self_edge():
    llm = FakeLLM(
        {
            "assertions": [
                {
                    "predicate": "SUPPLIES_TO",
                    "object_entity": {
                        "entity_type": "risk",
                        "canonical_key": "jpy",
                        "canonical_name": "Rủi ro JPY",
                    },
                    "quote": "FPT chịu rủi ro JPY",
                },
                {
                    "predicate": "PARTNERS_WITH",
                    "object_entity": {
                        "entity_type": "security",
                        "canonical_key": "fpt",
                        "canonical_name": "FPT",
                    },
                    "quote": "FPT chịu rủi ro JPY",
                },
            ]
        }
    )
    extractor = LocalGraphExtractor(llm)

    result = await extractor(
        SimpleNamespace(
            document={"ticker": "FPT", "analysis_date": "2026-08-28"},
            parent={"text": "FPT chịu rủi ro JPY"},
            claims=(),
            default_subject={
                "entity_type": "security",
                "canonical_key": "FPT",
                "canonical_name": "FPT",
            },
        )
    )

    assert result == []


def test_graph_extraction_schema_rejects_invalid_predicate_and_double_object():
    with pytest.raises(ValidationError):
        ExtractedAssertion.model_validate(
            {
                "predicate": "RELATED_TO",
                "object_literal": "x",
                "quote": "quote",
            }
        )
    with pytest.raises(ValidationError):
        ExtractedAssertion.model_validate(
            {
                "predicate": "HAS_RISK",
                "object_literal": "x",
                "object_entity": {
                    "entity_type": "risk",
                    "canonical_key": "x",
                    "canonical_name": "x",
                },
                "quote": "quote",
            }
        )


def test_graph_extractor_fingerprint_versions_index_model_and_digest():
    first = graph_extractor_fingerprint(
        index_fingerprint="index-a", model_name="gemma", model_digest="digest-a"
    )
    assert first == graph_extractor_fingerprint(
        index_fingerprint="index-a", model_name="gemma", model_digest="digest-a"
    )
    assert first != graph_extractor_fingerprint(
        index_fingerprint="index-b", model_name="gemma", model_digest="digest-a"
    )
    assert first != graph_extractor_fingerprint(
        index_fingerprint="index-a", model_name="gemma", model_digest="digest-b"
    )
