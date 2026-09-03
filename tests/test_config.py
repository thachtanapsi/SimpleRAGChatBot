from pathlib import Path

import pytest

from rag_app.config import Settings


def test_default_base_dir_is_current_working_directory(tmp_path, monkeypatch):
    monkeypatch.delenv("RAG_BASE_DIR", raising=False)
    monkeypatch.chdir(tmp_path)
    assert Settings.from_env().base_dir == tmp_path.resolve()


def test_explicit_base_dir_environment_wins(tmp_path, monkeypatch):
    configured = tmp_path / "configured"
    monkeypatch.setenv("RAG_BASE_DIR", str(configured))
    assert Settings.from_env().base_dir == configured.resolve()


def test_dense_k_uses_legacy_fallback_and_new_value_wins(tmp_path, monkeypatch):
    monkeypatch.setenv("RAG_CHILD_SEARCH_K", "17")
    monkeypatch.delenv("RAG_DENSE_SEARCH_K", raising=False)
    assert Settings.from_env(tmp_path).dense_search_k == 17
    monkeypatch.setenv("RAG_DENSE_SEARCH_K", "29")
    assert Settings.from_env(tmp_path).dense_search_k == 29


def test_hybrid_boolean_and_resolved_revision_affect_configuration(tmp_path, monkeypatch):
    monkeypatch.delenv("RAG_HYBRID_SEARCH", raising=False)
    assert Settings.from_env(tmp_path).hybrid_search is False
    monkeypatch.setenv("RAG_HYBRID_SEARCH", "0")
    settings = Settings.from_env(tmp_path)
    assert settings.hybrid_search is False
    assert settings.index_fingerprint_for("commit-a") != settings.index_fingerprint_for(
        "commit-b"
    )


def test_cross_page_configuration_and_fingerprint(tmp_path, monkeypatch):
    default = Settings.from_env(tmp_path)
    assert default.cross_page_enabled is True
    assert default.cross_page_context_chars == 1200

    monkeypatch.setenv("RAG_CROSS_PAGE_ENABLED", "0")
    monkeypatch.setenv("RAG_CROSS_PAGE_CONTEXT_CHARS", "900")
    changed = Settings.from_env(tmp_path)
    assert changed.cross_page_enabled is False
    assert changed.cross_page_context_chars == 900
    assert changed.index_fingerprint_for("commit") != default.index_fingerprint_for(
        "commit"
    )


def test_advanced_rag_defaults_are_off_and_do_not_change_index(tmp_path, monkeypatch):
    default = Settings.from_env(tmp_path)
    assert default.rerank_enabled is False
    assert default.agentic_enabled is False
    assert default.self_check_enabled is False
    assert default.graph_build_enabled is False
    assert default.graph_enabled is False
    assert default.rerank_candidate_k == 40
    assert default.rerank_child_k == 12
    assert default.graph_max_hops == 2

    monkeypatch.setenv("RAG_HYBRID_SEARCH", "1")
    monkeypatch.setenv("RAG_RERANK_ENABLED", "1")
    monkeypatch.setenv("RAG_AGENTIC_ENABLED", "1")
    monkeypatch.setenv("RAG_SELF_CHECK_ENABLED", "1")
    monkeypatch.setenv("RAG_GRAPH_BUILD_ENABLED", "1")
    monkeypatch.setenv("RAG_GRAPH_ENABLED", "1")
    enabled = Settings.from_env(tmp_path)
    assert enabled.index_fingerprint_for("commit") == default.index_fingerprint_for(
        "commit"
    )


def test_review_and_graph_models_default_to_chat_model(tmp_path, monkeypatch):
    monkeypatch.setenv("RAG_CHAT_MODEL", "local-review-model")
    settings = Settings.from_env(tmp_path)
    assert settings.review_model == "local-review-model"
    assert settings.graph_extractor_model == "local-review-model"


def test_advanced_answer_token_limit_is_independently_configurable(
    tmp_path, monkeypatch
):
    default = Settings.from_env(tmp_path)
    assert default.advanced_answer_num_predict == 1024

    monkeypatch.setenv("RAG_ANSWER_NUM_PREDICT", "2048")
    monkeypatch.setenv("RAG_ADVANCED_ANSWER_NUM_PREDICT", "768")
    configured = Settings.from_env(tmp_path)

    assert configured.answer_num_predict == 2048
    assert configured.advanced_answer_num_predict == 768


@pytest.mark.parametrize("value", ["255", "1025"])
def test_advanced_answer_token_limit_is_bounded(tmp_path, monkeypatch, value):
    monkeypatch.setenv("RAG_ADVANCED_ANSWER_NUM_PREDICT", value)

    with pytest.raises(ValueError, match="RAG_ADVANCED_ANSWER_NUM_PREDICT"):
        Settings.from_env(tmp_path)


@pytest.mark.parametrize(
    ("environment", "message"),
    [
        ({"RAG_AGENTIC_ENABLED": "1"}, "RAG_HYBRID_SEARCH"),
        (
            {
                "RAG_HYBRID_SEARCH": "1",
                "RAG_AGENTIC_ENABLED": "1",
            },
            "RAG_RERANK_ENABLED",
        ),
        ({"RAG_SELF_CHECK_ENABLED": "1"}, "RAG_AGENTIC_ENABLED"),
        ({"RAG_GRAPH_ENABLED": "1"}, "RAG_AGENTIC_ENABLED"),
    ],
)
def test_invalid_advanced_feature_dependencies(
    tmp_path, monkeypatch, environment, message
):
    for name, value in environment.items():
        monkeypatch.setenv(name, value)
    with pytest.raises(ValueError, match=message):
        Settings.from_env(tmp_path)


def test_graph_routing_requires_graph_build(tmp_path, monkeypatch):
    monkeypatch.setenv("RAG_HYBRID_SEARCH", "1")
    monkeypatch.setenv("RAG_RERANK_ENABLED", "1")
    monkeypatch.setenv("RAG_AGENTIC_ENABLED", "1")
    monkeypatch.setenv("RAG_GRAPH_ENABLED", "1")
    with pytest.raises(ValueError, match="RAG_GRAPH_BUILD_ENABLED"):
        Settings.from_env(tmp_path)
