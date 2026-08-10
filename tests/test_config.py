from pathlib import Path

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
