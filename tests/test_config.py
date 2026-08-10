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

