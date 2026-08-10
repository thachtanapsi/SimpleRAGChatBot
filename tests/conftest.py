from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from rag_app.config import Settings


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    base = Settings.from_env(tmp_path)
    configured = replace(
        base,
        data_dir=tmp_path / "data",
        papers_dir=tmp_path / "papers",
        parent_chunk_size=80,
        parent_chunk_overlap=10,
        child_chunk_size=30,
        child_chunk_overlap=5,
        min_pdf_text_chars=10,
    )
    configured.ensure_directories()
    return configured
