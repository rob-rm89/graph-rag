"""Shared pytest fixtures."""

from __future__ import annotations

from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Environment variables that would silently change LightRAG or project defaults.
LIGHTRAG_ENV_VARS = (
    "ENTITY_TYPES",
    "USER_PROMPT_PREFIX",
    "USER_PROMPT_PREFIX_FILE",
    "RERANK_BY_DEFAULT",
    "CHUNK_SIZE",
    "CHUNK_OVERLAP_SIZE",
    "MAX_GLEANING",
    "FORCE_LLM_SUMMARY_ON_MERGE",
    "WORKSPACE",
    "WORKING_DIR",
    "DATA_DIR",
    "CANVAS_PATH",
    "OPENAI_API_KEY",
    "OPENAI_BASE_URL",
)


@pytest.fixture(autouse=True)
def clean_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in LIGHTRAG_ENV_VARS:
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def project_root() -> Path:
    return PROJECT_ROOT


@pytest.fixture
def sample_paper(project_root: Path) -> Path:
    path = project_root / "data" / "sample_paper.md"
    assert path.exists(), "sample corpus missing"
    return path
