"""Factory for fully initialised LightRAG instances with explicit local storage.

Architectural rule (CLAUDE.md): every LightRAG instance is created with explicit
backend definitions and immediately awaits ``initialize_storages()`` followed by
``initialize_pipeline_status()``.  ``initialize_storages`` already performs the
pipeline-status initialisation internally in lightrag-hku >= 1.5, and the call is
idempotent, so the explicit second call is harmless and keeps the contract
visible at the call site.
"""

from __future__ import annotations

import contextlib
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

from lightrag import LightRAG, RoleLLMConfig
from lightrag.kg.shared_storage import initialize_pipeline_status
from lightrag.utils import EmbeddingFunc, Tokenizer, logger

from config import Settings, build_entity_types_guidance
from llm import LLMFunc, make_embedding_func, make_llm_func, make_role_configs

KV_STORAGE = "JsonKVStorage"
VECTOR_STORAGE = "NanoVectorDBStorage"
GRAPH_STORAGE = "NetworkXStorage"
DOC_STATUS_STORAGE = "JsonDocStatusStorage"
GRAPHML_FILENAME = "graph_chunk_entity_relation.graphml"


def graphml_path(settings: Settings) -> Path:
    """Location of the persisted NetworkX graph for the given settings."""
    base = settings.working_dir
    if settings.workspace:
        base = base / settings.workspace
    return base / GRAPHML_FILENAME


async def build_rag(
    settings: Settings,
    *,
    llm_func: LLMFunc | None = None,
    embedding_func: EmbeddingFunc | None = None,
    role_configs: dict[str, RoleLLMConfig | dict[str, Any]] | None = None,
    tokenizer: Tokenizer | None = None,
    **overrides: Any,
) -> LightRAG:
    """Construct and initialise a LightRAG instance.

    Parameters
    ----------
    settings:
        Project settings (paths, models, language, thresholds).
    llm_func / embedding_func / tokenizer:
        Optional injected backends.  When omitted the OpenAI-compatible adapters
        from :mod:`llm` are used.  Tests inject deterministic stubs here.
    role_configs:
        Per-role LLM routing.  Defaults to the fast/strong split from
        :func:`llm.make_role_configs` when no custom ``llm_func`` is injected.
    overrides:
        Extra keyword arguments forwarded verbatim to :class:`LightRAG`.
    """
    settings.ensure_directories()

    if llm_func is None and not settings.openai_api_key:
        logger.warning(
            "OPENAI_API_KEY is not set; live LLM calls will fail unless the endpoint "
            "at OPENAI_BASE_URL accepts anonymous requests (use --backend stub offline)"
        )
    llm = llm_func or make_llm_func(settings.extract_model, settings)
    embedding = embedding_func or make_embedding_func(settings)
    if role_configs is None and llm_func is None:
        role_configs = make_role_configs(settings)

    kwargs: dict[str, Any] = {
        "working_dir": str(settings.working_dir),
        "workspace": settings.workspace,
        "kv_storage": KV_STORAGE,
        "vector_storage": VECTOR_STORAGE,
        "graph_storage": GRAPH_STORAGE,
        "doc_status_storage": DOC_STATUS_STORAGE,
        "llm_model_func": llm,
        "llm_model_name": settings.extract_model,
        "embedding_func": embedding,
        "role_llm_configs": role_configs or None,
        "tokenizer": tokenizer,
        "addon_params": {
            "language": settings.language,
            "entity_types_guidance": build_entity_types_guidance(),
        },
        "entity_extract_max_gleaning": settings.max_gleaning,
        "cosine_better_than_threshold": settings.cosine_threshold,
    }
    kwargs.update(overrides)

    logger.info(
        "Constructing LightRAG: working_dir=%s graph=%s vector=%s kv=%s doc_status=%s",
        kwargs["working_dir"],
        GRAPH_STORAGE,
        VECTOR_STORAGE,
        KV_STORAGE,
        DOC_STATUS_STORAGE,
    )
    rag = LightRAG(**kwargs)

    await rag.initialize_storages()
    await initialize_pipeline_status(workspace=rag.workspace)
    logger.info("LightRAG storages initialised; pipeline status ready")
    return rag


@contextlib.asynccontextmanager
async def rag_session(settings: Settings, **kwargs: Any) -> AsyncIterator[LightRAG]:
    """Async context manager guaranteeing ``finalize_storages()`` on exit."""
    rag = await build_rag(settings, **kwargs)
    try:
        yield rag
    finally:
        await rag.finalize_storages()
        logger.info("LightRAG storages finalised")
