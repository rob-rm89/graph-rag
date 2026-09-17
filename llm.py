"""LLM and embedding adapters for OpenAI-compatible endpoints.

LightRAG calls ``llm_model_func(prompt, system_prompt=..., history_messages=...,
**kwargs)`` and, through its per-role wrapper, always binds ``hashing_kv``.
The adapters below honour that contract and route two models:

* ``settings.extract_model`` (fast, cheap) for entity extraction and keyword
  extraction, and
* ``settings.query_model`` (high-level reasoning) for answer generation,

as required by the project architecture rules.
"""

from __future__ import annotations

import json
import re
from collections.abc import Awaitable, Callable
from typing import Any

import json_repair
import numpy as np
from lightrag import RoleLLMConfig
from lightrag.llm.openai import openai_complete_if_cache, openai_embed
from lightrag.utils import EmbeddingFunc, logger

from config import Settings

LLMFunc = Callable[..., Awaitable[str]]

JSON_RESPONSE_FORMAT: dict[str, str] = {"type": "json_object"}
_CODE_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE | re.MULTILINE)


def make_llm_func(model: str, settings: Settings) -> LLMFunc:
    """Build an ``llm_model_func`` bound to one OpenAI-compatible chat model."""

    async def _complete(
        prompt: str,
        system_prompt: str | None = None,
        history_messages: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> str:
        # LightRAG's role wrapper injects the response-cache storage; caching is
        # handled upstream, so the adapter must swallow it.
        kwargs.pop("hashing_kv", None)
        return await openai_complete_if_cache(
            model,
            prompt,
            system_prompt=system_prompt,
            history_messages=history_messages or [],
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url,
            **kwargs,
        )

    _complete.__name__ = f"openai_complete_{model}"
    _complete.__qualname__ = _complete.__name__
    return _complete


def make_embedding_func(settings: Settings) -> EmbeddingFunc:
    """Wrap ``openai_embed`` so the model name and dimension follow settings.

    ``openai_embed`` ships as an :class:`EmbeddingFunc` pinned to
    ``text-embedding-3-small`` / 1536 dims; using its raw ``.func`` lets the
    project pick a different model without tripping the dimension check.
    """
    raw_embed = openai_embed.func

    async def _embed(texts: list[str]) -> np.ndarray:
        return await raw_embed(
            texts,
            model=settings.embedding_model,
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url,
        )

    return EmbeddingFunc(
        embedding_dim=settings.embedding_dim,
        max_token_size=8192,
        model_name=settings.embedding_model,
        func=_embed,
    )


def make_role_configs(settings: Settings) -> dict[str, RoleLLMConfig]:
    """Route extraction/keywords to the fast model and answers to the strong one."""
    extract_llm = make_llm_func(settings.extract_model, settings)
    query_llm = make_llm_func(settings.query_model, settings)
    logger.info(
        "LLM routing: extract/keyword -> %s, query -> %s",
        settings.extract_model,
        settings.query_model,
    )
    return {
        "extract": RoleLLMConfig(
            func=extract_llm, metadata={"model": settings.extract_model}
        ),
        "keyword": RoleLLMConfig(
            func=extract_llm, metadata={"model": settings.extract_model}
        ),
        "query": RoleLLMConfig(
            func=query_llm, metadata={"model": settings.query_model}
        ),
    }


def parse_json_object(raw: str) -> dict[str, Any]:
    """Parse an LLM reply into a JSON object, tolerating fences and minor damage."""
    text = _CODE_FENCE_RE.sub("", (raw or "").strip()).strip()
    if not text:
        return {}
    try:
        parsed: Any = json.loads(text)
    except json.JSONDecodeError:
        parsed = json_repair.loads(text)
    if isinstance(parsed, dict):
        return parsed
    logger.warning("LLM JSON reply was not an object (got %s)", type(parsed).__name__)
    return {}


async def complete_json(
    llm_func: LLMFunc, system_prompt: str, user_prompt: str
) -> dict[str, Any]:
    """Run one structured-output completion and return the parsed object."""
    raw = await llm_func(
        user_prompt, system_prompt=system_prompt, response_format=JSON_RESPONSE_FORMAT
    )
    return parse_json_object(raw)
