"""Deterministic offline backend: stub LLM, hashing embeddings, codepoint tokenizer.

This module lets the *real* LightRAG pipeline run end-to-end without network
access or API keys.  It is used by the test-suite and by ``main.py --backend
stub``.  Nothing here is a model; the stub recognises which LightRAG prompt it
received and returns a syntactically valid reply that the corresponding parser
accepts.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import AsyncIterator
from typing import Any

import numpy as np
from lightrag.utils import EmbeddingFunc, Tokenizer, logger

from config import (
    REL_AFFILIATED_WITH,
    REL_AUTHORED_BY,
    REL_CITES,
    REL_PUBLISHED_IN,
    REL_PUBLISHED_YEAR,
)
from ingestion import PROFILER_MARKER

TUPLE_DELIMITER = "<|#|>"
COMPLETION_DELIMITER = "<|COMPLETE|>"
JSON_OBJECT_FORMAT = {"type": "json_object"}
STUB_ANSWER_PREFIX = "STUB ANSWER"

# --------------------------------------------------------------------------- #
# Tokenizer
# --------------------------------------------------------------------------- #


class CodepointTokenizer:
    """Lossless 1-token-per-character tokenizer (implements ``TokenizerInterface``).

    ``decode(encode(s)) == s`` holds for every string, which LightRAG's chunker
    relies on to map token windows back to source spans.  Stateless, therefore
    thread-safe and trivially deep-copyable.
    """

    def encode(self, content: str) -> list[int]:
        return [ord(char) for char in content]

    def decode(self, tokens: list[int]) -> str:
        return "".join(chr(token) for token in tokens)

    def __deepcopy__(self, memo: dict[int, Any]) -> CodepointTokenizer:
        return self

    def __copy__(self) -> CodepointTokenizer:
        return self


def codepoint_tokenizer() -> Tokenizer:
    """LightRAG ``Tokenizer`` wrapper that never touches tiktoken."""
    return Tokenizer(model_name="codepoint", tokenizer=CodepointTokenizer())


# --------------------------------------------------------------------------- #
# Knowledge catalogue for data/sample_paper.md
# --------------------------------------------------------------------------- #

SAMPLE_TITLE = (
    "Sparse Attention Routing for Long-Document Retrieval-Augmented Generation"
)

# name -> (lower-case entity type, description)
SAMPLE_ENTITIES: dict[str, tuple[str, str]] = {
    SAMPLE_TITLE: (
        "paper",
        "Paper proposing Sparse Attention Routing for long-document RAG, "
        "published at EMNLP 2024.",
    ),
    "Elena Marchetti": (
        "author",
        "First author, affiliated with Politecnico di Torino.",
    ),
    "Kwame Mensah": (
        "author",
        "Second author, affiliated with the University of Ghana.",
    ),
    "Sofia Lindqvist": (
        "author",
        "Third author, affiliated with KTH Royal Institute of Technology.",
    ),
    # Initials-only spelling: lets tests exercise author reconciliation.
    "E. Marchetti": ("author", "Author cited by initials."),
    "EMNLP 2024": ("venue", "Conference on Empirical Methods in NLP, 2024 edition."),
    "2024": ("year", "Publication year of the paper."),
    "Politecnico di Torino": ("organization", "Italian technical university."),
    "University of Ghana": ("organization", "Public university in Accra, Ghana."),
    "KTH Royal Institute of Technology": (
        "organization",
        "Swedish technical university.",
    ),
    "Sparse Attention Routing": (
        "method",
        "Method that learns a sparse routing distribution over document regions "
        "before budgeted decoder attention.",
    ),
    "Dual-Level Retrieval": (
        "method",
        "Retrieval scheme pairing low-level entity lookup with high-level "
        "thematic aggregation.",
    ),
    "Transformer": ("method", "Attention-based neural sequence architecture."),
    "Retrieval-Augmented Generation": (
        "concept",
        "Paradigm that retrieves evidence before generating an answer.",
    ),
    "Long-Document Understanding": (
        "concept",
        "Research problem of reasoning over documents exceeding the context window.",
    ),
    "Knowledge Graph": (
        "concept",
        "Graph of extracted entities and relations used for structured retrieval.",
    ),
    "NarrativeQA": ("dataset", "Question answering over books and film scripts."),
    "LongBench": ("dataset", "Multi-task benchmark for long-context understanding."),
    "Exact Match": ("metric", "Whether a generated answer matches a gold answer."),
    "ROUGE-L": ("metric", "Longest-common-subsequence overlap metric."),
    "LightRAG": ("method", "Graph-based RAG system with dual-level retrieval."),
    "GraphRAG": ("method", "Graph RAG system built on community summaries."),
    "Attention Is All You Need": (
        "citedwork",
        "Vaswani et al. (2017); introduced the Transformer.",
    ),
    "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks": (
        "citedwork",
        "Lewis et al. (2020); established retrieve-then-generate.",
    ),
    "LightRAG: Simple and Fast Retrieval-Augmented Generation": (
        "citedwork",
        "Guo et al. (2024); introduced dual-level retrieval.",
    ),
    "From Local to Global: A Graph RAG Approach to Query-Focused Summarization": (
        "citedwork",
        "Edge et al. (2024); graph community summaries for global questions.",
    ),
}

# (source, target, keywords, description)
SAMPLE_RELATIONS: list[tuple[str, str, str, str]] = [
    (
        SAMPLE_TITLE,
        "Elena Marchetti",
        REL_AUTHORED_BY,
        "Elena Marchetti authored the paper.",
    ),
    (SAMPLE_TITLE, "Kwame Mensah", REL_AUTHORED_BY, "Kwame Mensah authored the paper."),
    (
        SAMPLE_TITLE,
        "Sofia Lindqvist",
        REL_AUTHORED_BY,
        "Sofia Lindqvist authored the paper.",
    ),
    (SAMPLE_TITLE, "EMNLP 2024", REL_PUBLISHED_IN, "The paper appeared at EMNLP 2024."),
    (SAMPLE_TITLE, "2024", REL_PUBLISHED_YEAR, "The paper was published in 2024."),
    (
        "Elena Marchetti",
        "Politecnico di Torino",
        REL_AFFILIATED_WITH,
        "Elena Marchetti is affiliated with Politecnico di Torino.",
    ),
    (
        "Kwame Mensah",
        "University of Ghana",
        REL_AFFILIATED_WITH,
        "Kwame Mensah is affiliated with the University of Ghana.",
    ),
    (
        "Sofia Lindqvist",
        "KTH Royal Institute of Technology",
        REL_AFFILIATED_WITH,
        "Sofia Lindqvist is affiliated with KTH.",
    ),
    (
        SAMPLE_TITLE,
        "Sparse Attention Routing",
        "proposes, method",
        "The paper proposes Sparse Attention Routing.",
    ),
    (
        "Sparse Attention Routing",
        "Dual-Level Retrieval",
        "combines, retrieval",
        "Combines Dual-Level Retrieval with budgeted decoder attention.",
    ),
    (
        "Sparse Attention Routing",
        "Transformer",
        "uses, decoder",
        "Sparse Attention Routing uses a Transformer decoder.",
    ),
    (
        "Sparse Attention Routing",
        "Retrieval-Augmented Generation",
        "improves, paradigm",
        "Sparse Attention Routing improves Retrieval-Augmented Generation.",
    ),
    (
        "Sparse Attention Routing",
        "Long-Document Understanding",
        "addresses, problem",
        "Sparse Attention Routing addresses Long-Document Understanding.",
    ),
    (
        "Dual-Level Retrieval",
        "Knowledge Graph",
        "indexes, entities",
        "Dual-Level Retrieval indexes Knowledge Graph entities.",
    ),
    (
        "Sparse Attention Routing",
        "NarrativeQA",
        "evaluated on, dataset",
        "Sparse Attention Routing was evaluated on NarrativeQA.",
    ),
    (
        "Sparse Attention Routing",
        "LongBench",
        "evaluated on, dataset",
        "Sparse Attention Routing was evaluated on LongBench.",
    ),
    (
        "Sparse Attention Routing",
        "Exact Match",
        "improves, metric",
        "Sparse Attention Routing improves Exact Match.",
    ),
    (
        "Sparse Attention Routing",
        "ROUGE-L",
        "improves, metric",
        "Sparse Attention Routing improves ROUGE-L.",
    ),
    (
        "LightRAG",
        "Dual-Level Retrieval",
        "introduces",
        "LightRAG introduced Dual-Level Retrieval.",
    ),
    (
        "GraphRAG",
        "Knowledge Graph",
        "builds on",
        "GraphRAG builds community summaries over a graph.",
    ),
    (
        SAMPLE_TITLE,
        "Attention Is All You Need",
        REL_CITES,
        "The paper cites Attention Is All You Need.",
    ),
    (
        SAMPLE_TITLE,
        "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks",
        REL_CITES,
        "The paper cites the RAG paper by Lewis et al.",
    ),
    (
        SAMPLE_TITLE,
        "LightRAG: Simple and Fast Retrieval-Augmented Generation",
        REL_CITES,
        "The paper cites the LightRAG paper.",
    ),
    (
        SAMPLE_TITLE,
        "From Local to Global: A Graph RAG Approach to Query-Focused Summarization",
        REL_CITES,
        "The paper cites the GraphRAG paper.",
    ),
]

SAMPLE_PROFILE: dict[str, Any] = {
    "title": SAMPLE_TITLE,
    "authors": ["Elena Marchetti", "Kwame Mensah", "Sofia Lindqvist"],
    "year": 2024,
    "venue": "EMNLP 2024",
    "doi": "10.0000/synthetic.2024.001",
    "affiliations": [
        "Politecnico di Torino",
        "University of Ghana",
        "KTH Royal Institute of Technology",
    ],
    "references": [
        "Attention Is All You Need",
        "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks",
        "LightRAG: Simple and Fast Retrieval-Augmented Generation",
        "From Local to Global: A Graph RAG Approach to Query-Focused Summarization",
    ],
}

_INPUT_TEXT_RE = re.compile(r"---Input Text---(.*?)---Output---", re.DOTALL)
# "Title: ..." / "**Authors:** ..." header lines let the stub profile any document.
_HEADER_RE = re.compile(
    r"^\**(Title|Authors|Year|Venue|DOI|Affiliations|References)\**:\**"
    r"[ \t]*(.+?)[ \t]*$",
    re.MULTILINE,
)


def _split_list(value: str | None, separator: str) -> list[str]:
    if not value:
        return []
    return [part.strip() for part in value.split(separator) if part.strip()]


# --------------------------------------------------------------------------- #
# Stub LLM
# --------------------------------------------------------------------------- #


class StubLLM:
    """Prompt-routing stand-in for ``llm_model_func``.

    Every call is recorded in :attr:`calls` as ``{"kind": ..., "kwargs": [...]}``
    so tests can assert which pipeline stages ran.  Unrecognised prompts raise
    so that unexpected LLM usage fails loudly instead of silently degrading.
    """

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def __call__(
        self,
        prompt: str,
        system_prompt: str | None = None,
        history_messages: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> str | AsyncIterator[str]:
        kwargs.pop("hashing_kv", None)
        kind, response = self._route(prompt or "", system_prompt or "", kwargs)
        self.calls.append({"kind": kind, "kwargs": sorted(kwargs)})
        logger.debug("StubLLM handled a %s prompt", kind)
        if kwargs.get("stream"):
            return _as_stream(response)
        return response

    def count(self, kind: str) -> int:
        return sum(1 for call in self.calls if call["kind"] == kind)

    # -- routing ------------------------------------------------------------ #

    def _route(
        self, prompt: str, system_prompt: str, kwargs: dict[str, Any]
    ) -> tuple[str, str]:
        if "Based on the last extraction task" in prompt:
            return "gleaning", COMPLETION_DELIMITER
        if (
            "---Input Text---" in prompt
            and "Knowledge Graph Specialist" in system_prompt
        ):
            return "extract", self._extract(prompt)
        if not system_prompt and "proficient in data curation and synthesis" in prompt:
            return "summary", "Merged summary of the accumulated descriptions."
        if "high_level_keywords" in prompt and (
            kwargs.get("response_format") == JSON_OBJECT_FORMAT
        ):
            return "keywords", self._keywords(prompt)
        if PROFILER_MARKER in system_prompt:
            return "profile", self._profile(prompt)
        if (
            "synthesizing information from a provided knowledge base" in system_prompt
            or "---Context---" in system_prompt
        ):
            return "answer", self._answer(system_prompt)
        raise AssertionError(
            "StubLLM received an unrecognised prompt: "
            f"system={system_prompt[:120]!r} prompt={prompt[:160]!r}"
        )

    # -- responders --------------------------------------------------------- #

    @staticmethod
    def _extract(prompt: str) -> str:
        match = _INPUT_TEXT_RE.search(prompt)
        text = (match.group(1) if match else prompt).lower()
        present = [name for name in SAMPLE_ENTITIES if name.lower() in text]
        present_set = set(present)
        rows = [
            TUPLE_DELIMITER.join(
                ("entity", name, SAMPLE_ENTITIES[name][0], SAMPLE_ENTITIES[name][1])
            )
            for name in present
        ]
        rows.extend(
            TUPLE_DELIMITER.join(("relation", src, tgt, keywords, description))
            for src, tgt, keywords, description in SAMPLE_RELATIONS
            if src in present_set and tgt in present_set
        )
        rows.append(COMPLETION_DELIMITER)
        return "\n".join(rows)

    @staticmethod
    def _keywords(prompt: str) -> str:
        lowered = prompt.lower()
        low_level = [name for name in SAMPLE_ENTITIES if name.lower() in lowered]
        if not low_level:
            low_level = ["Sparse Attention Routing"]
        payload = {
            "high_level_keywords": [
                "retrieval-augmented generation",
                "long-document understanding",
            ],
            "low_level_keywords": low_level[:8],
        }
        return json.dumps(payload)

    @staticmethod
    def _profile(prompt: str) -> str:
        if SAMPLE_TITLE.lower() in prompt.lower():
            return json.dumps(SAMPLE_PROFILE)
        fields = {
            match.group(1).lower(): match.group(2).strip()
            for match in _HEADER_RE.finditer(prompt)
        }
        if "title" in fields:
            year = fields.get("year", "")
            return json.dumps(
                {
                    "title": fields["title"],
                    "authors": _split_list(fields.get("authors"), ","),
                    "year": int(year) if year.isdigit() else None,
                    "venue": fields.get("venue"),
                    "doi": fields.get("doi"),
                    "affiliations": _split_list(fields.get("affiliations"), ","),
                    "references": _split_list(fields.get("references"), ";"),
                }
            )
        title = next(
            (
                line.strip().lstrip("#").strip()
                for line in prompt.splitlines()
                if line.strip() and not line.startswith("---")
            ),
            "Untitled document",
        )
        return json.dumps(
            {
                "title": title,
                "authors": [],
                "year": None,
                "venue": None,
                "doi": None,
                "affiliations": [],
                "references": [],
            }
        )

    @staticmethod
    def _answer(system_prompt: str) -> str:
        mentioned = [
            name
            for name in ("Sparse Attention Routing", "Elena Marchetti", "EMNLP 2024")
            if name.lower() in system_prompt.lower()
        ]
        detail = ", ".join(mentioned) if mentioned else "the retrieved knowledge base"
        return f"{STUB_ANSWER_PREFIX}: grounded in {detail}."


async def _as_stream(text: str) -> AsyncIterator[str]:
    for chunk in text.split(" "):
        yield chunk + " "


# --------------------------------------------------------------------------- #
# Embeddings
# --------------------------------------------------------------------------- #

_WORD_RE = re.compile(r"\w+")


def hash_embedding(text: str, dim: int) -> np.ndarray:
    """Feature-hashing bag-of-words vector, L2-normalised, never all-zero."""
    vector = np.zeros(dim, dtype=np.float32)
    for token in _WORD_RE.findall(text.lower()):
        digest = hashlib.md5(token.encode("utf-8")).digest()  # noqa: S324
        index = int.from_bytes(digest[:4], "little") % dim
        sign = 1.0 if digest[4] & 1 else -1.0
        vector[index] += sign
    vector[0] += 1e-3
    return vector / np.linalg.norm(vector)


def stub_embedding_func(dim: int = 64) -> EmbeddingFunc:
    """Deterministic :class:`EmbeddingFunc` with ``dim`` dimensions."""

    async def _embed(texts: list[str]) -> np.ndarray:
        return np.stack([hash_embedding(text, dim) for text in texts]).astype(
            np.float32
        )

    return EmbeddingFunc(embedding_dim=dim, func=_embed, model_name=f"stub-hash-{dim}")


# --------------------------------------------------------------------------- #
# LightRAG construction helpers
# --------------------------------------------------------------------------- #


def offline_overrides() -> dict[str, Any]:
    """LightRAG kwargs that keep an offline run cheap and deterministic.

    With the codepoint tokenizer one token equals one character, hence the
    character-sized chunk settings and the very high summary threshold.
    """
    return {
        "entity_extract_max_gleaning": 0,
        "summary_max_tokens": 100_000,
        "chunk_token_size": 4000,
        "chunk_overlap_token_size": 200,
        "cosine_better_than_threshold": 0.05,
        "embedding_func_max_async": 2,
        "llm_model_max_async": 2,
        "max_parallel_insert": 1,
        "rerank_model_func": None,
    }


def offline_rag_kwargs(stub: StubLLM | None = None) -> dict[str, Any]:
    """Keyword arguments for :func:`rag_factory.build_rag` in offline mode."""
    return {
        "llm_func": stub or StubLLM(),
        "embedding_func": stub_embedding_func(),
        "tokenizer": codepoint_tokenizer(),
        **offline_overrides(),
    }
