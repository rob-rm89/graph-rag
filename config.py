"""Central configuration for the Unified Graph-RAG System.

This module owns three concerns that every other module shares:

* :class:`Settings` -- runtime settings resolved from environment variables
  (a ``.env`` file is honoured via ``python-dotenv``).
* The LitGraph entity-type registry and the prose *guidance* string that is
  injected into LightRAG's entity-extraction prompt through
  ``addon_params["entity_types_guidance"]``.
* The JSON Canvas colour presets used to visually separate bibliographic
  nodes from semantic concept nodes.
"""

from __future__ import annotations

import dataclasses
import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv
from lightrag.utils import logger

load_dotenv(override=False)

# --------------------------------------------------------------------------- #
# Runtime settings
# --------------------------------------------------------------------------- #

TEXT_SUFFIXES: frozenset[str] = frozenset({".txt", ".md"})
PDF_SUFFIX = ".pdf"
SUPPORTED_SUFFIXES: frozenset[str] = TEXT_SUFFIXES | {PDF_SUFFIX}


def _env(name: str, default: str) -> str:
    return os.getenv(name, default)


def _env_optional(name: str) -> str | None:
    value = os.getenv(name)
    return value if value else None


@dataclass(frozen=True)
class Settings:
    """Immutable runtime configuration.

    All defaults are read lazily from the environment so that tests can
    override variables before instantiation.  Use :meth:`with_overrides` to
    derive a modified copy.
    """

    working_dir: Path = field(
        default_factory=lambda: Path(_env("WORKING_DIR", "./rag_storage"))
    )
    data_dir: Path = field(default_factory=lambda: Path(_env("DATA_DIR", "./data")))
    canvas_path: Path = field(
        default_factory=lambda: Path(
            _env("CANVAS_PATH", "./output/knowledge_graph.canvas")
        )
    )
    extract_model: str = field(
        default_factory=lambda: _env("LLM_EXTRACT_MODEL", "gpt-4o-mini")
    )
    query_model: str = field(default_factory=lambda: _env("LLM_QUERY_MODEL", "gpt-4o"))
    embedding_model: str = field(
        default_factory=lambda: _env("EMBEDDING_MODEL", "text-embedding-3-small")
    )
    embedding_dim: int = field(
        default_factory=lambda: int(_env("EMBEDDING_DIM", "1536"))
    )
    openai_api_key: str | None = field(
        default_factory=lambda: _env_optional("OPENAI_API_KEY")
    )
    openai_base_url: str | None = field(
        default_factory=lambda: _env_optional("OPENAI_BASE_URL")
    )
    language: str = field(default_factory=lambda: _env("SUMMARY_LANGUAGE", "English"))
    max_gleaning: int = field(default_factory=lambda: int(_env("MAX_GLEANING", "1")))
    cosine_threshold: float = field(
        default_factory=lambda: float(_env("COSINE_THRESHOLD", "0.2"))
    )
    workspace: str = ""

    def with_overrides(self, **overrides: object) -> Settings:
        """Return a copy with the given fields replaced."""
        return dataclasses.replace(self, **overrides)  # type: ignore[arg-type]

    def ensure_directories(self) -> None:
        """Create the working, data and output directories if missing."""
        for path in (self.working_dir, self.data_dir, self.canvas_path.parent):
            path.mkdir(parents=True, exist_ok=True)
        logger.debug(
            "Directories ready: working_dir=%s data_dir=%s canvas_dir=%s",
            self.working_dir,
            self.data_dir,
            self.canvas_path.parent,
        )


# --------------------------------------------------------------------------- #
# LitGraph entity-type registry
# --------------------------------------------------------------------------- #

BIBLIOGRAPHIC_TYPES: dict[str, str] = {
    "Paper": (
        "The scholarly work described by the document itself, or any scholarly "
        "article explicitly named in the text. Use the full title as the entity name."
    ),
    "Author": (
        "A person credited as an author of a paper. Use the full personal name "
        "exactly as written."
    ),
    "Venue": (
        "The journal, conference, workshop, or publisher in which a paper appeared."
    ),
    "Year": (
        "A four-digit publication year attached to a paper, e.g. 2017. "
        "Use the bare four digits as the entity name."
    ),
    "CitedWork": (
        "A referenced publication listed in the bibliography or cited in the body "
        "that is not the document itself. Use its title, or first author plus year."
    ),
}

SEMANTIC_TYPES: dict[str, str] = {
    "Concept": "Abstract ideas, theories, principles, research problems, or topics.",
    "Method": "Procedures, algorithms, model architectures, techniques, or workflows.",
    "Dataset": "Named datasets, corpora, or benchmarks for training or evaluation.",
    "Metric": "Quantitative evaluation measures or reported results, e.g. BLEU.",
    "Organization": "Universities, companies, laboratories, or funding institutions.",
}

FALLBACK_TYPE = "Other"

# Relationship keyword vocabulary written by the bibliographic profiling pass.
REL_AUTHORED_BY = "authored_by"
REL_PUBLISHED_IN = "published_in"
REL_PUBLISHED_YEAR = "published_year"
REL_CITES = "cites"
REL_AFFILIATED_WITH = "affiliated_with"


def normalize_type(entity_type: str | None) -> str:
    """Normalise an entity type the way LightRAG stores it.

    LightRAG lower-cases extracted types and strips spaces (``"Cited Work"`` ->
    ``"citedwork"``), so every lookup in this project goes through this helper.
    """
    return (entity_type or "").replace(" ", "").lower()


BIBLIOGRAPHIC_KEYS: frozenset[str] = frozenset(
    normalize_type(name) for name in BIBLIOGRAPHIC_TYPES
)
SEMANTIC_KEYS: frozenset[str] = frozenset(
    normalize_type(name) for name in SEMANTIC_TYPES
)


def is_bibliographic(entity_type: str | None) -> bool:
    return normalize_type(entity_type) in BIBLIOGRAPHIC_KEYS


def build_entity_types_guidance() -> str:
    """Render the prose guidance injected into LightRAG's extraction prompt.

    The shape mirrors LightRAG's built-in ``default_entity_types_guidance``
    (a bulleted ``- Type: definition`` list with an ``Other`` fallback) and adds
    explicit LitGraph rules so bibliographic metadata is always emitted as
    first-class entities and relationships.
    """
    lines: list[str] = [
        "Classify each entity using one of the following types. "
        f"If no type fits, use `{FALLBACK_TYPE}`.",
        "",
        "Bibliographic types (LitGraph metadata). These are MANDATORY whenever "
        "the text contains them:",
    ]
    lines.extend(f"- {name}: {desc}" for name, desc in BIBLIOGRAPHIC_TYPES.items())
    lines.append("")
    lines.append("Semantic types:")
    lines.extend(f"- {name}: {desc}" for name, desc in SEMANTIC_TYPES.items())
    lines.extend(
        [
            "",
            "Bibliographic extraction rules:",
            "- Always emit the document's own title as a `Paper` entity.",
            "- Emit EVERY listed author as a separate `Author` entity and add a "
            "relationship from the `Paper` to each `Author` "
            f"(keywords: {REL_AUTHORED_BY}).",
            "- Emit the publication `Venue` and the four-digit `Year` as entities, "
            "each related to the `Paper` "
            f"(keywords: {REL_PUBLISHED_IN}, {REL_PUBLISHED_YEAR}).",
            "- Emit each bibliography entry as a `CitedWork` entity related to the "
            f"citing `Paper` (keywords: {REL_CITES}).",
            "- Relate each `Author` to their `Organization` affiliation when stated "
            f"(keywords: {REL_AFFILIATED_WITH}).",
            "- Semantic entities (concepts, methods, datasets, metrics) must be "
            "related to the `Paper` that discusses them.",
        ]
    )
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# JSON Canvas colour presets ("1" red, "2" orange, "3" yellow, "4" green,
# "5" cyan, "6" purple)
# --------------------------------------------------------------------------- #

NODE_COLOR_BY_TYPE: dict[str, str] = {
    "paper": "6",
    "author": "4",
    "venue": "5",
    "year": "3",
    "citedwork": "1",
    **{key: "2" for key in SEMANTIC_KEYS},
}
EDGE_COLOR_BIBLIO = "4"
EDGE_COLOR_SEMANTIC = "2"


def node_color(entity_type: str | None) -> str | None:
    """Preset colour for a node, or ``None`` to omit the attribute."""
    return NODE_COLOR_BY_TYPE.get(normalize_type(entity_type))


def edge_color(source_type: str | None, target_type: str | None) -> str:
    """Bibliographic edges are green ("4"); purely semantic edges are orange."""
    if is_bibliographic(source_type) or is_bibliographic(target_type):
        return EDGE_COLOR_BIBLIO
    return EDGE_COLOR_SEMANTIC
