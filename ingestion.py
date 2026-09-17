"""Ingestion Engine.

Loads academic text files, profiles their LitGraph bibliographic metadata with a
dedicated LLM prompt, appends that metadata as explicit nodes and edges through
LightRAG's custom knowledge-graph insertion, and finally runs LightRAG's own
graph-enhanced indexing (whose extraction prompt is biased towards the same
bibliographic types via ``config.build_entity_types_guidance``).

Order matters: the custom-KG path *overwrites* node attributes while the
extraction path *merges* them, so the explicit bibliographic nodes are written
first and enriched by extraction afterwards.
"""

from __future__ import annotations

import asyncio
import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from lightrag import LightRAG
from lightrag.utils import compute_mdhash_id, logger, setup_logger
from pypdf import PdfReader
from pypdf.errors import PyPdfError

from config import (
    PDF_SUFFIX,
    REL_AFFILIATED_WITH,
    REL_AUTHORED_BY,
    REL_CITES,
    REL_PUBLISHED_IN,
    REL_PUBLISHED_YEAR,
    SUPPORTED_SUFFIXES,
    TEXT_SUFFIXES,
    Settings,
    normalize_type,
)
from llm import LLMFunc, complete_json, make_llm_func
from rag_factory import rag_session

# --------------------------------------------------------------------------- #
# Profiling prompt
# --------------------------------------------------------------------------- #

PROFILER_MARKER = "LitGraph bibliographic profiler"

PROFILING_SYSTEM_PROMPT = f"""You are the {PROFILER_MARKER}: a bibliographic metadata \
extraction engine for academic documents.

Read the document excerpt supplied by the user and return ONLY a JSON object with \
exactly these keys:
- "title": string, the document's own title.
- "authors": array of strings, every author's full name in the order listed.
- "year": integer four-digit publication year, or null.
- "venue": string naming the journal, conference, workshop or publisher, or null.
- "doi": string, or null.
- "affiliations": array of strings, the institutions of the authors.
- "references": array of strings, the title of every work in the bibliography \
(use "First-author Year" when a title is absent).

Do not invent values. Use null or empty arrays when information is absent. \
Output JSON only, with no commentary."""

HEAD_CHARS = 6000
TAIL_CHARS = 4000
LEDGER_FILENAME = "litgraph_profiles.json"

_YEAR_RE = re.compile(r"\b(1[5-9]\d{2}|20\d{2})\b")


def build_profiling_prompt(text: str) -> str:
    """Send the document head (title/authors) and tail (references) to the LLM."""
    if len(text) <= HEAD_CHARS + TAIL_CHARS:
        excerpt = text
    else:
        excerpt = f"{text[:HEAD_CHARS]}\n\n[...]\n\n{text[-TAIL_CHARS:]}"
    return f"---Document---\n{excerpt}\n\n---Output JSON---"


# --------------------------------------------------------------------------- #
# Bibliographic record
# --------------------------------------------------------------------------- #


@dataclass
class BibliographicRecord:
    """Structured LitGraph metadata for one document."""

    title: str
    authors: list[str] = field(default_factory=list)
    year: int | None = None
    venue: str | None = None
    doi: str | None = None
    affiliations: list[str] = field(default_factory=list)
    references: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BibliographicRecord:
        return cls(
            title=str(data.get("title") or "Untitled document"),
            authors=list(data.get("authors") or []),
            year=data.get("year"),
            venue=data.get("venue"),
            doi=data.get("doi"),
            affiliations=list(data.get("affiliations") or []),
            references=list(data.get("references") or []),
        )


def _clean_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip().strip("\"'")
    return text or None


def _clean_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, list | tuple | set):
        return []
    seen: set[str] = set()
    cleaned: list[str] = []
    for item in value:
        text = _clean_str(item)
        if text and text.lower() not in seen:
            seen.add(text.lower())
            cleaned.append(text)
    return cleaned


def _coerce_year(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value if 1500 <= value <= 2099 else None
    match = _YEAR_RE.search(str(value or ""))
    return int(match.group(1)) if match else None


def parse_profile(raw: dict[str, Any], *, fallback_title: str) -> BibliographicRecord:
    """Coerce a (possibly sloppy) LLM JSON object into a clean record."""
    title = _clean_str(raw.get("title")) or fallback_title
    record = BibliographicRecord(
        title=title,
        authors=_clean_list(raw.get("authors")),
        year=_coerce_year(raw.get("year")),
        venue=_clean_str(raw.get("venue")),
        doi=_clean_str(raw.get("doi")),
        affiliations=_clean_list(raw.get("affiliations")),
        references=[ref for ref in _clean_list(raw.get("references")) if ref != title],
    )
    logger.debug("Parsed bibliographic record: %s", record)
    return record


def fallback_title(text: str, name: str) -> str:
    """First Markdown heading or non-empty line, else the file stem."""
    for line in text.splitlines():
        stripped = line.strip().lstrip("#").strip()
        if stripped:
            return stripped[:200]
    return Path(name).stem


# --------------------------------------------------------------------------- #
# Record -> LightRAG custom knowledge graph
# --------------------------------------------------------------------------- #


def _header_text(record: BibliographicRecord) -> str:
    lines = [f"Title: {record.title}"]
    if record.authors:
        lines.append(f"Authors: {', '.join(record.authors)}")
    if record.venue:
        lines.append(f"Venue: {record.venue}")
    if record.year:
        lines.append(f"Year: {record.year}")
    if record.doi:
        lines.append(f"DOI: {record.doi}")
    if record.affiliations:
        lines.append(f"Affiliations: {', '.join(record.affiliations)}")
    if record.references:
        lines.append("References: " + "; ".join(record.references))
    return "\n".join(lines)


def record_to_custom_kg(
    record: BibliographicRecord, *, source_alias: str, file_path: str
) -> dict[str, list[dict[str, Any]]]:
    """Translate a record into LightRAG's ``ainsert_custom_kg`` payload.

    Pure function.  Entity types are emitted already normalised (lower-case,
    no spaces) to match what LightRAG's extraction pipeline stores, so later
    merges never split the type vote.  Never emits self-loops or duplicates.
    """
    paper = record.title.strip()
    entities: list[dict[str, Any]] = []
    relationships: list[dict[str, Any]] = []
    seen_entities: set[str] = set()
    seen_edges: set[tuple[str, str]] = set()

    def add_entity(name: str | None, entity_type: str, description: str) -> str | None:
        clean = _clean_str(name)
        if not clean or clean in seen_entities:
            return clean
        seen_entities.add(clean)
        entities.append(
            {
                "entity_name": clean,
                "entity_type": normalize_type(entity_type),
                "description": description,
                "source_id": source_alias,
                "file_path": file_path,
            }
        )
        return clean

    def add_relation(
        src: str | None, tgt: str | None, keywords: str, description: str
    ) -> None:
        if not src or not tgt or src == tgt or (src, tgt) in seen_edges:
            return
        seen_edges.add((src, tgt))
        relationships.append(
            {
                "src_id": src,
                "tgt_id": tgt,
                "description": description,
                "keywords": keywords,
                "weight": 1.0,
                "source_id": source_alias,
                "file_path": file_path,
            }
        )

    paper_desc = f"Academic paper titled '{paper}'"
    if record.venue:
        paper_desc += f", published in {record.venue}"
    if record.year:
        paper_desc += f" ({record.year})"
    paper_desc += "."
    add_entity(paper, "Paper", paper_desc)

    for author in record.authors:
        name = add_entity(author, "Author", f"Author of '{paper}'.")
        add_relation(paper, name, REL_AUTHORED_BY, f"'{paper}' was authored by {name}.")

    if record.venue:
        venue = add_entity(record.venue, "Venue", f"Publication venue of '{paper}'.")
        add_relation(
            paper, venue, REL_PUBLISHED_IN, f"'{paper}' was published in {venue}."
        )

    if record.year:
        year = add_entity(str(record.year), "Year", f"Publication year of '{paper}'.")
        add_relation(
            paper, year, REL_PUBLISHED_YEAR, f"'{paper}' was published in {year}."
        )

    for reference in record.references:
        cited = add_entity(reference, "CitedWork", f"Work cited by '{paper}'.")
        add_relation(paper, cited, REL_CITES, f"'{paper}' cites '{cited}'.")

    for affiliation in record.affiliations:
        org = add_entity(
            affiliation, "Organization", f"Institution affiliated with '{paper}'."
        )
        # Without per-author affiliation data, attach the institution to the
        # paper's authors collectively (one edge per author keeps the graph honest
        # about what the profiler actually knows).
        for author in record.authors:
            add_relation(
                _clean_str(author),
                org,
                REL_AFFILIATED_WITH,
                f"{author} is listed with affiliation {org} on '{paper}'.",
            )

    chunk = {
        "content": _header_text(record),
        "source_id": source_alias,
        "file_path": file_path,
        "chunk_order_index": 0,
    }
    return {"chunks": [chunk], "entities": entities, "relationships": relationships}


# --------------------------------------------------------------------------- #
# Document loading
# --------------------------------------------------------------------------- #

_HYPHEN_BREAK_RE = re.compile(r"(\w)-\n(\w)")
_MULTI_BLANK_RE = re.compile(r"\n{3,}")


def normalise_pdf_text(text: str) -> str:
    """Repair end-of-line hyphenation and collapse runs of blank lines."""
    text = _HYPHEN_BREAK_RE.sub(r"\1\2", text)
    return _MULTI_BLANK_RE.sub("\n\n", text).strip()


def extract_pdf_text(path: Path) -> str:
    """Extract text from every page of a PDF (pages joined by blank lines).

    Raises ``ValueError`` for unreadable or password-protected files.  Scanned
    PDFs without a text layer yield an empty string and are rejected by
    :func:`read_document`.
    """
    try:
        reader = PdfReader(str(path))
        if reader.is_encrypted:
            reader.decrypt("")  # owner-password-only PDFs open with an empty password
        pages = [page.extract_text() or "" for page in reader.pages]
    except (PyPdfError, OSError, ValueError) as exc:
        raise ValueError(f"cannot read PDF {path.name}: {exc}") from exc
    text = normalise_pdf_text(
        "\n\n".join(page.strip() for page in pages if page.strip())
    )
    logger.info(
        "Extracted %d characters from %d PDF page(s) in %s",
        len(text),
        len(pages),
        path.name,
    )
    return text


def read_document(path: Path) -> str:
    """Return the plain text of a supported document (``.txt``, ``.md``, ``.pdf``).

    Raises ``ValueError`` for unsupported types and for files without text.
    """
    suffix = path.suffix.lower()
    if suffix == PDF_SUFFIX:
        text = extract_pdf_text(path)
    elif suffix in TEXT_SUFFIXES:
        text = path.read_text(encoding="utf-8", errors="replace")
    else:
        raise ValueError(f"unsupported file type {suffix!r}: {path.name}")
    if not text.strip():
        raise ValueError(f"{path.name} contains no extractable text")
    return text


# --------------------------------------------------------------------------- #
# Engine
# --------------------------------------------------------------------------- #


@dataclass
class IngestionReport:
    ingested: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    profiles: dict[str, BibliographicRecord] = field(default_factory=dict)


class IngestionEngine:
    """Drive profiling + LightRAG indexing for every document in ``data_dir``."""

    def __init__(self, rag: LightRAG, settings: Settings, profiling_llm: LLMFunc):
        self._rag = rag
        self._settings = settings
        self._llm = profiling_llm
        self._ledger_path = settings.working_dir / LEDGER_FILENAME
        self._ledger: dict[str, dict[str, Any]] = self._load_ledger()

    # -- discovery ---------------------------------------------------------- #

    def discover_documents(self) -> list[Path]:
        data_dir = self._settings.data_dir
        if not data_dir.is_dir():
            logger.warning("Data directory %s does not exist", data_dir)
            return []
        docs = sorted(
            p
            for p in data_dir.iterdir()
            if p.is_file() and p.suffix.lower() in SUPPORTED_SUFFIXES
        )
        logger.info("Discovered %d document(s) in %s", len(docs), data_dir)
        return docs

    # -- profiling ---------------------------------------------------------- #

    async def profile_document(self, text: str, name: str) -> BibliographicRecord:
        logger.info("Profiling bibliographic metadata for %s", name)
        raw = await complete_json(
            self._llm, PROFILING_SYSTEM_PROMPT, build_profiling_prompt(text)
        )
        if not raw:
            logger.warning("Profiler returned no JSON for %s; using fallbacks", name)
        record = parse_profile(raw, fallback_title=fallback_title(text, name))
        logger.info(
            "Profile for %s: title=%r authors=%d year=%s venue=%r refs=%d",
            name,
            record.title,
            len(record.authors),
            record.year,
            record.venue,
            len(record.references),
        )
        return record

    # -- ingestion ---------------------------------------------------------- #

    async def ingest_text(
        self, text: str, name: str
    ) -> tuple[str, BibliographicRecord]:
        doc_id = compute_mdhash_id(text.strip(), prefix="doc-")
        cached = self._ledger.get(doc_id)
        if cached:
            record = BibliographicRecord.from_dict(cached)
            logger.info("Bibliographic nodes for %s already present; skipping", name)
        else:
            record = await self.profile_document(text, name)
            custom_kg = record_to_custom_kg(
                record, source_alias=f"{doc_id}-biblio", file_path=name
            )
            logger.info(
                "Appending %d bibliographic entities and %d relationships for %s",
                len(custom_kg["entities"]),
                len(custom_kg["relationships"]),
                name,
            )
            await self._rag.ainsert_custom_kg(custom_kg, full_doc_id=doc_id)
            self._ledger[doc_id] = record.to_dict()
            self._save_ledger()

        logger.info("Indexing %s through LightRAG (doc_id=%s)", name, doc_id)
        await self._rag.ainsert(text, ids=[doc_id], file_paths=[name])
        return doc_id, record

    async def ingest_document(self, path: Path) -> tuple[str, BibliographicRecord]:
        text = read_document(path)
        return await self.ingest_text(text, path.name)

    async def ingest_all(self) -> IngestionReport:
        report = IngestionReport()
        for path in self.discover_documents():
            try:
                doc_id, record = await self.ingest_document(path)
            except ValueError as exc:
                logger.warning("Skipping %s: %s", path.name, exc)
                report.skipped.append(path.name)
                continue
            report.ingested.append(path.name)
            report.profiles[doc_id] = record
        logger.info(
            "Ingestion finished: %d ingested, %d skipped",
            len(report.ingested),
            len(report.skipped),
        )
        return report

    # -- ledger ------------------------------------------------------------- #

    def _load_ledger(self) -> dict[str, dict[str, Any]]:
        if not self._ledger_path.exists():
            return {}
        try:
            data = json.loads(self._ledger_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning(
                "Could not read profile ledger %s: %s", self._ledger_path, exc
            )
            return {}
        return data if isinstance(data, dict) else {}

    def _save_ledger(self) -> None:
        self._ledger_path.parent.mkdir(parents=True, exist_ok=True)
        self._ledger_path.write_text(
            json.dumps(self._ledger, indent=2, ensure_ascii=False), encoding="utf-8"
        )


# --------------------------------------------------------------------------- #
# CLI entry point
# --------------------------------------------------------------------------- #


async def main() -> None:
    setup_logger("lightrag", level="INFO", enable_file_logging=False)
    settings = Settings()
    async with rag_session(settings) as rag:
        engine = IngestionEngine(
            rag, settings, make_llm_func(settings.extract_model, settings)
        )
        report = await engine.ingest_all()
    for name in report.ingested:
        logger.info("Ingested %s", name)


if __name__ == "__main__":
    asyncio.run(main())
