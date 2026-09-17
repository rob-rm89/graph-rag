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
from metadata_lookup import MetadataResolver
from rag_factory import rag_session
from reconciliation import REGISTRY_FILENAME, EntityRegistry, MergePlan, Reconciler

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
    # Optional precision data, typically filled by metadata_lookup.
    author_affiliations: dict[str, list[str]] = field(default_factory=dict)
    reference_dois: dict[str, str] = field(default_factory=dict)
    openalex_id: str | None = None
    metadata_source: str | None = None

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
            author_affiliations={
                str(author): list(institutions)
                for author, institutions in (
                    data.get("author_affiliations") or {}
                ).items()
            },
            reference_dois=dict(data.get("reference_dois") or {}),
            openalex_id=data.get("openalex_id"),
            metadata_source=data.get("metadata_source"),
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
    if record.openalex_id:
        lines.append(f"OpenAlex: {record.openalex_id}")
    if record.affiliations:
        lines.append(f"Affiliations: {', '.join(record.affiliations)}")
    if record.references:
        lines.append("References: " + "; ".join(record.references))
    if record.metadata_source:
        lines.append(f"Metadata source: {record.metadata_source}")
    return "\n".join(lines)


def record_to_custom_kg(
    record: BibliographicRecord,
    *,
    source_alias: str,
    file_path: str,
    existing_entities: frozenset[str] = frozenset(),
) -> dict[str, list[dict[str, Any]]]:
    """Translate a record into LightRAG's ``ainsert_custom_kg`` payload.

    Pure function.  Entity types are emitted already normalised (lower-case,
    no spaces) to match what LightRAG's extraction pipeline stores, so later
    merges never split the type vote.  Never emits self-loops or duplicates.

    Names in ``existing_entities`` already exist in the graph (resolved by the
    reconciler): they receive relationships but no entity row, because the
    custom-KG path overwrites node attributes.  The paper itself is always
    written since its own profile is authoritative.
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
        if clean in existing_entities and clean != paper:
            return clean
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

    org_desc = f"Institution affiliated with '{paper}'."
    if record.author_affiliations:
        # Authoritative per-author institutions (from OpenAlex/Crossref).
        attributed: set[str] = set()
        for author, institutions in record.author_affiliations.items():
            name = _clean_str(author)
            for institution in institutions:
                org = add_entity(institution, "Organization", org_desc)
                if org:
                    attributed.add(org)
                add_relation(
                    name,
                    org,
                    REL_AFFILIATED_WITH,
                    f"{name} is affiliated with {org} on '{paper}'.",
                )
        for affiliation in record.affiliations:
            org = add_entity(affiliation, "Organization", org_desc)
            if org and org not in attributed:
                add_relation(
                    paper,
                    org,
                    REL_AFFILIATED_WITH,
                    f"'{paper}' lists the affiliation {org}.",
                )
    else:
        for affiliation in record.affiliations:
            org = add_entity(affiliation, "Organization", org_desc)
            # Without per-author affiliation data, attach the institution to the
            # paper's authors collectively (one edge per author keeps the graph
            # honest about what the profiler actually knows).
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
class IngestResult:
    doc_id: str
    record: BibliographicRecord
    merges: list[MergePlan] = field(default_factory=list)
    linked_papers: list[str] = field(default_factory=list)


@dataclass
class IngestionReport:
    ingested: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    profiles: dict[str, BibliographicRecord] = field(default_factory=dict)
    merges: list[MergePlan] = field(default_factory=list)
    linked_papers: dict[str, list[str]] = field(default_factory=dict)


class IngestionEngine:
    """Drive profiling, reconciliation and LightRAG indexing for ``data_dir``."""

    def __init__(
        self,
        rag: LightRAG,
        settings: Settings,
        profiling_llm: LLMFunc,
        *,
        reconcile_graph: bool | None = None,
        metadata_resolver: MetadataResolver | None = None,
    ):
        self._rag = rag
        self._settings = settings
        self._llm = profiling_llm
        self._resolver = metadata_resolver
        self._ledger_path = settings.working_dir / LEDGER_FILENAME
        self._ledger: dict[str, dict[str, Any]] = self._load_ledger()
        self._registry = EntityRegistry(settings.working_dir / REGISTRY_FILENAME)
        self._reconciler = Reconciler(self._registry)
        self._reconcile_after_ingest = (
            settings.reconcile_graph if reconcile_graph is None else reconcile_graph
        )

    @property
    def registry(self) -> EntityRegistry:
        return self._registry

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

    async def ingest_text(self, text: str, name: str) -> IngestResult:
        doc_id = compute_mdhash_id(text.strip(), prefix="doc-")
        cached = self._ledger.get(doc_id)
        merges: list[MergePlan] = []
        linked: list[str] = []
        if cached:
            record = BibliographicRecord.from_dict(cached)
            logger.info("Bibliographic nodes for %s already present; skipping", name)
        else:
            profile = await self.profile_document(text, name)
            if self._resolver is not None:
                profile = await self._resolver.enrich(profile)
            reconciliation = self._reconciler.reconcile(profile, doc_id)
            record = reconciliation.record
            linked = reconciliation.linked_papers
            custom_kg = record_to_custom_kg(
                record,
                source_alias=f"{doc_id}-biblio",
                file_path=name,
                existing_entities=reconciliation.existing,
            )
            logger.info(
                "Appending %d bibliographic entities and %d relationships for %s "
                "(%d resolved to existing nodes, %d cross-paper citations)",
                len(custom_kg["entities"]),
                len(custom_kg["relationships"]),
                name,
                len(reconciliation.existing),
                len(linked),
            )
            await self._rag.ainsert_custom_kg(custom_kg, full_doc_id=doc_id)
            merges = await self._apply_merges(reconciliation.merges)
            self._registry.save()
            self._ledger[doc_id] = record.to_dict()
            self._save_ledger()

        logger.info("Indexing %s through LightRAG (doc_id=%s)", name, doc_id)
        await self._rag.ainsert(text, ids=[doc_id], file_paths=[name])
        return IngestResult(doc_id, record, merges, linked)

    async def ingest_document(self, path: Path) -> IngestResult:
        text = read_document(path)
        return await self.ingest_text(text, path.name)

    async def ingest_all(self) -> IngestionReport:
        report = IngestionReport()
        for path in self.discover_documents():
            try:
                result = await self.ingest_document(path)
            except ValueError as exc:
                logger.warning("Skipping %s: %s", path.name, exc)
                report.skipped.append(path.name)
                continue
            report.ingested.append(path.name)
            report.profiles[result.doc_id] = result.record
            report.merges.extend(result.merges)
            if result.linked_papers:
                report.linked_papers[result.record.title] = result.linked_papers
        if self._reconcile_after_ingest and report.ingested:
            report.merges.extend(await self.reconcile_graph())
        logger.info(
            "Ingestion finished: %d ingested, %d skipped, %d entity merge(s)",
            len(report.ingested),
            len(report.skipped),
            len(report.merges),
        )
        return report

    # -- reconciliation ----------------------------------------------------- #

    async def _apply_merges(self, plans: list[MergePlan]) -> list[MergePlan]:
        """Execute merge plans through LightRAG, skipping stale or missing nodes."""
        storage = self._rag.chunk_entity_relation_graph
        applied: list[MergePlan] = []
        for plan in plans:
            sources = [
                source
                for source in plan.sources
                if source != plan.target and await storage.has_node(source)
            ]
            if not sources:
                continue
            if not await storage.has_node(plan.target):
                logger.warning(
                    "Merge target %r is not in the graph; leaving %s unmerged",
                    plan.target,
                    sources,
                )
                continue
            try:
                await self._rag.amerge_entities(sources, plan.target)
            except Exception as exc:  # noqa: BLE001 - a failed merge must not abort ingest
                logger.error("Merging %s into %r failed: %s", sources, plan.target, exc)
                continue
            logger.info("Merged %s into %r (%s)", sources, plan.target, plan.reason)
            applied.append(MergePlan(tuple(sources), plan.target, plan.reason))
        return applied

    async def reconcile_graph(self) -> list[MergePlan]:
        """Merge near-duplicate nodes already in the graph (graph-wide pass)."""
        storage = self._rag.chunk_entity_relation_graph
        nodes = await storage.get_all_nodes()
        kinds = {
            str(node["id"]): normalize_type(node.get("entity_type"))
            for node in nodes
            if node.get("id")
        }
        plans = self._reconciler.plan_graph_merges(nodes)
        applied = await self._apply_merges(plans)
        for plan in applied:
            self._reconciler.absorb_merge(plan, kind=kinds.get(plan.target))
        self._registry.save()
        logger.info("Graph reconciliation applied %d merge(s)", len(applied))
        return applied

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
    resolver = (
        MetadataResolver.from_settings(settings) if settings.metadata_lookup else None
    )
    try:
        async with rag_session(settings) as rag:
            engine = IngestionEngine(
                rag,
                settings,
                make_llm_func(settings.extract_model, settings),
                metadata_resolver=resolver,
            )
            report = await engine.ingest_all()
    finally:
        if resolver is not None:
            await resolver.aclose()
    for name in report.ingested:
        logger.info("Ingested %s", name)


if __name__ == "__main__":
    asyncio.run(main())
