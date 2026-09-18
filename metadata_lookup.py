"""External bibliographic metadata lookup: OpenAlex first, Crossref as fallback.

The LLM profiler guesses title, authors, year, venue and references from the
document text.  When the paper can be identified in OpenAlex or Crossref those
fields are replaced or completed with authoritative values, per-author
institutions become available, and the reference list gains DOIs, which makes
:mod:`reconciliation` far more precise.

Design constraints:

* Never fails ingestion: every network or parsing problem degrades to
  "no enrichment" with a warning.
* Fully testable offline through an injected ``httpx.AsyncClient`` (the tests
  use ``httpx.MockTransport``).
* Results, including misses, are cached on disk (``metadata_cache.json``) so
  re-ingestion and repeated lookups cost no requests.
* No contact address is sent unless ``METADATA_MAILTO`` is configured.
"""

from __future__ import annotations

import asyncio
import json
import re
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

import httpx
from lightrag.utils import logger

from reconciliation import (
    normalize_doi,
    normalize_title,
    parse_person,
    title_similarity,
    titles_match,
)

if TYPE_CHECKING:  # pragma: no cover - import cycle guard
    from config import Settings
    from ingestion import BibliographicRecord

OPENALEX_API = "https://api.openalex.org"
CROSSREF_API = "https://api.crossref.org"
CACHE_FILENAME = "metadata_cache.json"
USER_AGENT = "graph-rag/0.1 (+https://github.com/rob-rm89/graph-rag)"
OPENALEX_SELECT = (
    "id,doi,title,display_name,publication_year,authorships,primary_location,"
    "referenced_works"
)
OPENALEX_REFERENCE_SELECT = "id,doi,title,display_name,publication_year"
CROSSREF_SEARCH_SELECT = "DOI,title,author,issued,container-title"
SEARCH_RESULTS = 10
BATCH_SIZE = 50
MIN_TITLE_SIMILARITY = 0.9
REPLACE_TITLE_SIMILARITY = 0.95
SOURCE_OPENALEX = "openalex"
SOURCE_CROSSREF = "crossref"


# --------------------------------------------------------------------------- #
# Data model
# --------------------------------------------------------------------------- #


@dataclass
class ExternalReference:
    title: str
    doi: str | None = None


@dataclass
class ExternalWork:
    """A work as described by an external catalogue."""

    source: str
    id: str
    title: str
    doi: str | None = None
    year: int | None = None
    venue: str | None = None
    authors: list[str] = field(default_factory=list)
    author_affiliations: dict[str, list[str]] = field(default_factory=dict)
    # External identifiers per author / institution name, e.g.
    # {"John Smith": ["openalex:A123", "orcid:0000-..."]}.
    author_ids: dict[str, list[str]] = field(default_factory=dict)
    institution_ids: dict[str, list[str]] = field(default_factory=dict)
    references: list[ExternalReference] = field(default_factory=list)
    reference_ids: list[str] = field(default_factory=list)
    matched_by: str = "title"  # "doi" or "title"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ExternalWork:
        payload = dict(data)
        payload["references"] = [
            ExternalReference(**ref) for ref in payload.get("references") or []
        ]
        return cls(**payload)


def _clean(value: Any) -> str | None:
    if value is None:
        return None
    text = " ".join(str(value).split())
    return text or None


def _first(value: Any) -> Any:
    if isinstance(value, list):
        return value[0] if value else None
    return value


def _year(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int) and 1500 <= value <= 2100:
        return value
    return None


def _dedupe(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if item and item not in seen:
            seen.add(item)
            out.append(item)
    return out


def short_openalex_id(value: Any) -> str:
    return str(value or "").rsplit("/", 1)[-1]


_ORCID_RE = re.compile(r"(\d{4}-\d{4}-\d{4}-\d{3}[\dX])", re.IGNORECASE)
_OPENALEX_ID_RE = re.compile(r"^[AIWSPF]\d+$")


def openalex_identifier(value: Any) -> str | None:
    """``https://openalex.org/A123`` -> ``openalex:A123``."""
    short = short_openalex_id(value)
    return f"openalex:{short}" if _OPENALEX_ID_RE.match(short) else None


def orcid_identifier(value: Any) -> str | None:
    """Any ORCID URL or bare id -> ``orcid:0000-0002-1825-0097``."""
    match = _ORCID_RE.search(str(value or ""))
    return f"orcid:{match.group(1).upper()}" if match else None


def ror_identifier(value: Any) -> str | None:
    """``https://ror.org/00f54p054`` -> ``ror:00f54p054``."""
    text = str(value or "").strip().rstrip("/")
    return f"ror:{text.rsplit('/', 1)[-1]}" if text else None


def _add_identifiers(
    store: dict[str, list[str]], name: str, ids: list[str | None]
) -> None:
    for identifier in ids:
        if identifier:
            store.setdefault(name, [])
            if identifier not in store[name]:
                store[name].append(identifier)


# --------------------------------------------------------------------------- #
# Parsers
# --------------------------------------------------------------------------- #


def parse_openalex_work(data: dict[str, Any], matched_by: str) -> ExternalWork:
    authors: list[str] = []
    affiliations: dict[str, list[str]] = {}
    author_ids: dict[str, list[str]] = {}
    institution_ids: dict[str, list[str]] = {}
    for authorship in data.get("authorships") or []:
        author = authorship.get("author") or {}
        name = _clean(author.get("display_name")) or _clean(
            authorship.get("raw_author_name")
        )
        if not name:
            continue
        if name not in authors:
            authors.append(name)
        _add_identifiers(
            author_ids,
            name,
            [
                openalex_identifier(author.get("id")),
                orcid_identifier(author.get("orcid")),
            ],
        )
        for inst in authorship.get("institutions") or []:
            institution = _clean(inst.get("display_name"))
            if not institution:
                continue
            affiliations.setdefault(name, [])
            if institution not in affiliations[name]:
                affiliations[name].append(institution)
            _add_identifiers(
                institution_ids,
                institution,
                [openalex_identifier(inst.get("id")), ror_identifier(inst.get("ror"))],
            )
    location = data.get("primary_location") or {}
    source = (location.get("source") or {}).get("display_name")
    return ExternalWork(
        source=SOURCE_OPENALEX,
        id=short_openalex_id(data.get("id")),
        title=_clean(data.get("title") or data.get("display_name")) or "",
        doi=normalize_doi(data.get("doi")),
        year=_year(data.get("publication_year")),
        venue=_clean(source),
        authors=authors,
        author_affiliations=affiliations,
        author_ids=author_ids,
        institution_ids=institution_ids,
        reference_ids=[
            short_openalex_id(ref) for ref in data.get("referenced_works") or []
        ],
        matched_by=matched_by,
    )


def parse_crossref_work(message: dict[str, Any], matched_by: str) -> ExternalWork:
    authors: list[str] = []
    affiliations: dict[str, list[str]] = {}
    author_ids: dict[str, list[str]] = {}
    for author in message.get("author") or []:
        name = _clean(
            " ".join(
                part for part in (author.get("given"), author.get("family")) if part
            )
        ) or _clean(author.get("name"))
        if not name:
            continue
        if name not in authors:
            authors.append(name)
        _add_identifiers(author_ids, name, [orcid_identifier(author.get("ORCID"))])
        for affiliation in author.get("affiliation") or []:
            institution = _clean(affiliation.get("name"))
            if institution:
                affiliations.setdefault(name, [])
                if institution not in affiliations[name]:
                    affiliations[name].append(institution)

    year: int | None = None
    for key in ("issued", "published-print", "published-online", "created"):
        parts = (message.get(key) or {}).get("date-parts") or []
        if parts and parts[0]:
            year = _year(parts[0][0])
            if year:
                break

    references: list[ExternalReference] = []
    for ref in message.get("reference") or []:
        doi = normalize_doi(ref.get("DOI"))
        title = _clean(
            ref.get("article-title")
            or ref.get("volume-title")
            or ref.get("unstructured")
        )
        if title:
            references.append(ExternalReference(title=title[:200], doi=doi))
        elif doi:
            references.append(ExternalReference(title=doi, doi=doi))

    doi = normalize_doi(message.get("DOI"))
    title = _clean(_first(message.get("title"))) or ""
    return ExternalWork(
        source=SOURCE_CROSSREF,
        id=doi or title,
        title=title,
        doi=doi,
        year=year,
        venue=_clean(_first(message.get("container-title"))),
        authors=authors,
        author_affiliations=affiliations,
        author_ids=author_ids,
        references=references,
        matched_by=matched_by,
    )


# --------------------------------------------------------------------------- #
# API clients
# --------------------------------------------------------------------------- #


class OpenAlexClient:
    def __init__(
        self,
        http: httpx.AsyncClient,
        mailto: str | None = None,
        base_url: str = OPENALEX_API,
    ) -> None:
        self.http = http
        self.mailto = mailto
        self.base_url = base_url.rstrip("/")

    def _params(self, **params: Any) -> dict[str, Any]:
        if self.mailto:
            params["mailto"] = self.mailto
        return params

    async def work_by_doi(self, doi: str) -> dict[str, Any] | None:
        response = await self.http.get(
            f"{self.base_url}/works/doi:{doi}",
            params=self._params(select=OPENALEX_SELECT),
        )
        if response.status_code == 404:
            return None
        response.raise_for_status()
        return response.json()

    async def search(self, title: str) -> list[dict[str, Any]]:
        response = await self.http.get(
            f"{self.base_url}/works",
            params=self._params(
                search=title, **{"per-page": SEARCH_RESULTS}, select=OPENALEX_SELECT
            ),
        )
        response.raise_for_status()
        return list(response.json().get("results") or [])

    async def works_by_ids(self, ids: list[str]) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        for start in range(0, len(ids), BATCH_SIZE):
            batch = ids[start : start + BATCH_SIZE]
            response = await self.http.get(
                f"{self.base_url}/works",
                params=self._params(
                    filter=f"ids.openalex:{'|'.join(batch)}",
                    **{"per-page": BATCH_SIZE},
                    select=OPENALEX_REFERENCE_SELECT,
                ),
            )
            response.raise_for_status()
            results.extend(response.json().get("results") or [])
        return results


class CrossrefClient:
    def __init__(
        self,
        http: httpx.AsyncClient,
        mailto: str | None = None,
        base_url: str = CROSSREF_API,
    ) -> None:
        self.http = http
        self.base_url = base_url.rstrip("/")
        agent = USER_AGENT + (f" (mailto:{mailto})" if mailto else "")
        self.headers = {"User-Agent": agent}

    async def work_by_doi(self, doi: str) -> dict[str, Any] | None:
        response = await self.http.get(
            f"{self.base_url}/works/{doi}", headers=self.headers
        )
        if response.status_code == 404:
            return None
        response.raise_for_status()
        return response.json().get("message")

    async def search(self, title: str) -> list[dict[str, Any]]:
        response = await self.http.get(
            f"{self.base_url}/works",
            params={
                "query.bibliographic": title,
                "rows": SEARCH_RESULTS,
                "select": CROSSREF_SEARCH_SELECT,
            },
            headers=self.headers,
        )
        response.raise_for_status()
        return list((response.json().get("message") or {}).get("items") or [])


# --------------------------------------------------------------------------- #
# Candidate selection and record merging (pure functions)
# --------------------------------------------------------------------------- #


def pick_best_candidate(
    candidates: list[ExternalWork],
    title: str,
    year: int | None = None,
    authors: list[str] | None = None,
) -> ExternalWork | None:
    """Choose the candidate whose title, authors and year best fit the record.

    Catalogues return near-duplicates (reprints, homonymous titles), so the
    first hit is never trusted blindly.  A title must match; when both sides
    list authors at least one family name must agree; a publication year more
    than one year off rejects the candidate unless author overlap vouches for
    it, in which case it only lowers the score.
    """
    target = normalize_title(title)
    wanted = {
        person.family
        for name in authors or []
        if (person := parse_person(name)) and person.family
    }
    best: ExternalWork | None = None
    best_score = 0.0
    for candidate in candidates:
        cand_norm = normalize_title(candidate.title)
        if not titles_match(target, cand_norm, threshold=MIN_TITLE_SIMILARITY):
            continue
        score = title_similarity(target, cand_norm)
        overlap_ratio = 0.0
        if wanted and candidate.authors:
            families = {
                person.family
                for name in candidate.authors
                if (person := parse_person(name)) and person.family
            }
            overlap = len(wanted & families)
            if overlap == 0:
                continue  # both sides name authors and none agree
            overlap_ratio = overlap / len(wanted)
            score += 0.1 * overlap_ratio
        if year and candidate.year:
            if candidate.year == year:
                score += 0.05
            elif abs(candidate.year - year) > 1:
                # Catalogues mislabel years (reprints, versions): the year is a
                # veto only when no author evidence supports the candidate.
                if not overlap_ratio:
                    continue
                score -= 0.1
        if score > best_score:
            best, best_score = candidate, score
    return best


def enrich_record(
    record: BibliographicRecord, work: ExternalWork, *, max_references: int
) -> BibliographicRecord:
    """Merge an external work into a profiled record.

    External values win for authors, year, venue and DOI.  The title is
    replaced when the match came from a DOI or is near-identical.  References
    extracted from the document are kept first (they are in-text evidence),
    then catalogue references are appended up to ``max_references``; DOIs are
    attached to whichever spelling survives.
    """
    similar = title_similarity(
        normalize_title(record.title), normalize_title(work.title)
    )
    title = (
        work.title
        if work.title
        and (work.matched_by == "doi" or similar >= REPLACE_TITLE_SIMILARITY)
        else record.title
    )
    author_affiliations = {
        author: list(institutions)
        for author, institutions in work.author_affiliations.items()
    }
    affiliations = _dedupe(
        list(record.affiliations)
        + [inst for insts in author_affiliations.values() for inst in insts]
    )

    references: list[str] = []
    reference_dois = dict(record.reference_dois)
    by_norm: dict[str, str] = {}
    title_norm = normalize_title(title)

    def add_reference(label: str, doi: str | None, *, force: bool) -> None:
        norm = normalize_title(label)
        if not norm or norm == title_norm:
            return
        if norm in by_norm:
            if doi and by_norm[norm] not in reference_dois:
                reference_dois[by_norm[norm]] = doi
            return
        if not force and len(references) >= max_references:
            return
        by_norm[norm] = label
        references.append(label)
        if doi:
            reference_dois[label] = doi

    for ref in record.references:
        add_reference(ref, reference_dois.get(ref), force=True)
    for ext in work.references:
        add_reference(ext.title, ext.doi, force=False)

    return replace(
        record,
        title=title,
        authors=list(work.authors) or list(record.authors),
        year=work.year or record.year,
        venue=work.venue or record.venue,
        doi=work.doi or record.doi,
        affiliations=affiliations,
        references=references,
        reference_dois=reference_dois,
        author_affiliations=author_affiliations,
        author_ids={name: list(ids) for name, ids in work.author_ids.items()},
        institution_ids={name: list(ids) for name, ids in work.institution_ids.items()},
        openalex_id=work.id if work.source == SOURCE_OPENALEX else record.openalex_id,
        metadata_source=work.source,
    )


# --------------------------------------------------------------------------- #
# Resolver
# --------------------------------------------------------------------------- #


class MetadataResolver:
    """Look up works in OpenAlex/Crossref with on-disk caching."""

    def __init__(
        self,
        *,
        http: httpx.AsyncClient | None = None,
        mailto: str | None = None,
        cache_path: Path | None = None,
        max_references: int = 40,
        use_openalex: bool = True,
        use_crossref: bool = True,
        timeout: float = 15.0,
        concurrency: int = 2,
    ) -> None:
        self._own_http = http is None
        self.http = http or httpx.AsyncClient(
            timeout=timeout, headers={"User-Agent": USER_AGENT}, follow_redirects=True
        )
        self.openalex = OpenAlexClient(self.http, mailto) if use_openalex else None
        self.crossref = CrossrefClient(self.http, mailto) if use_crossref else None
        self.max_references = max_references
        self.requests_failed = 0
        self._semaphore = asyncio.Semaphore(concurrency)
        self._cache_path = cache_path
        self._cache: dict[str, dict[str, Any] | None] = self._load_cache()

    @classmethod
    def from_settings(
        cls, settings: Settings, http: httpx.AsyncClient | None = None
    ) -> MetadataResolver:
        return cls(
            http=http,
            mailto=settings.metadata_mailto,
            cache_path=settings.working_dir / CACHE_FILENAME,
            max_references=settings.metadata_max_references,
        )

    async def aclose(self) -> None:
        if self._own_http:
            await self.http.aclose()

    # -- cache -------------------------------------------------------------- #

    def _load_cache(self) -> dict[str, dict[str, Any] | None]:
        if self._cache_path is None or not self._cache_path.exists():
            return {}
        try:
            data = json.loads(self._cache_path.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else {}
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning(
                "Could not read metadata cache %s: %s", self._cache_path, exc
            )
            return {}

    def _save_cache(self) -> None:
        if self._cache_path is None:
            return
        try:
            self._cache_path.parent.mkdir(parents=True, exist_ok=True)
            self._cache_path.write_text(
                json.dumps(self._cache, indent=2, ensure_ascii=False), encoding="utf-8"
            )
        except OSError as exc:
            logger.warning(
                "Could not write metadata cache %s: %s", self._cache_path, exc
            )

    # -- lookup ------------------------------------------------------------- #

    async def lookup(
        self,
        title: str,
        *,
        doi: str | None = None,
        year: int | None = None,
        authors: list[str] | None = None,
    ) -> ExternalWork | None:
        norm_doi = normalize_doi(doi)
        key = f"doi:{norm_doi}" if norm_doi else f"title:{normalize_title(title)}"
        if key in self._cache:
            cached = self._cache[key]
            return ExternalWork.from_dict(cached) if cached else None
        work = await self._lookup_uncached(title, norm_doi, year, authors)
        self._cache[key] = work.to_dict() if work else None
        self._save_cache()
        return work

    async def _lookup_uncached(
        self,
        title: str,
        norm_doi: str | None,
        year: int | None,
        authors: list[str] | None,
    ) -> ExternalWork | None:
        try:
            async with self._semaphore:
                work: ExternalWork | None = None
                if self.openalex is not None:
                    work = await self._openalex_lookup(title, norm_doi, year, authors)
                if work is None and self.crossref is not None:
                    work = await self._crossref_lookup(title, norm_doi, year, authors)
        except (httpx.HTTPError, ValueError, KeyError, TypeError) as exc:
            self.requests_failed += 1
            logger.warning("External metadata lookup failed for %r: %s", title, exc)
            return None
        if work is None:
            logger.info("No external metadata found for %r", title)
        return work

    async def _openalex_lookup(
        self,
        title: str,
        norm_doi: str | None,
        year: int | None,
        authors: list[str] | None,
    ) -> ExternalWork | None:
        assert self.openalex is not None
        work: ExternalWork | None = None
        if norm_doi:
            data = await self.openalex.work_by_doi(norm_doi)
            if data is not None:
                work = parse_openalex_work(data, "doi")
        if work is None and title:
            candidates = [
                parse_openalex_work(item, "title")
                for item in await self.openalex.search(title)
            ]
            work = pick_best_candidate(candidates, title, year, authors)
        if work is None:
            return None
        await self._resolve_openalex_references(work)
        return work

    async def _resolve_openalex_references(self, work: ExternalWork) -> None:
        assert self.openalex is not None
        ids = work.reference_ids[: self.max_references]
        if not ids:
            return
        references: list[ExternalReference] = []
        for item in await self.openalex.works_by_ids(ids):
            ref_title = _clean(item.get("title") or item.get("display_name"))
            if ref_title:
                references.append(
                    ExternalReference(
                        title=ref_title, doi=normalize_doi(item.get("doi"))
                    )
                )
        work.references = references

    async def _crossref_lookup(
        self,
        title: str,
        norm_doi: str | None,
        year: int | None,
        authors: list[str] | None,
    ) -> ExternalWork | None:
        assert self.crossref is not None
        if norm_doi:
            message = await self.crossref.work_by_doi(norm_doi)
            if message is not None:
                return parse_crossref_work(message, "doi")
        if not title:
            return None
        candidates = [
            parse_crossref_work(item, "title")
            for item in await self.crossref.search(title)
        ]
        best = pick_best_candidate(candidates, title, year, authors)
        if best is None:
            return None
        if best.doi:
            full = await self.crossref.work_by_doi(best.doi)
            if full is not None:
                return parse_crossref_work(full, "title")
        return best

    # -- record enrichment -------------------------------------------------- #

    async def enrich(self, record: BibliographicRecord) -> BibliographicRecord:
        work = await self.lookup(
            record.title, doi=record.doi, year=record.year, authors=record.authors
        )
        if work is None:
            return record
        enriched = enrich_record(record, work, max_references=self.max_references)
        logger.info(
            "Enriched %r from %s (matched by %s): authors %d -> %d, "
            "references %d -> %d, doi=%s",
            record.title,
            work.source,
            work.matched_by,
            len(record.authors),
            len(enriched.authors),
            len(record.references),
            len(enriched.references),
            enriched.doi,
        )
        return enriched
