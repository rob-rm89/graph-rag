"""OpenAlex/Crossref metadata lookup, fully offline via httpx.MockTransport."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import httpx
from lightrag.kg.shared_storage import finalize_share_data

from config import REL_AFFILIATED_WITH, Settings
from ingestion import BibliographicRecord, IngestionEngine
from metadata_lookup import (
    CACHE_FILENAME,
    ExternalReference,
    MetadataResolver,
    enrich_record,
    parse_crossref_work,
    parse_openalex_work,
    pick_best_candidate,
)
from offline_backend import StubLLM, offline_rag_kwargs
from rag_factory import build_rag

ATTENTION = "Attention Is All You Need"
ATTENTION_DOI = "10.5555/attention"

OPENALEX_WORK: dict[str, Any] = {
    "id": "https://openalex.org/W2963403868",
    "doi": f"https://doi.org/{ATTENTION_DOI}",
    "title": ATTENTION,
    "display_name": ATTENTION,
    "publication_year": 2017,
    "authorships": [
        {
            "author": {"display_name": "Ashish Vaswani"},
            "institutions": [{"display_name": "Google (United States)"}],
            "raw_author_name": "Ashish Vaswani",
        },
        {
            "author": {"display_name": "Noam Shazeer"},
            "institutions": [{"display_name": "Google (United States)"}],
        },
        {"author": {"display_name": "Illia Polosukhin"}, "institutions": []},
    ],
    "primary_location": {
        "source": {"display_name": "Neural Information Processing Systems"}
    },
    "referenced_works": ["https://openalex.org/W1", "https://openalex.org/W2"],
}
OPENALEX_DECOY: dict[str, Any] = {
    "id": "https://openalex.org/W999",
    "doi": "https://doi.org/10.65215/decoy",
    "title": ATTENTION,
    "publication_year": 2025,
    "authorships": [{"author": {"display_name": "Someone Else"}, "institutions": []}],
    "primary_location": None,
    "referenced_works": [],
}
OPENALEX_REFERENCES: dict[str, Any] = {
    "results": [
        {
            "id": "https://openalex.org/W1",
            "title": (
                "Neural Machine Translation by Jointly Learning to Align and Translate"
            ),
            "doi": "https://doi.org/10.5555/nmt",
            "publication_year": 2015,
        },
        {
            "id": "https://openalex.org/W2",
            "title": "Long Short-Term Memory",
            "doi": None,
            "publication_year": 1997,
        },
    ]
}
CROSSREF_MESSAGE: dict[str, Any] = {
    "DOI": ATTENTION_DOI,
    "title": [ATTENTION],
    "author": [
        {
            "given": "Ashish",
            "family": "Vaswani",
            "affiliation": [{"name": "Google Brain"}],
        },
        {"given": "Noam", "family": "Shazeer", "affiliation": []},
    ],
    "issued": {"date-parts": [[2017, 6, 12]]},
    "container-title": ["Advances in Neural Information Processing Systems"],
    "reference": [
        {
            "key": "r1",
            "DOI": "10.5555/nmt",
            "article-title": (
                "Neural Machine Translation by Jointly Learning to Align and Translate"
            ),
        },
        {
            "key": "r2",
            "unstructured": "Hochreiter and Schmidhuber. Long Short-Term Memory.",
        },
        {"key": "r3", "DOI": "10.5555/doi-only"},
    ],
}


class FakeCatalogues:
    """MockTransport handler emulating both APIs; records every request URL."""

    def __init__(
        self,
        *,
        openalex_doi: dict[str, Any] | None = None,
        openalex_search: list[dict[str, Any]] | None = None,
        openalex_refs: dict[str, Any] | None = None,
        crossref_doi: dict[str, Any] | None = None,
        crossref_search: list[dict[str, Any]] | None = None,
        error: bool = False,
    ) -> None:
        self.openalex_doi = openalex_doi
        self.openalex_search = openalex_search or []
        self.openalex_refs = openalex_refs or {"results": []}
        self.crossref_doi = crossref_doi
        self.crossref_search = crossref_search or []
        self.error = error
        self.requests: list[str] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(str(request.url))
        if self.error:
            raise httpx.ConnectError("network down", request=request)
        host, path = request.url.host, request.url.path
        if host == "api.openalex.org":
            if path.startswith("/works/doi:"):
                if self.openalex_doi is None:
                    return httpx.Response(404, json={"error": "not found"})
                return httpx.Response(200, json=self.openalex_doi)
            if path == "/works" and "filter" in request.url.params:
                return httpx.Response(200, json=self.openalex_refs)
            if path == "/works":
                return httpx.Response(200, json={"results": self.openalex_search})
        if host == "api.crossref.org":
            if path.startswith("/works/"):
                if self.crossref_doi is None:
                    return httpx.Response(404, text="Resource not found.")
                return httpx.Response(200, json={"message": self.crossref_doi})
            if path == "/works":
                return httpx.Response(
                    200, json={"message": {"items": self.crossref_search}}
                )
        return httpx.Response(404)

    def client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(transport=httpx.MockTransport(self.handler))


def resolver_for(
    api: FakeCatalogues, cache_dir: Path | None = None, **kw: Any
) -> MetadataResolver:
    cache_path = cache_dir / CACHE_FILENAME if cache_dir else None
    return MetadataResolver(http=api.client(), cache_path=cache_path, **kw)


# --------------------------------------------------------------------------- #
# Parsers and pure merge logic
# --------------------------------------------------------------------------- #


def test_parse_openalex_work() -> None:
    work = parse_openalex_work(OPENALEX_WORK, "doi")
    assert work.id == "W2963403868" and work.doi == ATTENTION_DOI
    assert work.year == 2017 and work.venue == "Neural Information Processing Systems"
    assert work.authors == ["Ashish Vaswani", "Noam Shazeer", "Illia Polosukhin"]
    assert work.author_affiliations == {
        "Ashish Vaswani": ["Google (United States)"],
        "Noam Shazeer": ["Google (United States)"],
    }
    assert work.reference_ids == ["W1", "W2"] and work.references == []


def test_parse_crossref_work() -> None:
    work = parse_crossref_work(CROSSREF_MESSAGE, "title")
    assert work.source == "crossref" and work.doi == ATTENTION_DOI
    assert work.authors == ["Ashish Vaswani", "Noam Shazeer"]
    assert work.author_affiliations == {"Ashish Vaswani": ["Google Brain"]}
    assert work.year == 2017
    assert work.venue == "Advances in Neural Information Processing Systems"
    assert [ref.title for ref in work.references] == [
        "Neural Machine Translation by Jointly Learning to Align and Translate",
        "Hochreiter and Schmidhuber. Long Short-Term Memory.",
        "10.5555/doi-only",
    ]
    assert (
        work.references[0].doi == "10.5555/nmt"
        and work.references[2].doi == "10.5555/doi-only"
    )


def test_pick_best_candidate_uses_year_and_authors() -> None:
    decoy = parse_openalex_work(OPENALEX_DECOY, "title")
    real = parse_openalex_work(OPENALEX_WORK, "title")
    assert pick_best_candidate([decoy, real], ATTENTION, 2017, ["A. Vaswani"]) is real
    assert (
        pick_best_candidate([decoy, real], ATTENTION, 2025, ["Someone Else"]) is decoy
    )
    assert pick_best_candidate([decoy, real], ATTENTION, 2017, ["Nobody Known"]) is None
    assert pick_best_candidate([real], "A Completely Different Title") is None
    # Without year/author hints the best title match wins (first exact title).
    assert pick_best_candidate([decoy, real], ATTENTION) is decoy
    # A mislabelled catalogue year is tolerated when the authors vouch for it
    # (OpenAlex lists the 2017 Transformer paper as 2025), but not on its own.
    mislabelled = parse_openalex_work(
        {**OPENALEX_WORK, "publication_year": 2025}, "title"
    )
    assert (
        pick_best_candidate([mislabelled], ATTENTION, 2017, ["Vaswani, A."])
        is mislabelled
    )
    assert pick_best_candidate([mislabelled], ATTENTION, 2017) is None


def test_enrich_record_merge_policy() -> None:
    record = BibliographicRecord(
        title="Attention is all you need",
        authors=["A. Vaswani"],
        affiliations=["Google"],
        references=["Long short-term memory", "Some In-Text Reference"],
    )
    work = parse_openalex_work(OPENALEX_WORK, "doi")
    work.references = [
        ExternalReference(
            "Neural Machine Translation by Jointly Learning to Align and Translate",
            "10.5555/nmt",
        ),
        ExternalReference("Long Short-Term Memory", "10.5555/lstm"),
        ExternalReference("One Too Many", None),
    ]
    enriched = enrich_record(record, work, max_references=3)
    assert enriched.title == ATTENTION  # DOI match: catalogue title wins
    assert enriched.authors == ["Ashish Vaswani", "Noam Shazeer", "Illia Polosukhin"]
    assert enriched.year == 2017 and enriched.doi == ATTENTION_DOI
    assert enriched.venue == "Neural Information Processing Systems"
    assert enriched.affiliations == ["Google", "Google (United States)"]
    assert enriched.author_affiliations["Ashish Vaswani"] == ["Google (United States)"]
    # In-text references first and kept in the document's spelling, then catalogue
    # references up to the cap; duplicates collapse and inherit the DOI.
    assert enriched.references == [
        "Long short-term memory",
        "Some In-Text Reference",
        "Neural Machine Translation by Jointly Learning to Align and Translate",
    ]
    nmt = "Neural Machine Translation by Jointly Learning to Align and Translate"
    assert enriched.reference_dois == {
        "Long short-term memory": "10.5555/lstm",
        nmt: "10.5555/nmt",
    }
    assert (
        enriched.openalex_id == "W2963403868" and enriched.metadata_source == "openalex"
    )

    fuzzy = ExternalWork_title_only()
    kept = enrich_record(record, fuzzy, max_references=3)
    assert kept.title == record.title  # weak title match keeps the profiled title


def ExternalWork_title_only():  # noqa: N802 - tiny test helper
    work = parse_openalex_work(OPENALEX_WORK, "title")
    work.title = "Attention is all you need: an extended technical report"
    return work


# --------------------------------------------------------------------------- #
# Resolver behaviour
# --------------------------------------------------------------------------- #


def test_lookup_by_doi_resolves_references_through_openalex(tmp_path: Path) -> None:
    api = FakeCatalogues(openalex_doi=OPENALEX_WORK, openalex_refs=OPENALEX_REFERENCES)
    resolver = resolver_for(api, tmp_path)
    work = asyncio.run(
        resolver.lookup("anything", doi=f"https://doi.org/{ATTENTION_DOI}")
    )
    assert work is not None and work.source == "openalex" and work.matched_by == "doi"
    assert [ref.doi for ref in work.references] == ["10.5555/nmt", None]
    assert any("/works/doi:10.5555/attention" in url for url in api.requests)
    assert any("filter=ids.openalex" in url and "W1" in url for url in api.requests)
    assert not any("crossref" in url for url in api.requests)


def test_title_search_scores_candidates() -> None:
    api = FakeCatalogues(openalex_search=[OPENALEX_DECOY, OPENALEX_WORK])
    resolver = resolver_for(api)
    work = asyncio.run(
        resolver.lookup(ATTENTION, year=2017, authors=["Ashish Vaswani"])
    )
    assert work is not None and work.id == "W2963403868" and work.matched_by == "title"


def test_falls_back_to_crossref() -> None:
    api = FakeCatalogues(
        crossref_search=[
            {k: v for k, v in CROSSREF_MESSAGE.items() if k != "reference"}
        ],
        crossref_doi=CROSSREF_MESSAGE,
    )
    resolver = resolver_for(api)
    work = asyncio.run(
        resolver.lookup(ATTENTION, year=2017, authors=["Vaswani, Ashish"])
    )
    assert work is not None and work.source == "crossref"
    assert len(work.references) == 3  # full record fetched by DOI after the search
    assert any("api.openalex.org" in url for url in api.requests)
    assert any("query.bibliographic" in url for url in api.requests)


def test_network_errors_degrade_gracefully() -> None:
    api = FakeCatalogues(error=True)
    resolver = resolver_for(api)
    record = BibliographicRecord(title=ATTENTION, authors=["A. Vaswani"])
    assert asyncio.run(resolver.lookup(ATTENTION)) is None
    assert asyncio.run(resolver.enrich(record)) == record
    assert resolver.requests_failed >= 1


def test_cache_short_circuits_repeat_lookups(tmp_path: Path) -> None:
    api = FakeCatalogues(openalex_doi=OPENALEX_WORK, openalex_refs=OPENALEX_REFERENCES)
    first = asyncio.run(
        resolver_for(api, tmp_path).lookup(ATTENTION, doi=ATTENTION_DOI)
    )
    assert first is not None and api.requests
    miss = asyncio.run(resolver_for(api, tmp_path).lookup("Unknown Title Nobody Wrote"))
    assert miss is None

    api2 = FakeCatalogues(error=True)  # any request would now blow up
    resolver = resolver_for(api2, tmp_path)
    again = asyncio.run(resolver.lookup(ATTENTION, doi=ATTENTION_DOI))
    assert again == first
    assert asyncio.run(resolver.lookup("Unknown Title Nobody Wrote")) is None
    assert api2.requests == []
    cached = json.loads((tmp_path / CACHE_FILENAME).read_text("utf-8"))
    assert (
        f"doi:{ATTENTION_DOI}" in cached
        and cached["title:unknown title nobody wrote"] is None
    )


def test_from_settings_uses_working_dir(tmp_path: Path) -> None:
    settings = Settings().with_overrides(working_dir=tmp_path)
    resolver = MetadataResolver.from_settings(settings, http=FakeCatalogues().client())
    assert resolver._cache_path == tmp_path / CACHE_FILENAME
    assert resolver.max_references == settings.metadata_max_references


# --------------------------------------------------------------------------- #
# Offline end-to-end: enrichment feeds the graph
# --------------------------------------------------------------------------- #

DOC = f"""Title: {ATTENTION}
Authors: A. Vaswani
Year: 2017
DOI: {ATTENTION_DOI}
References: Long Short-Term Memory

The Transformer relies on Dual-Level Retrieval of nothing in particular.
"""


def test_enrichment_flows_into_graph(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "attention.md").write_text(DOC, encoding="utf-8")
    settings = Settings().with_overrides(
        working_dir=tmp_path / "rag",
        data_dir=data_dir,
        canvas_path=tmp_path / "c.canvas",
    )
    api = FakeCatalogues(openalex_doi=OPENALEX_WORK, openalex_refs=OPENALEX_REFERENCES)

    async def run() -> tuple[list[dict], list[dict], BibliographicRecord]:
        stub = StubLLM()
        resolver = resolver_for(api, settings.working_dir)
        rag = await build_rag(settings, **offline_rag_kwargs(stub))
        try:
            engine = IngestionEngine(rag, settings, stub, metadata_resolver=resolver)
            report = await engine.ingest_all()
            storage = rag.chunk_entity_relation_graph
            record = next(iter(report.profiles.values()))
            return await storage.get_all_nodes(), await storage.get_all_edges(), record
        finally:
            await resolver.aclose()
            await rag.finalize_storages()
            finalize_share_data()

    nodes, edges, record = asyncio.run(run())
    names = {node["id"] for node in nodes}
    assert record.metadata_source == "openalex" and record.doi == ATTENTION_DOI
    assert {"Ashish Vaswani", "Noam Shazeer", "Illia Polosukhin"} <= names
    assert "A. Vaswani" not in names  # catalogue names replaced the profiler's initials
    affiliated = {
        (e["source"], e["target"])
        for e in edges
        if REL_AFFILIATED_WITH in str(e.get("keywords", ""))
    }
    assert any(
        "Ashish Vaswani" in pair and "Google (United States)" in pair
        for pair in affiliated
    )
    assert not any(
        "Illia Polosukhin" in pair for pair in affiliated
    )  # no institution known
    assert (
        "Neural Machine Translation by Jointly Learning to Align and Translate" in names
    )
    registry = json.loads(
        (settings.working_dir / "litgraph_registry.json").read_text("utf-8")
    )
    dois = {entry["name"]: entry["doi"] for entry in registry["entries"]}
    assert dois[ATTENTION] == ATTENTION_DOI
    assert (
        dois["Neural Machine Translation by Jointly Learning to Align and Translate"]
        == "10.5555/nmt"
    )
