"""External identifiers (OpenAlex IDs, ORCID, ROR) in reconciliation."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import httpx
from lightrag.kg.shared_storage import finalize_share_data

from config import Settings
from ingestion import BibliographicRecord, IngestionEngine, record_to_custom_kg
from metadata_lookup import (
    MetadataResolver,
    enrich_record,
    parse_crossref_work,
    parse_openalex_work,
)
from offline_backend import StubLLM, offline_rag_kwargs
from rag_factory import build_rag
from reconciliation import (
    KIND_AUTHOR,
    KIND_CITEDWORK,
    KIND_ORGANIZATION,
    KIND_PAPER,
    EntityRegistry,
    Reconciler,
)

STANFORD_ROR = "https://ror.org/00f54p054"
SMITH_ORCID = "https://orcid.org/0000-0002-1825-0097"


def authorship(
    name: str, author_id: str, institutions: tuple = (), orcid: str | None = None
) -> dict[str, Any]:
    return {
        "author": {
            "id": f"https://openalex.org/{author_id}",
            "display_name": name,
            "orcid": orcid,
        },
        "institutions": [
            {"id": f"https://openalex.org/{iid}", "display_name": iname, "ror": ror}
            for iid, iname, ror in institutions
        ],
    }


def openalex_work(
    wid: str, doi: str, title: str, year: int, authorships: list[dict[str, Any]]
) -> dict[str, Any]:
    return {
        "id": f"https://openalex.org/{wid}",
        "doi": f"https://doi.org/{doi}",
        "title": title,
        "display_name": title,
        "publication_year": year,
        "authorships": authorships,
        "primary_location": {"source": {"display_name": "Journal of Tests"}},
        "referenced_works": [],
    }


WORK_A = openalex_work(
    "W1",
    "10.5555/a",
    "Graph Methods for Citation Analysis",
    2020,
    [
        authorship(
            "John Smith",
            "A1",
            (("I1", "Stanford University", STANFORD_ROR),),
            SMITH_ORCID,
        ),
        authorship("Mary Jones", "A3", (("I1", "Stanford University", STANFORD_ROR),)),
    ],
)
# Same person A1 spelled with initials, plus a *different* John Smith (A2).
WORK_B = openalex_work(
    "W2",
    "10.5555/b",
    "Sparse Graph Routing",
    2021,
    [
        authorship("J. Smith", "A1", (("I1", "Stanford", None),)),
        authorship("John Smith", "A2", (("I2", "Other University", None),)),
    ],
)


# --------------------------------------------------------------------------- #
# Parsing
# --------------------------------------------------------------------------- #


def test_parse_openalex_captures_identifiers() -> None:
    work = parse_openalex_work(WORK_A, "doi")
    assert work.author_ids == {
        "John Smith": ["openalex:A1", "orcid:0000-0002-1825-0097"],
        "Mary Jones": ["openalex:A3"],
    }
    assert work.institution_ids == {
        "Stanford University": ["openalex:I1", "ror:00f54p054"]
    }


def test_parse_crossref_captures_orcid() -> None:
    message = {
        "DOI": "10.5555/c",
        "title": ["Notes"],
        "author": [
            {
                "given": "Ada",
                "family": "Lovelace",
                "ORCID": "http://orcid.org/0000-0001-2345-6789",
            },
            {"given": "Charles", "family": "Babbage"},
        ],
        "issued": {"date-parts": [[1843]]},
    }
    work = parse_crossref_work(message, "doi")
    assert work.author_ids == {"Ada Lovelace": ["orcid:0000-0001-2345-6789"]}
    assert work.institution_ids == {}


def test_enrich_record_carries_identifiers() -> None:
    record = BibliographicRecord(title="Graph Methods for Citation Analysis")
    enriched = enrich_record(
        record, parse_openalex_work(WORK_A, "doi"), max_references=10
    )
    assert enriched.author_ids["John Smith"] == [
        "openalex:A1",
        "orcid:0000-0002-1825-0097",
    ]
    assert enriched.institution_ids["Stanford University"] == [
        "openalex:I1",
        "ror:00f54p054",
    ]
    kg = record_to_custom_kg(enriched, source_alias="a", file_path="a.md")
    by_name = {e["entity_name"]: e for e in kg["entities"]}
    assert "openalex:A1" in by_name["John Smith"]["description"]
    assert "ror:00f54p054" in by_name["Stanford University"]["description"]


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #


def test_registry_identity_resolution(tmp_path: Path) -> None:
    registry = EntityRegistry(tmp_path / "registry.json")
    registry.register("John Smith", KIND_AUTHOR, ids=["openalex:A1"])
    registry.register("Jane Smith", KIND_AUTHOR, ids=["openalex:A9"])
    registry.register(
        "Google (United States)", KIND_ORGANIZATION, ids=["ror:00njsd438"]
    )
    registry.save()
    registry = EntityRegistry(tmp_path / "registry.json")  # identifiers persist

    assert registry.resolve_person("J. Smith") is None  # ambiguous by name alone
    assert registry.resolve_person("J. Smith", ids=["openalex:A1"]).name == "John Smith"
    assert (
        registry.resolve_person("John Smith", ids=["openalex:A2"]) is None
    )  # other person
    assert (
        registry.resolve_person("John Smith").name == "John Smith"
    )  # no evidence -> name
    assert (
        registry.resolve_person("Jonathan Smith", ids=["openalex:A1"]).name
        == "John Smith"
    )
    assert registry.resolve_named(
        KIND_ORGANIZATION, "Google", ids=["ror:00njsd438"]
    ).name == ("Google (United States)")
    assert registry.resolve_named(KIND_ORGANIZATION, "Google") is None
    assert registry.get("John Smith").ids == ["openalex:A1"]


# --------------------------------------------------------------------------- #
# Record-level reconciliation
# --------------------------------------------------------------------------- #


def test_reconcile_merges_by_id_and_splits_homonyms() -> None:
    registry = EntityRegistry()
    reconciler = Reconciler(registry)
    record_a = enrich_record(
        BibliographicRecord(title="Graph Methods for Citation Analysis"),
        parse_openalex_work(WORK_A, "doi"),
        max_references=10,
    )
    reconciler.reconcile(record_a, "doc-a")
    assert registry.get("John Smith").ids == [
        "openalex:A1",
        "orcid:0000-0002-1825-0097",
    ]

    record_b = enrich_record(
        BibliographicRecord(title="Sparse Graph Routing"),
        parse_openalex_work(WORK_B, "doi"),
        max_references=10,
    )
    result = reconciler.reconcile(record_b, "doc-b")
    # "J. Smith" carries A1 -> the existing John Smith; the second John Smith
    # carries A2 -> a distinct, disambiguated node.
    assert result.record.authors == ["John Smith", "John Smith (Other University)"]
    assert "John Smith" in result.existing
    assert result.record.author_affiliations == {
        "John Smith": [
            "Stanford University"
        ],  # "Stanford" resolved via institution id I1
        "John Smith (Other University)": ["Other University"],
    }
    assert result.record.author_ids["John Smith (Other University)"] == ["openalex:A2"]
    assert registry.get("John Smith (Other University)").ids == ["openalex:A2"]
    assert "J. Smith" in registry.get("John Smith").aliases
    assert not result.merges


def test_reconcile_completes_name_through_id() -> None:
    registry = EntityRegistry()
    registry.register("J. Smith", KIND_AUTHOR, ids=["openalex:A1"])
    reconciler = Reconciler(registry)
    record = BibliographicRecord(
        title="Later Paper",
        authors=["John Smith"],
        author_ids={"John Smith": ["openalex:A1"]},
    )
    result = reconciler.reconcile(record, "doc-x")
    assert result.record.authors == ["John Smith"]
    assert [plan.sources for plan in result.merges] == [("J. Smith",)]
    assert registry.get("John Smith").ids == ["openalex:A1"]
    assert "J. Smith" not in registry


# --------------------------------------------------------------------------- #
# Graph-wide planning respects identities
# --------------------------------------------------------------------------- #


def test_plan_graph_merges_respects_identifiers() -> None:
    registry = EntityRegistry()
    registry.register("John Smith", KIND_AUTHOR, ids=["openalex:A1"])
    registry.register("John Smith (Other University)", KIND_AUTHOR, ids=["openalex:A2"])
    registry.register("J. Smith", KIND_AUTHOR, ids=["openalex:A1"])  # stray twin of A1
    registry.register("Jon Smith", KIND_AUTHOR, ids=["openalex:A2"])  # stray twin of A2
    registry.register("Deep Learning", KIND_PAPER, doi="10.1000/x")
    registry.register("Deep Learning.", KIND_CITEDWORK, doi="10.1000/y")
    nodes = [
        {"id": "John Smith", "entity_type": "author"},
        {"id": "John Smith (Other University)", "entity_type": "author"},
        {"id": "J. Smith", "entity_type": "author"},
        {"id": "Jon Smith", "entity_type": "author"},
        {"id": "Deep Learning", "entity_type": "paper"},
        {"id": "Deep Learning.", "entity_type": "citedwork"},
        {"id": "Deep learning", "entity_type": "citedwork"},  # no DOI: may merge
    ]
    plans = Reconciler(registry).plan_graph_merges(nodes)
    by_target = {plan.target: set(plan.sources) for plan in plans}
    assert by_target["John Smith"] == {"J. Smith"}
    assert by_target["John Smith (Other University)"] == {"Jon Smith"}
    assert by_target["Deep Learning"] == {"Deep learning"}  # distinct DOI kept apart
    assert "Deep Learning." not in {s for plan in plans for s in plan.sources}


# --------------------------------------------------------------------------- #
# Offline end-to-end with two catalogue-enriched documents
# --------------------------------------------------------------------------- #


class FakeOpenAlex:
    def __init__(self, works_by_doi: dict[str, dict[str, Any]]) -> None:
        self.works = works_by_doi

    def handler(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if request.url.host == "api.openalex.org" and path.startswith("/works/doi:"):
            work = self.works.get(path.removeprefix("/works/doi:"))
            return httpx.Response(200, json=work) if work else httpx.Response(404)
        if request.url.host == "api.openalex.org" and path == "/works":
            return httpx.Response(200, json={"results": []})
        return httpx.Response(404)


DOC_A = """Title: Graph Methods for Citation Analysis
Authors: J Smith
DOI: 10.5555/a

Graph Methods for Citation Analysis studies Knowledge Graph structure.
"""
DOC_B = """Title: Sparse Graph Routing
Authors: Smith
DOI: 10.5555/b

Sparse Graph Routing builds on Dual-Level Retrieval.
"""


def test_identifiers_flow_through_pipeline(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "a.md").write_text(DOC_A, encoding="utf-8")
    (data_dir / "b.md").write_text(DOC_B, encoding="utf-8")
    settings = Settings().with_overrides(
        working_dir=tmp_path / "rag",
        data_dir=data_dir,
        canvas_path=tmp_path / "c.canvas",
    )
    api = FakeOpenAlex({"10.5555/a": WORK_A, "10.5555/b": WORK_B})

    async def run() -> list[dict[str, Any]]:
        stub = StubLLM()
        resolver = MetadataResolver(
            http=httpx.AsyncClient(transport=httpx.MockTransport(api.handler)),
            use_crossref=False,
        )
        rag = await build_rag(settings, **offline_rag_kwargs(stub))
        try:
            engine = IngestionEngine(rag, settings, stub, metadata_resolver=resolver)
            await engine.ingest_all()
            return await rag.chunk_entity_relation_graph.get_all_nodes()
        finally:
            await resolver.aclose()
            await rag.finalize_storages()
            finalize_share_data()

    nodes = asyncio.run(run())
    names = {node["id"] for node in nodes}
    assert {"John Smith", "John Smith (Other University)", "Mary Jones"} <= names
    assert "J. Smith" not in names and "Jon Smith" not in names
    assert "Stanford University" in names and "Stanford" not in names
    registry = json.loads(
        (settings.working_dir / "litgraph_registry.json").read_text("utf-8")
    )
    ids = {entry["name"]: entry["ids"] for entry in registry["entries"]}
    assert ids["John Smith"] == ["openalex:A1", "orcid:0000-0002-1825-0097"]
    assert ids["John Smith (Other University)"] == ["openalex:A2"]
    assert ids["Stanford University"] == ["openalex:I1", "ror:00f54p054"]
