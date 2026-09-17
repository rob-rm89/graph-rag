"""Entity reconciliation and cross-paper citation linking."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from lightrag.kg.shared_storage import finalize_share_data

from config import REL_CITES, Settings, normalize_type
from ingestion import BibliographicRecord, IngestionEngine, record_to_custom_kg
from offline_backend import StubLLM, offline_rag_kwargs
from rag_factory import build_rag
from reconciliation import (
    KIND_AUTHOR,
    KIND_CITEDWORK,
    KIND_PAPER,
    KIND_VENUE,
    EntityRegistry,
    MergePlan,
    Reconciler,
    normalize_doi,
    normalize_title,
    parse_person,
    persons_compatible,
    titles_match,
)

PAPER_A = "Sparse Attention Routing for Long-Document Retrieval-Augmented Generation"
PAPER_A_VARIANT = (
    "Sparse attention routing for long document retrieval augmented generation"
)
VASWANI = "Attention Is All You Need"


# --------------------------------------------------------------------------- #
# Normalisation and name parsing
# --------------------------------------------------------------------------- #


def test_normalize_doi_variants() -> None:
    assert normalize_doi("https://doi.org/10.1000/ABC.123.") == "10.1000/abc.123"
    assert normalize_doi("doi:10.1000/xyz") == "10.1000/xyz"
    assert normalize_doi("Some title. 10.1000/in-text)") == "10.1000/in-text"
    assert normalize_doi("no doi here") is None
    assert normalize_doi(None) is None


def test_normalize_title_and_matching() -> None:
    assert normalize_title("Attention Is All You Need (2017).") == normalize_title(
        VASWANI
    )
    assert titles_match(normalize_title(PAPER_A), normalize_title(PAPER_A_VARIANT))
    assert titles_match(
        normalize_title(VASWANI),
        normalize_title(
            "Attention Is All You Need. Advances in Neural Information Processing"
        ),
    )
    assert not titles_match(
        normalize_title("Graph RAG"), normalize_title("GraphRAG Survey")
    )
    assert not titles_match(
        normalize_title("Deep Residual Learning"),
        normalize_title("Deep Reinforcement Learning"),
    )


def test_parse_person_forms() -> None:
    full = parse_person("Elena Marchetti")
    assert full and full.given == ("Elena",) and full.family == "marchetti"
    assert (
        parse_person("Marchetti, Elena") == full
        or parse_person("Marchetti, Elena").key == full.key
    )
    initials = parse_person("E.M. Marchetti")
    assert initials and initials.given == ("E", "M") and initials.completeness == 0
    particle = parse_person("Ludwig van Beethoven")
    assert (
        particle
        and particle.family == "van beethoven"
        and particle.given == ("Ludwig",)
    )
    assert parse_person("Plato").given == ()
    assert parse_person("") is None


def test_persons_compatible_rules() -> None:
    elena = parse_person("Elena Marchetti")
    e_init = parse_person("E. Marchetti")
    em_init = parse_person("E. M. Marchetti")
    eva = parse_person("Eva Marchetti")
    other = parse_person("Elena Rossi")
    assert persons_compatible(elena, e_init) and persons_compatible(e_init, elena)
    assert persons_compatible(elena, em_init)
    assert not persons_compatible(elena, eva)
    assert not persons_compatible(elena, other)
    assert not persons_compatible(e_init, parse_person("K. Marchetti"))


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #


def test_registry_resolution_and_persistence(tmp_path: Path) -> None:
    path = tmp_path / "registry.json"
    registry = EntityRegistry(path)
    registry.register(
        PAPER_A, KIND_PAPER, doi="10.0000/synthetic.2024.001", doc="doc-a"
    )
    registry.register("Elena Marchetti", KIND_AUTHOR, doc="doc-a")
    registry.register("Jane Smith", KIND_AUTHOR)
    registry.register("John Smith", KIND_AUTHOR)
    registry.register("EMNLP 2024", KIND_VENUE)
    registry.save()

    reloaded = EntityRegistry(path)
    assert len(reloaded) == 5
    assert reloaded.resolve_work(PAPER_A_VARIANT).name == PAPER_A
    assert (
        reloaded.resolve_work("anything", doi="DOI:10.0000/SYNTHETIC.2024.001").name
        == PAPER_A
    )
    assert reloaded.resolve_person("E. Marchetti").name == "Elena Marchetti"
    assert reloaded.resolve_person("Marchetti, Elena").name == "Elena Marchetti"
    assert reloaded.resolve_person("J. Smith") is None  # ambiguous: Jane vs John
    assert reloaded.resolve_person("Jane Smith").name == "Jane Smith"
    assert reloaded.resolve_named(KIND_VENUE, "emnlp  2024").name == "EMNLP 2024"
    assert reloaded.resolve_work("Completely Different Title") is None


# --------------------------------------------------------------------------- #
# Record-level reconciliation
# --------------------------------------------------------------------------- #


def test_reconcile_links_reference_to_ingested_paper_and_resolves_initials() -> None:
    registry = EntityRegistry()
    reconciler = Reconciler(registry)
    paper_a = BibliographicRecord(
        title=PAPER_A,
        authors=["Elena Marchetti", "Kwame Mensah"],
        year=2024,
        venue="EMNLP 2024",
        references=[VASWANI],
    )
    first = reconciler.reconcile(paper_a, "doc-a")
    assert first.record == paper_a
    assert first.new_cited_works == [VASWANI]
    assert not first.merges and not first.linked_papers

    paper_b = BibliographicRecord(
        title="Routing Ablations",
        authors=["E. Marchetti", "Sofia Lindqvist"],
        year=2025,
        venue="emnlp 2024",
        references=[
            PAPER_A_VARIANT,
            "Attention is all you need (2017)",
            "Brand New Work",
        ],
    )
    second = reconciler.reconcile(paper_b, "doc-b")
    assert second.record.authors == ["Elena Marchetti", "Sofia Lindqvist"]
    assert second.record.venue == "EMNLP 2024"
    assert second.record.references == [PAPER_A, VASWANI, "Brand New Work"]
    assert second.linked_papers == [PAPER_A]
    assert second.new_cited_works == ["Brand New Work"]
    assert {PAPER_A, VASWANI, "Elena Marchetti", "EMNLP 2024"} <= second.existing
    assert "Sofia Lindqvist" not in second.existing
    assert not second.merges
    assert registry.get(PAPER_A).aliases == [PAPER_A_VARIANT]


def test_reconcile_upgrades_cited_work_and_completes_author_name() -> None:
    registry = EntityRegistry()
    reconciler = Reconciler(registry)
    citing = BibliographicRecord(
        title="A Citing Paper", authors=["E. Marchetti"], references=[VASWANI]
    )
    reconciler.reconcile(citing, "doc-1")
    assert registry.get(VASWANI).kind == KIND_CITEDWORK

    cited = BibliographicRecord(
        title="Attention is all you need.",
        authors=["Elena Marchetti"],
        references=["A citing paper"],
    )
    result = reconciler.reconcile(cited, "doc-2")
    assert result.record.title == "Attention is all you need."
    assert result.record.authors == ["Elena Marchetti"]
    assert result.record.references == ["A Citing Paper"]
    assert result.linked_papers == ["A Citing Paper"]
    assert set(result.merges) == {
        MergePlan(
            (VASWANI,),
            "Attention is all you need.",
            "cited work is now an ingested paper",
        ),
        MergePlan(("E. Marchetti",), "Elena Marchetti", "author name completed"),
    }
    assert registry.get("Attention is all you need.").kind == KIND_PAPER
    assert (
        VASWANI not in registry
        and VASWANI in registry.get("Attention is all you need.").aliases
    )
    assert registry.get("Elena Marchetti").aliases == ["E. Marchetti"]
    assert "E. Marchetti" not in registry


def test_record_to_custom_kg_skips_existing_entities() -> None:
    record = BibliographicRecord(
        title="Paper B", authors=["Elena Marchetti"], references=[PAPER_A, "New Ref"]
    )
    kg = record_to_custom_kg(
        record,
        source_alias="alias",
        file_path="b.md",
        existing_entities=frozenset({PAPER_A, "Elena Marchetti"}),
    )
    names = {e["entity_name"] for e in kg["entities"]}
    assert names == {"Paper B", "New Ref"}  # existing nodes are not overwritten
    pairs = {(r["src_id"], r["tgt_id"], r["keywords"]) for r in kg["relationships"]}
    assert ("Paper B", PAPER_A, REL_CITES) in pairs
    assert ("Paper B", "Elena Marchetti", "authored_by") in pairs


# --------------------------------------------------------------------------- #
# Graph-wide planning
# --------------------------------------------------------------------------- #


def test_plan_graph_merges_clusters_duplicates() -> None:
    registry = EntityRegistry()
    registry.register(PAPER_A, KIND_PAPER)
    registry.register("Elena Marchetti", KIND_AUTHOR)
    nodes = [
        {"id": PAPER_A, "entity_type": "paper"},
        {"id": PAPER_A_VARIANT, "entity_type": "citedwork"},
        {"id": VASWANI, "entity_type": "citedwork"},
        {"id": "Attention is all you need (2017)", "entity_type": "citedwork"},
        {"id": "Elena Marchetti", "entity_type": "author"},
        {"id": "E. Marchetti", "entity_type": "author"},
        {"id": "Kwame Mensah", "entity_type": "author"},
        {"id": "Jane Smith", "entity_type": "author"},
        {"id": "John Smith", "entity_type": "author"},
        {"id": "J. Smith", "entity_type": "author"},
        {"id": "EMNLP 2024", "entity_type": "venue"},
        {"id": "emnlp 2024", "entity_type": "venue"},
        {"id": "Transformer", "entity_type": "method"},
    ]
    plans = Reconciler(registry).plan_graph_merges(nodes)
    by_target = {plan.target: plan for plan in plans}
    assert by_target[PAPER_A].sources == (PAPER_A_VARIANT,)
    assert by_target["Attention is all you need (2017)"].sources == (VASWANI,)
    assert by_target["Elena Marchetti"].sources == ("E. Marchetti",)
    assert by_target["EMNLP 2024"].sources == ("emnlp 2024",)
    merged = {s for plan in plans for s in plan.sources}
    assert "J. Smith" not in merged  # ambiguous between Jane and John
    assert "Kwame Mensah" not in merged and "Transformer" not in merged


# --------------------------------------------------------------------------- #
# Offline end-to-end: two documents through real LightRAG
# --------------------------------------------------------------------------- #

DOC_A = f"""Title: {PAPER_A}
Authors: Elena Marchetti, Kwame Mensah
Year: 2024
Venue: EMNLP 2024
DOI: 10.0000/synthetic.2024.001
Affiliations: Politecnico di Torino
References: {VASWANI}

Sparse Attention Routing was evaluated on NarrativeQA and LongBench.
"""

DOC_B = f"""Title: Routing Ablations for Retrieval-Augmented Generation
Authors: E. Marchetti, Sofia Lindqvist
Year: 2025
Venue: emnlp 2024
References: {PAPER_A_VARIANT}; Attention is all you need (2017)

This note by E. Marchetti revisits Sparse Attention Routing and Dual-Level Retrieval.
"""


def test_two_documents_are_linked_and_deduplicated(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "a_paper.md").write_text(DOC_A, encoding="utf-8")
    (data_dir / "b_paper.md").write_text(DOC_B, encoding="utf-8")
    settings = Settings().with_overrides(
        working_dir=tmp_path / "rag",
        data_dir=data_dir,
        canvas_path=tmp_path / "c.canvas",
    )

    async def run() -> tuple[list, list[dict], list[dict]]:
        stub = StubLLM()
        rag = await build_rag(settings, **offline_rag_kwargs(stub))
        try:
            engine = IngestionEngine(rag, settings, stub)
            report = await engine.ingest_all()
            storage = rag.chunk_entity_relation_graph
            return (
                report.merges,
                await storage.get_all_nodes(),
                await storage.get_all_edges(),
            )
        finally:
            await rag.finalize_storages()
            finalize_share_data()

    merges, nodes, edges = asyncio.run(run())
    names = {node["id"]: normalize_type(node.get("entity_type")) for node in nodes}

    # Cross-paper citation: B cites A's paper node, no CitedWork twin for A.
    assert names[PAPER_A] == KIND_PAPER
    assert PAPER_A_VARIANT not in names
    cites = {
        (e["source"], e["target"])
        for e in edges
        if REL_CITES in str(e.get("keywords", ""))
    }
    assert any(
        PAPER_A in pair and "Routing Ablations" in "".join(pair) for pair in cites
    )

    # Author reconciliation: profiler initials resolved, extraction twin merged.
    assert "Elena Marchetti" in names and "E. Marchetti" not in names
    assert any(plan.target == "Elena Marchetti" for plan in merges)

    # Shared cited work collapsed to one node with two citing papers.
    vaswani_nodes = [n for n in names if normalize_title(n) == normalize_title(VASWANI)]
    assert len(vaswani_nodes) == 1
    citing = {src for src, tgt in cites if tgt == vaswani_nodes[0]} | {
        tgt for src, tgt in cites if src == vaswani_nodes[0]
    }
    assert len(citing) == 2

    # Venue collapsed on normalised name.
    assert sum(1 for n, t in names.items() if t == KIND_VENUE) == 1

    registry = json.loads(
        (settings.working_dir / "litgraph_registry.json").read_text("utf-8")
    )
    kinds = {entry["name"]: entry["kind"] for entry in registry["entries"]}
    assert kinds[PAPER_A] == KIND_PAPER and kinds["Elena Marchetti"] == KIND_AUTHOR
