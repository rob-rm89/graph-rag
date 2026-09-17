"""Offline integration test: the real LightRAG pipeline driven by deterministic stubs.

Exercises, without any network access:
ingestion (profiling -> custom KG -> LightRAG extraction) -> hybrid and mix
queries -> JSON Canvas export -> integrity verification, plus the CLI.
"""

from __future__ import annotations

import asyncio
import json
from collections import Counter
from pathlib import Path
from typing import Any

from lightrag.kg.shared_storage import finalize_share_data

from canvas_exporter import CanvasExporter
from config import (
    BIBLIOGRAPHIC_KEYS,
    REL_AUTHORED_BY,
    REL_CITES,
    Settings,
    normalize_type,
)
from ingestion import LEDGER_FILENAME, IngestionEngine
from offline_backend import STUB_ANSWER_PREFIX, StubLLM, offline_rag_kwargs
from query import QueryInterface
from rag_factory import build_rag, graphml_path
from verify_integrity import verify_canvas

QUESTION = "What is Sparse Attention Routing and on which datasets was it evaluated?"


async def _run_pipeline(settings: Settings) -> dict[str, Any]:
    stub = StubLLM()
    rag = await build_rag(settings, **offline_rag_kwargs(stub))
    try:
        engine = IngestionEngine(rag, settings, stub)
        report = await engine.ingest_all()
        storage = rag.chunk_entity_relation_graph
        nodes = await storage.get_all_nodes()
        edges = await storage.get_all_edges()
        calls_after_ingest = Counter(call["kind"] for call in stub.calls)

        # Re-ingesting the same corpus must be idempotent: no new profiling or
        # extraction calls, and the graph must not grow.
        await engine.ingest_all()
        calls_after_reingest = Counter(call["kind"] for call in stub.calls)
        nodes_after_reingest = await storage.get_all_nodes()

        interface = QueryInterface(rag)
        results = await interface.compare_modes(QUESTION)
        calls_after_queries = Counter(call["kind"] for call in stub.calls)
        retrieval = await interface.retrieve("Who authored the paper?", "hybrid")

        exporter = await CanvasExporter.from_storage(storage)
        canvas_path = exporter.export(settings.canvas_path)
        return {
            "report": report,
            "nodes": nodes,
            "edges": edges,
            "nodes_after_reingest": nodes_after_reingest,
            "calls_after_ingest": calls_after_ingest,
            "calls_after_reingest": calls_after_reingest,
            "calls_after_queries": calls_after_queries,
            "calls_final": Counter(call["kind"] for call in stub.calls),
            "results": results,
            "retrieval": retrieval,
            "canvas_path": canvas_path,
        }
    finally:
        await rag.finalize_storages()
        finalize_share_data()


def test_offline_pipeline_end_to_end(tmp_path: Path, sample_paper: Path) -> None:
    settings = Settings().with_overrides(
        working_dir=tmp_path / "rag",
        data_dir=sample_paper.parent,
        canvas_path=tmp_path / "out" / "graph.canvas",
    )
    outcome = asyncio.run(_run_pipeline(settings))

    # -- ingestion -------------------------------------------------------- #
    report = outcome["report"]
    assert sample_paper.name in report.ingested and not report.skipped
    record = next(iter(report.profiles.values()))
    assert record.title.startswith("Sparse Attention Routing")
    assert len(record.authors) == 3 and record.year == 2024

    nodes, edges = outcome["nodes"], outcome["edges"]
    type_counts = Counter(normalize_type(n.get("entity_type")) for n in nodes)
    for key in BIBLIOGRAPHIC_KEYS:
        assert type_counts[key] >= 1, f"no {key} node: {type_counts}"
    assert type_counts["concept"] >= 1 and type_counts["method"] >= 1
    keywords = [str(e.get("keywords", "")) for e in edges]
    assert sum(REL_AUTHORED_BY in k for k in keywords) >= 3
    assert sum(REL_CITES in k for k in keywords) >= 4
    for node in nodes:
        assert node.get("description") and node.get("source_id")

    # -- stub call accounting --------------------------------------------- #
    calls = outcome["calls_after_ingest"]
    assert calls["profile"] == 1
    assert calls["extract"] >= 1
    assert calls["gleaning"] == 0  # entity_extract_max_gleaning=0 in offline mode
    reingest = outcome["calls_after_reingest"]
    assert reingest["profile"] == 1 and reingest["extract"] == calls["extract"]
    assert len(outcome["nodes_after_reingest"]) == len(nodes)

    # -- retrieval -------------------------------------------------------- #
    results = outcome["results"]
    for mode in ("hybrid", "mix"):
        result = results[mode]
        assert result.ok, result.summary()
        assert result.answer.startswith(STUB_ANSWER_PREFIX)
        assert result.entities and result.relationships
    assert results["mix"].chunks, "mix mode must surface vector-retrieved chunks"
    query_calls = outcome["calls_after_queries"]
    assert query_calls["keywords"] == 2 and query_calls["answer"] == 2
    # aquery_data still extracts keywords but never generates an answer.
    final_calls = outcome["calls_final"]
    assert final_calls["answer"] == 2 and final_calls["keywords"] >= 2
    assert outcome["retrieval"].get("entities")

    # -- export + integrity ----------------------------------------------- #
    canvas = json.loads(outcome["canvas_path"].read_text(encoding="utf-8"))
    verification = verify_canvas(canvas)
    assert verification.ok, verification.render()
    assert len(canvas["nodes"]) == len(nodes)
    assert len(canvas["edges"]) == len(edges)
    colors = Counter(n.get("color") for n in canvas["nodes"])
    assert colors["6"] == type_counts["paper"]
    assert colors["4"] == type_counts["author"]
    assert all(e["toEnd"] == "arrow" for e in canvas["edges"])

    # -- persistence ------------------------------------------------------ #
    assert graphml_path(settings).exists()
    ledger = json.loads((settings.working_dir / LEDGER_FILENAME).read_text("utf-8"))
    assert len(ledger) == 1


def test_cli_all_with_stub_backend(tmp_path: Path, sample_paper: Path, capsys) -> None:
    from main import main

    out = tmp_path / "cli.canvas"
    exit_code = main(
        [
            "--backend",
            "stub",
            "--working-dir",
            str(tmp_path / "rag"),
            "--data-dir",
            str(sample_paper.parent),
            "--log-level",
            "WARNING",
            "all",
            "--out",
            str(out),
        ]
    )
    captured = capsys.readouterr().out
    assert exit_code == 0
    assert "Ingested 1 document(s)" in captured
    assert STUB_ANSWER_PREFIX in captured
    assert "Integrity check: PASS" in captured
    assert verify_canvas(json.loads(out.read_text(encoding="utf-8"))).ok
