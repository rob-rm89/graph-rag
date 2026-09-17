"""Command-line orchestrator linking ingestion, retrieval and canvas export.

Examples::

    python main.py --backend stub all
    python main.py ingest
    python main.py query "Who authored the paper?" --mode both
    python main.py export --out output/knowledge_graph.canvas --max-nodes 500
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path
from typing import Any

from lightrag import LightRAG
from lightrag.utils import logger, setup_logger

from canvas_exporter import (
    FORCEATLAS2,
    FRUCHTERMAN_REINGOLD,
    CanvasExporter,
    LayoutConfig,
)
from config import Settings
from ingestion import IngestionEngine
from llm import LLMFunc, make_llm_func
from query import DEMO_QUESTIONS, QueryInterface, format_result
from rag_factory import rag_session
from reconciliation import MergePlan
from verify_integrity import load_canvas, verify_canvas

BACKEND_OPENAI = "openai"
BACKEND_STUB = "stub"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="graph-rag", description="Unified Graph-RAG System pipeline."
    )
    parser.add_argument(
        "--backend",
        choices=(BACKEND_OPENAI, BACKEND_STUB),
        default=BACKEND_OPENAI,
        help="LLM backend: OpenAI-compatible API or the deterministic offline stub",
    )
    parser.add_argument("--working-dir", type=Path, default=None)
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--log-level", default="INFO")

    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser(
        "ingest", help="Profile and index every .txt/.md/.pdf in the data directory"
    )

    query = sub.add_parser("query", help="Ask questions in hybrid and/or mix mode")
    query.add_argument("question", nargs="*", help="Defaults to the demo questions")
    query.add_argument("--mode", choices=("hybrid", "mix", "both"), default="both")

    export = sub.add_parser("export", help="Export the knowledge graph to JSON Canvas")
    _add_export_args(export)

    sub.add_parser(
        "reconcile", help="Merge near-duplicate entities already in the graph"
    )

    everything = sub.add_parser("all", help="ingest -> query -> export in one run")
    everything.add_argument(
        "question", nargs="*", help="Defaults to the demo questions"
    )
    everything.add_argument("--mode", choices=("hybrid", "mix", "both"), default="both")
    _add_export_args(everything)
    return parser


def _add_export_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--out", type=Path, default=None, help="Output .canvas path")
    parser.add_argument("--max-nodes", type=int, default=None)
    parser.add_argument(
        "--algorithm",
        choices=(FRUCHTERMAN_REINGOLD, FORCEATLAS2),
        default=FRUCHTERMAN_REINGOLD,
    )
    parser.add_argument(
        "--no-verify", action="store_true", help="Skip the integrity check after export"
    )


def resolve_settings(args: argparse.Namespace) -> Settings:
    settings = Settings()
    overrides: dict[str, Any] = {}
    if args.working_dir is not None:
        overrides["working_dir"] = args.working_dir
    if args.data_dir is not None:
        overrides["data_dir"] = args.data_dir
    if getattr(args, "out", None) is not None:
        overrides["canvas_path"] = args.out
    return settings.with_overrides(**overrides) if overrides else settings


def resolve_backend(backend: str, settings: Settings) -> tuple[dict[str, Any], LLMFunc]:
    """Return (build_rag kwargs, profiling LLM) for the chosen backend."""
    if backend == BACKEND_STUB:
        from offline_backend import StubLLM, offline_rag_kwargs

        logger.warning("Using the deterministic offline stub backend (no real LLM)")
        stub = StubLLM()
        return offline_rag_kwargs(stub), stub
    if not settings.openai_api_key:
        raise SystemExit(
            "OPENAI_API_KEY is not set. Add it to .env or run with --backend stub."
        )
    return {}, make_llm_func(settings.extract_model, settings)


async def run_ingest(rag: LightRAG, settings: Settings, profiling_llm: LLMFunc) -> None:
    engine = IngestionEngine(rag, settings, profiling_llm)
    report = await engine.ingest_all()
    print(f"Ingested {len(report.ingested)} document(s); skipped {len(report.skipped)}")
    for doc_id, record in report.profiles.items():
        print(
            f"  {doc_id}: {record.title!r} ({record.year}) "
            f"authors={len(record.authors)}"
        )
    for title, targets in report.linked_papers.items():
        print(f"  {title!r} cites ingested paper(s): {targets}")
    print_merges(report.merges)


def print_merges(merges: list[MergePlan]) -> None:
    print(f"Entity merges applied: {len(merges)}")
    for plan in merges:
        print(f"  {', '.join(plan.sources)} -> {plan.target} ({plan.reason})")


async def run_reconcile(
    rag: LightRAG, settings: Settings, profiling_llm: LLMFunc
) -> None:
    engine = IngestionEngine(rag, settings, profiling_llm)
    print_merges(await engine.reconcile_graph())


async def run_query(rag: LightRAG, questions: list[str], mode: str) -> None:
    interface = QueryInterface(rag)
    for question in questions:
        if mode == "both":
            results = list((await interface.compare_modes(question)).values())
        elif mode == "hybrid":
            results = [await interface.hybrid(question)]
        else:
            results = [await interface.mix(question)]
        for result in results:
            print(format_result(result))
            print()


async def run_export(
    rag: LightRAG,
    settings: Settings,
    *,
    max_nodes: int | None,
    algorithm: str,
    verify: bool,
) -> Path:
    layout = LayoutConfig(max_nodes=max_nodes, algorithm=algorithm)
    exporter = await CanvasExporter.from_storage(
        rag.chunk_entity_relation_graph, layout
    )
    path = exporter.export(settings.canvas_path)
    print(f"Canvas written to {path}")
    if verify:
        report = verify_canvas(load_canvas(path))
        print(report.render())
        if not report.ok:
            raise SystemExit(1)
    return path


async def run(args: argparse.Namespace) -> int:
    settings = resolve_settings(args)
    rag_kwargs, profiling_llm = resolve_backend(args.backend, settings)
    command = args.command
    async with rag_session(settings, **rag_kwargs) as rag:
        if command in ("ingest", "all"):
            await run_ingest(rag, settings, profiling_llm)
        if command == "reconcile":
            await run_reconcile(rag, settings, profiling_llm)
        if command in ("query", "all"):
            questions = args.question or list(DEMO_QUESTIONS)
            await run_query(rag, questions, args.mode)
        if command in ("export", "all"):
            await run_export(
                rag,
                settings,
                max_nodes=args.max_nodes,
                algorithm=args.algorithm,
                verify=not args.no_verify,
            )
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    setup_logger("lightrag", level=args.log_level, enable_file_logging=False)
    return asyncio.run(run(args))


if __name__ == "__main__":
    sys.exit(main())
