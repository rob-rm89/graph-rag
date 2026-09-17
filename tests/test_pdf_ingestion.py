"""PDF ingestion: text extraction, dispatch, discovery and an offline end-to-end run."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from lightrag.kg.shared_storage import finalize_share_data

from config import SUPPORTED_SUFFIXES, Settings
from ingestion import (
    IngestionEngine,
    extract_pdf_text,
    normalise_pdf_text,
    read_document,
)
from offline_backend import StubLLM, offline_rag_kwargs
from rag_factory import build_rag


def make_pdf(path: Path, pages: list[str]) -> Path:
    """Write a minimal but valid PDF with one Helvetica text line per page.

    Built from raw bytes with a correct cross-reference table so the test does
    not depend on any private pypdf writer API.
    """
    kids: list[str] = []
    objects: list[tuple[int, bytes]] = []
    next_num = 4
    for text in pages:
        page_num, content_num = next_num, next_num + 1
        next_num += 2
        kids.append(f"{page_num} 0 R")
        content = f"BT /F1 18 Tf 72 720 Td ({text}) Tj ET".encode("latin-1")
        objects.append(
            (
                page_num,
                (
                    "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
                    "/Resources << /Font << /F1 3 0 R >> >> "
                    f"/Contents {content_num} 0 R >>"
                ).encode(),
            )
        )
        objects.append(
            (
                content_num,
                b"<< /Length %d >>\nstream\n" % len(content) + content + b"\nendstream",
            )
        )
    fixed = [
        (1, b"<< /Type /Catalog /Pages 2 0 R >>"),
        (
            2,
            f"<< /Type /Pages /Kids [{' '.join(kids)}] /Count {len(pages)} >>".encode(),
        ),
        (3, b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"),
    ]
    all_objects = fixed + objects
    out = bytearray(b"%PDF-1.4\n")
    offsets: dict[int, int] = {}
    for num, body in all_objects:
        offsets[num] = len(out)
        out += f"{num} 0 obj\n".encode() + body + b"\nendobj\n"
    xref_pos = len(out)
    count = len(all_objects) + 1
    out += f"xref\n0 {count}\n".encode() + b"0000000000 65535 f \n"
    for num in range(1, count):
        out += f"{offsets[num]:010d} 00000 n \n".encode()
    out += (
        f"trailer\n<< /Size {count} /Root 1 0 R >>\nstartxref\n{xref_pos}\n%%EOF\n"
    ).encode()
    path.write_bytes(bytes(out))
    return path


def test_pdf_suffix_is_supported() -> None:
    assert ".pdf" in SUPPORTED_SUFFIXES


def test_extract_pdf_text_preserves_page_order(tmp_path: Path) -> None:
    pdf = make_pdf(
        tmp_path / "two.pdf", ["First page about NarrativeQA", "Second page"]
    )
    text = extract_pdf_text(pdf)
    assert "First page about NarrativeQA" in text
    assert "Second page" in text
    assert text.index("First page") < text.index("Second page")
    assert "\n\n" in text  # pages separated by a blank line


def test_read_document_dispatches_by_suffix(tmp_path: Path) -> None:
    md = tmp_path / "note.md"
    md.write_text("# Title\nbody", encoding="utf-8")
    assert read_document(md).startswith("# Title")

    pdf = make_pdf(tmp_path / "doc.pdf", ["Hello PDF world"])
    assert "Hello PDF world" in read_document(pdf)

    with pytest.raises(ValueError, match="unsupported file type"):
        read_document(tmp_path / "image.png")


def test_read_document_rejects_textless_and_corrupt_pdfs(tmp_path: Path) -> None:
    blank = make_pdf(tmp_path / "blank.pdf", [""])
    with pytest.raises(ValueError, match="no extractable text"):
        read_document(blank)

    corrupt = tmp_path / "corrupt.pdf"
    corrupt.write_bytes(b"%PDF-1.4\nthis is not really a pdf")
    with pytest.raises(ValueError, match="cannot read PDF|no extractable text"):
        read_document(corrupt)


def test_normalise_pdf_text_repairs_hyphenation_and_blank_runs() -> None:
    raw = "Retrieval-Aug-\nmented Genera-\ntion\n\n\n\n\nNext paragraph"
    assert normalise_pdf_text(raw) == "Retrieval-Augmented Generation\n\nNext paragraph"


def test_pdf_flows_through_offline_pipeline(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    make_pdf(
        data_dir / "routing_note.pdf",
        [
            "Notes on Sparse Attention Routing",
            "The method was evaluated on NarrativeQA and LongBench.",
        ],
    )
    (data_dir / "scan.pdf").write_bytes(b"%PDF-1.4\n%% no objects at all")
    settings = Settings().with_overrides(
        working_dir=tmp_path / "rag",
        data_dir=data_dir,
        canvas_path=tmp_path / "c.canvas",
    )

    async def run() -> tuple[list[str], list[str], list[dict]]:
        stub = StubLLM()
        rag = await build_rag(settings, **offline_rag_kwargs(stub))
        try:
            engine = IngestionEngine(rag, settings, stub)
            assert [p.name for p in engine.discover_documents()] == [
                "routing_note.pdf",
                "scan.pdf",
            ]
            report = await engine.ingest_all()
            nodes = await rag.chunk_entity_relation_graph.get_all_nodes()
            return report.ingested, report.skipped, nodes
        finally:
            await rag.finalize_storages()
            finalize_share_data()

    ingested, skipped, nodes = asyncio.run(run())
    assert ingested == ["routing_note.pdf"]
    assert skipped == ["scan.pdf"]
    names = {node["id"] for node in nodes}
    assert {"Sparse Attention Routing", "NarrativeQA", "LongBench"} <= names
    assert all("routing_note.pdf" in str(node.get("file_path", "")) for node in nodes)
