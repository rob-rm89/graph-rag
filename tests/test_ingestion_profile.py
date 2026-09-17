"""Pure-function tests for the bibliographic profiling layer in ingestion.py."""

from __future__ import annotations

from config import (
    BIBLIOGRAPHIC_KEYS,
    REL_AFFILIATED_WITH,
    REL_AUTHORED_BY,
    REL_CITES,
    REL_PUBLISHED_IN,
    REL_PUBLISHED_YEAR,
    SEMANTIC_KEYS,
)
from ingestion import (
    HEAD_CHARS,
    TAIL_CHARS,
    BibliographicRecord,
    build_profiling_prompt,
    fallback_title,
    parse_profile,
    record_to_custom_kg,
)


def test_parse_profile_coerces_and_dedupes() -> None:
    raw = {
        "title": " 'A Study' ",
        "authors": ["Ada Lovelace", "ada lovelace", "", None, "Alan Turing"],
        "year": "Published in 2024.",
        "venue": "",
        "doi": None,
        "affiliations": "Analytical Engine Society",
        "references": ["A Study", "Some Other Work", "Some Other Work"],
    }
    record = parse_profile(raw, fallback_title="fallback")
    assert record.title == "A Study"
    assert record.authors == ["Ada Lovelace", "Alan Turing"]
    assert record.year == 2024
    assert record.venue is None
    assert record.affiliations == ["Analytical Engine Society"]
    assert record.references == ["Some Other Work"]  # own title removed, deduped


def test_parse_profile_falls_back() -> None:
    record = parse_profile({}, fallback_title="From Heading")
    assert record.title == "From Heading"
    assert record.authors == [] and record.year is None
    assert parse_profile({"year": 1200}, fallback_title="x").year is None
    assert parse_profile({"year": True}, fallback_title="x").year is None


def test_fallback_title_prefers_heading() -> None:
    assert fallback_title("\n\n# My Paper Title\nbody", "f.md") == "My Paper Title"
    assert fallback_title("   \n", "notes.txt") == "notes"


def test_build_profiling_prompt_keeps_head_and_tail() -> None:
    text = "H" * HEAD_CHARS + "M" * 5000 + "T" * TAIL_CHARS
    prompt = build_profiling_prompt(text)
    assert "[...]" in prompt
    assert "H" * HEAD_CHARS in prompt and "T" * TAIL_CHARS in prompt
    assert "M" * 5000 not in prompt
    short = build_profiling_prompt("short text")
    assert "[...]" not in short and "short text" in short


def sample_record() -> BibliographicRecord:
    return BibliographicRecord(
        title="Graph Things",
        authors=["Ann Author", "Bob Writer"],
        year=2023,
        venue="Journal of Graphs",
        doi="10.1/xyz",
        affiliations=["Graph University"],
        references=["Older Work", "Even Older Work"],
    )


def test_record_to_custom_kg_shapes() -> None:
    kg = record_to_custom_kg(
        sample_record(), source_alias="doc-1-biblio", file_path="g.md"
    )
    assert set(kg) == {"chunks", "entities", "relationships"}

    chunk = kg["chunks"][0]
    assert chunk["source_id"] == "doc-1-biblio" and "Graph Things" in chunk["content"]

    names = {e["entity_name"] for e in kg["entities"]}
    assert names == {
        "Graph Things",
        "Ann Author",
        "Bob Writer",
        "Journal of Graphs",
        "2023",
        "Older Work",
        "Even Older Work",
        "Graph University",
    }
    for entity in kg["entities"]:
        assert entity["entity_type"] in BIBLIOGRAPHIC_KEYS | SEMANTIC_KEYS
        assert entity["entity_type"] == entity["entity_type"].lower()
        assert entity["source_id"] == "doc-1-biblio"
        assert entity["file_path"] == "g.md"
        assert entity["description"]

    for rel in kg["relationships"]:
        assert {
            "src_id",
            "tgt_id",
            "description",
            "keywords",
            "weight",
            "source_id",
        } <= set(rel)
        assert rel["src_id"] != rel["tgt_id"]
        assert rel["src_id"] in names and rel["tgt_id"] in names
        assert rel["weight"] == 1.0

    by_keyword: dict[str, set[tuple[str, str]]] = {}
    for rel in kg["relationships"]:
        by_keyword.setdefault(rel["keywords"], set()).add(
            (rel["src_id"], rel["tgt_id"])
        )
    assert by_keyword[REL_AUTHORED_BY] == {
        ("Graph Things", "Ann Author"),
        ("Graph Things", "Bob Writer"),
    }
    assert by_keyword[REL_PUBLISHED_IN] == {("Graph Things", "Journal of Graphs")}
    assert by_keyword[REL_PUBLISHED_YEAR] == {("Graph Things", "2023")}
    assert by_keyword[REL_CITES] == {
        ("Graph Things", "Older Work"),
        ("Graph Things", "Even Older Work"),
    }
    assert by_keyword[REL_AFFILIATED_WITH] == {
        ("Ann Author", "Graph University"),
        ("Bob Writer", "Graph University"),
    }


def test_record_to_custom_kg_dedupes_and_skips_self_loops() -> None:
    record = BibliographicRecord(
        title="Self",
        authors=["Self", "Dup", "Dup"],
        references=["Self", "Ref"],
    )
    kg = record_to_custom_kg(record, source_alias="a", file_path="f")
    names = [e["entity_name"] for e in kg["entities"]]
    assert len(names) == len(set(names))
    assert all(r["src_id"] != r["tgt_id"] for r in kg["relationships"])
    pairs = [(r["src_id"], r["tgt_id"]) for r in kg["relationships"]]
    assert len(pairs) == len(set(pairs))


def test_record_roundtrip_dict() -> None:
    record = sample_record()
    assert BibliographicRecord.from_dict(record.to_dict()) == record
