"""Unit tests for canvas_exporter against a synthetic NetworkX graph (no LLM)."""

from __future__ import annotations

import json
import random
import re
from pathlib import Path

import networkx as nx
import pytest

from canvas_exporter import (
    Box,
    CanvasExporter,
    LayoutConfig,
    count_overlaps,
    make_canvas_id,
    resolve_overlaps,
)
from config import NODE_COLOR_BY_TYPE
from verify_integrity import verify_canvas

HEX16 = re.compile(r"^[0-9a-f]{16}$")
TYPES = (
    "paper",
    "author",
    "venue",
    "year",
    "citedwork",
    "concept",
    "method",
    "dataset",
    "metric",
    "organization",
    "other",
)


def synthetic_graph(n: int = 250, seed: int = 7, isolated: int = 5) -> nx.Graph:
    rng = random.Random(seed)
    graph = nx.Graph()
    for i in range(n):
        graph.add_node(
            f"Entity {i}",
            entity_id=f"Entity {i}",
            entity_type=TYPES[i % len(TYPES)],
            description=f"Description of entity {i}. " * (i % 7 + 1),
            source_id="chunk-abc",
            file_path="paper.md",
        )
    for i in range(1, n):
        j = rng.randrange(0, i)
        graph.add_edge(
            f"Entity {i}",
            f"Entity {j}",
            weight=1.0,
            keywords="related, synthetic",
            description="synthetic relation",
            source_id="chunk-abc",
            file_path="paper.md",
        )
    for _ in range(n // 3):
        u, v = rng.sample(range(n), 2)
        graph.add_edge(f"Entity {u}", f"Entity {v}", weight=2.0, keywords="extra")
    for k in range(isolated):
        graph.add_node(f"Isolated {k}", entity_type="concept", description="lonely")
    return graph


@pytest.fixture(scope="module")
def canvas() -> dict:
    return CanvasExporter(synthetic_graph()).build_canvas()


def test_canvas_has_arrays_and_unique_hex_ids(canvas: dict) -> None:
    graph = synthetic_graph()
    assert set(canvas) == {"nodes", "edges"}
    assert len(canvas["nodes"]) == graph.number_of_nodes()
    assert len(canvas["edges"]) == graph.number_of_edges()
    ids = [n["id"] for n in canvas["nodes"]] + [e["id"] for e in canvas["edges"]]
    assert len(ids) == len(set(ids))
    assert all(HEX16.match(i) for i in ids)


def test_geometry_is_strict_integers(canvas: dict) -> None:
    for node in canvas["nodes"]:
        for key in ("x", "y", "width", "height"):
            assert type(node[key]) is int, (key, node[key])
        assert node["width"] > 0 and node["height"] > 0
        assert node["type"] == "text"
        assert isinstance(node["text"], str) and node["text"].startswith("# ")


def test_no_overlaps_and_passes_verifier(canvas: dict) -> None:
    report = verify_canvas(canvas)
    assert report.ok, report.render()
    assert report.stats["overlapping_pairs"] == 0


def test_colors_follow_type_map(canvas: dict) -> None:
    type_re = re.compile(r"\*\*Type:\*\* (\w+)")
    seen: set[str] = set()
    for node in canvas["nodes"]:
        entity_type = type_re.search(node["text"]).group(1)
        seen.add(entity_type)
        expected = NODE_COLOR_BY_TYPE.get(entity_type)
        if expected is None:
            assert "color" not in node
        else:
            assert node["color"] == expected
    assert {"paper", "author", "concept", "other"} <= seen


def test_edges_reference_nodes_with_arrows(canvas: dict) -> None:
    node_ids = {n["id"] for n in canvas["nodes"]}
    for edge in canvas["edges"]:
        assert edge["fromNode"] in node_ids
        assert edge["toNode"] in node_ids
        assert edge["fromEnd"] == "none"
        assert edge["toEnd"] == "arrow"
        assert edge["color"] in {"2", "4"}
        assert edge["fromSide"] in {"top", "right", "bottom", "left"}
        assert edge["toSide"] in {"top", "right", "bottom", "left"}


def test_bibliographic_edges_point_away_from_paper() -> None:
    graph = nx.Graph()
    graph.add_node("A Paper", entity_type="paper", description="p")
    graph.add_node("An Author", entity_type="author", description="a")
    graph.add_node("A Concept", entity_type="concept", description="c")
    graph.add_edge("An Author", "A Paper", keywords="authored_by")
    graph.add_edge("A Concept", "A Paper", keywords="discusses")
    canvas = CanvasExporter(graph).build_canvas()
    ids = {n["text"].split("\n")[0][2:]: n["id"] for n in canvas["nodes"]}
    for edge in canvas["edges"]:
        assert edge["fromNode"] == ids["A Paper"]
        assert edge["color"] == "4"  # paper endpoint -> bibliographic colour
    assert {e["label"] for e in canvas["edges"]} == {"authored_by", "discusses"}


def test_max_nodes_prunes_by_degree() -> None:
    graph = synthetic_graph(n=120)
    canvas = CanvasExporter(graph, LayoutConfig(max_nodes=40)).build_canvas()
    assert len(canvas["nodes"]) == 40
    node_ids = {n["id"] for n in canvas["nodes"]}
    assert all(
        e["fromNode"] in node_ids and e["toNode"] in node_ids for e in canvas["edges"]
    )
    assert verify_canvas(canvas).ok


def test_export_is_deterministic(tmp_path: Path) -> None:
    graph = synthetic_graph(n=80)
    first = (
        CanvasExporter(graph).export(tmp_path / "a.canvas").read_text(encoding="utf-8")
    )
    second = (
        CanvasExporter(graph).export(tmp_path / "b.canvas").read_text(encoding="utf-8")
    )
    assert first == second
    assert json.loads(first)["nodes"]


def test_forceatlas2_algorithm_if_available() -> None:
    if not hasattr(nx, "forceatlas2_layout"):
        pytest.skip("networkx without forceatlas2_layout")
    canvas = CanvasExporter(
        synthetic_graph(n=60), LayoutConfig(algorithm="forceatlas2")
    ).build_canvas()
    assert verify_canvas(canvas).ok


def test_make_canvas_id_avoids_collisions() -> None:
    taken: set[str] = set()
    first = make_canvas_id("same-key", taken)
    second = make_canvas_id("same-key", taken)
    assert first != second
    assert HEX16.match(first) and HEX16.match(second)
    assert taken == {first, second}


def test_resolve_overlaps_separates_stacked_boxes() -> None:
    boxes = [
        Box(node=f"n{i}", x=0.0, y=0.0, width=400, height=200, degree=i % 4)
        for i in range(40)
    ]
    resolved = resolve_overlaps(boxes, gap=40, max_passes=60)
    assert count_overlaps(resolved, gap=40) == 0
    assert count_overlaps(resolved) == 0


def test_from_graphml_roundtrip(tmp_path: Path) -> None:
    graph = synthetic_graph(n=30, isolated=0)
    path = tmp_path / "graph_chunk_entity_relation.graphml"
    nx.write_graphml(graph, path)
    exporter = CanvasExporter.from_graphml(path)
    canvas = exporter.build_canvas()
    assert len(canvas["nodes"]) == 30
    assert verify_canvas(canvas).ok


def test_from_graphml_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        CanvasExporter.from_graphml(tmp_path / "missing.graphml")


def test_empty_graph_exports_empty_arrays() -> None:
    assert CanvasExporter(nx.Graph()).build_canvas() == {"nodes": [], "edges": []}


def test_layout_config_rejects_bad_grid() -> None:
    with pytest.raises(ValueError):
        LayoutConfig(grid=100, box_padding=40)


def test_landscape_transposes_tall_layouts() -> None:
    from canvas_exporter import landscape

    tall = {"a": (0.0, -1.0), "b": (0.1, 1.0), "c": (-0.1, 0.0)}
    wide = {"a": (-1.0, 0.0), "b": (1.0, 0.1), "c": (0.0, -0.1)}
    assert landscape(tall) == {"a": (-1.0, 0.0), "b": (1.0, 0.1), "c": (0.0, -0.1)}
    assert landscape(wide) == wide
    assert landscape({"only": (0.3, 0.7)}) == {"only": (0.3, 0.7)}


def test_exported_canvas_is_landscape() -> None:
    canvas = CanvasExporter(synthetic_graph(n=40, isolated=0)).build_canvas()
    xs = [n["x"] + n["width"] for n in canvas["nodes"]]
    ys = [n["y"] + n["height"] for n in canvas["nodes"]]
    assert max(xs) >= max(ys)


def test_orient_edge_rules() -> None:
    from canvas_exporter import orient_edge

    # Description names the subject: citing paper -> cited paper, either order given.
    desc = "'Paper B' cites 'Paper A'."
    assert orient_edge("Paper A", "Paper B", "paper", "paper", desc) is True
    assert orient_edge("Paper B", "Paper A", "paper", "paper", desc) is False
    unquoted = "Elena Marchetti is listed with affiliation KTH on 'Paper A'."
    assert orient_edge("KTH", "Elena Marchetti", "organization", "author", unquoted)
    # Paper-first when the description is uninformative.
    assert orient_edge("Author", "Paper", "author", "paper", "merged summary") is True
    assert orient_edge("Paper", "Author", "paper", "author", None) is False
    # Alphabetical fallback for semantic edges.
    assert orient_edge("Zeta", "Alpha", "concept", "method", "") is True
    assert orient_edge("Alpha", "Zeta", "concept", "method", "") is False


def test_export_independent_of_edge_insertion_order(tmp_path: Path) -> None:
    forward = synthetic_graph(n=60, isolated=0)
    reversed_graph = nx.Graph()
    reversed_graph.add_nodes_from(reversed(list(forward.nodes(data=True))))
    reversed_graph.add_edges_from(
        (v, u, d) for u, v, d in reversed(list(forward.edges(data=True)))
    )
    first = CanvasExporter(forward).export(tmp_path / "f.canvas").read_text("utf-8")
    second = (
        CanvasExporter(reversed_graph).export(tmp_path / "r.canvas").read_text("utf-8")
    )
    assert first == second
