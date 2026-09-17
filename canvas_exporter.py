"""Visual Translation Layer.

Reads the LightRAG knowledge graph from its ``NetworkXStorage`` backend, computes
a force-directed (Fruchterman-Reingold) layout with NetworkX, scales it to
integer pixel coordinates, removes bounding-box overlaps, and serialises the
result as an Obsidian-compatible JSON Canvas 1.0 file (https://jsoncanvas.org).

Mapping:
* every LightRAG entity  -> one ``text`` node (Markdown body, preset colour)
* every graph edge       -> one edge with ``fromEnd="none"``, ``toEnd="arrow"``
* ids                    -> unique 16-character lower-case hex strings
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import networkx as nx
from lightrag.base import BaseGraphStorage
from lightrag.constants import GRAPH_FIELD_SEP
from lightrag.kg.networkx_impl import NetworkXStorage
from lightrag.utils import logger, setup_logger

from config import Settings, edge_color, node_color, normalize_type
from rag_factory import graphml_path

CANVAS_ID_LENGTH = 16
FRUCHTERMAN_REINGOLD = "fruchterman_reingold"
FORCEATLAS2 = "forceatlas2"


# --------------------------------------------------------------------------- #
# Configuration and geometry primitives
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class LayoutConfig:
    """Tunable layout parameters (all pixel values are integers)."""

    node_width: int = 400
    min_height: int = 160
    max_height: int = 520
    padding: int = 80  # spacing target used when scaling the force layout
    box_padding: int = 40  # minimum clear gap enforced between node boxes
    iterations: int = 200
    seed: int = 42
    max_passes: int = 100
    expand_every: int = 8  # dilate the component when overlaps persist this long
    expand_factor: float = 1.15
    density_factor: float = 2.2  # layout area / total padded box area
    grid: int = 20
    max_nodes: int | None = None
    algorithm: str = FRUCHTERMAN_REINGOLD
    description_chars: int = 350
    label_chars: int = 60

    def __post_init__(self) -> None:
        if self.grid > self.box_padding:
            raise ValueError("grid must not exceed box_padding or snapping may overlap")
        if self.algorithm not in (FRUCHTERMAN_REINGOLD, FORCEATLAS2):
            raise ValueError(f"unknown layout algorithm {self.algorithm!r}")


@dataclass
class Box:
    """Axis-aligned node box; ``x``/``y`` are the top-left corner."""

    node: str
    x: float
    y: float
    width: int
    height: int
    degree: int

    @property
    def cx(self) -> float:
        return self.x + self.width / 2

    @property
    def cy(self) -> float:
        return self.y + self.height / 2


def make_canvas_id(key: str, taken: set[str]) -> str:
    """Deterministic 16-hex id derived from ``key``; re-hashes on collision."""
    candidate = hashlib.blake2b(key.encode("utf-8"), digest_size=8).hexdigest()
    counter = 0
    while candidate in taken:
        counter += 1
        salted = f"{key}#{counter}".encode()
        candidate = hashlib.blake2b(salted, digest_size=8).hexdigest()
    taken.add(candidate)
    return candidate


def split_field(value: Any) -> list[str]:
    """Split a ``<SEP>``-joined LightRAG attribute into clean parts."""
    parts = str(value if value is not None else "").split(GRAPH_FIELD_SEP)
    return [part.strip() for part in parts if part and part.strip()]


def boxes_overlap(a: Box, b: Box, gap: float = 0.0) -> bool:
    """True when the boxes (inflated by ``gap``) intersect; touching is fine."""
    return (
        a.x < b.x + b.width + gap
        and b.x < a.x + a.width + gap
        and a.y < b.y + b.height + gap
        and b.y < a.y + a.height + gap
    )


def count_overlaps(boxes: Iterable[Box], gap: float = 0.0) -> int:
    ordered = sorted(boxes, key=lambda b: (b.x, b.y, b.node))
    overlaps = 0
    for i, a in enumerate(ordered):
        limit = a.x + a.width + gap
        for b in ordered[i + 1 :]:
            if b.x >= limit:
                break
            if boxes_overlap(a, b, gap):
                overlaps += 1
    return overlaps


def resolve_overlaps(
    boxes: list[Box],
    gap: int,
    max_passes: int,
    *,
    expand_every: int = 8,
    expand_factor: float = 1.15,
) -> list[Box]:
    """Deterministically push overlapping boxes apart until none remain.

    Each pass sorts boxes by x and sweeps once; every overlapping pair is
    separated along the axis of smaller penetration, with high-degree nodes
    moving less so hubs keep their force-layout position.  Whenever overlaps
    persist for ``expand_every`` passes the whole component is dilated about
    its centroid by ``expand_factor``; because the available area then grows
    geometrically, the process converges regardless of the initial density.
    """
    for pass_number in range(1, max_passes + 1):
        if not _separation_pass(boxes, gap):
            return boxes
        if pass_number % expand_every == 0 and count_overlaps(boxes, gap):
            _dilate(boxes, expand_factor)
    remaining = count_overlaps(boxes, gap)
    if remaining:
        logger.warning(
            "%d overlaps left after %d passes; forcing apart", remaining, max_passes
        )
        _force_separate(boxes, gap)
    return boxes


def _separation_pass(boxes: list[Box], gap: int) -> bool:
    """One sort-and-sweep pass; returns True when any pair had to move."""
    boxes.sort(key=lambda b: (b.x, b.y, b.node))
    moved = False
    for i, a in enumerate(boxes):
        for b in boxes[i + 1 :]:
            if b.x >= a.x + a.width + gap:
                break
            if not boxes_overlap(a, b, gap):
                continue
            moved = True
            pen_x = (a.x + a.width + gap) - b.x
            pen_y = min(a.y + a.height, b.y + b.height) + gap - max(a.y, b.y)
            wa = 1.0 / (1 + a.degree)
            wb = 1.0 / (1 + b.degree)
            total = wa + wb
            if pen_x <= pen_y:
                shift = pen_x + 1
                a.x -= shift * wa / total
                b.x += shift * wb / total
            else:
                shift = pen_y + 1
                direction = 1.0 if a.cy <= b.cy else -1.0
                a.y -= direction * shift * wa / total
                b.y += direction * shift * wb / total
    return moved


def _dilate(boxes: list[Box], factor: float) -> None:
    """Scale box centres away from their centroid (shape-preserving expansion)."""
    if len(boxes) < 2:
        return
    centre_x = sum(b.cx for b in boxes) / len(boxes)
    centre_y = sum(b.cy for b in boxes) / len(boxes)
    for box in boxes:
        box.x = centre_x + (box.cx - centre_x) * factor - box.width / 2
        box.y = centre_y + (box.cy - centre_y) * factor - box.height / 2


def _force_separate(boxes: list[Box], gap: int) -> None:
    """Fallback: unweighted full pairwise separation (O(n^2) per round)."""
    for _ in range(10):
        moved = False
        boxes.sort(key=lambda b: (b.x, b.y, b.node))
        for i, a in enumerate(boxes):
            for b in boxes[i + 1 :]:
                if boxes_overlap(a, b, gap):
                    moved = True
                    b.x = a.x + a.width + gap + 1
        if not moved:
            return


def landscape(
    positions: dict[str, tuple[float, float]],
) -> dict[str, tuple[float, float]]:
    """Transpose a unit layout that is taller than wide.

    Canvases are read left-to-right, so a component whose force layout came out
    elongated along y is mirrored across the diagonal.  Distances are preserved,
    the operation is deterministic, and it costs nothing.
    """
    if len(positions) < 2:
        return positions
    xs = [p[0] for p in positions.values()]
    ys = [p[1] for p in positions.values()]
    if (max(ys) - min(ys)) > (max(xs) - min(xs)):
        return {node: (y, x) for node, (x, y) in positions.items()}
    return positions


def shelf_pack(components: list[list[Box]], gap: float) -> list[Box]:
    """Place already laid-out components in rows so they never intersect."""
    items: list[tuple[list[Box], float, float, float, float]] = []
    for comp in components:
        min_x = min(b.x for b in comp)
        min_y = min(b.y for b in comp)
        width = max(b.x + b.width for b in comp) - min_x
        height = max(b.y + b.height for b in comp) - min_y
        items.append((comp, min_x, min_y, width, height))
    items.sort(key=lambda t: (-(t[3] * t[4]), t[0][0].node))

    total_area = sum(w * h for _, _, _, w, h in items)
    widest = max(w for _, _, _, w, _ in items)
    row_limit = max(math.sqrt(total_area) * 1.3, widest)

    cursor_x = 0.0
    cursor_y = 0.0
    row_height = 0.0
    placed: list[Box] = []
    for comp, min_x, min_y, width, height in items:
        if cursor_x > 0 and cursor_x + width > row_limit:
            cursor_x = 0.0
            cursor_y += row_height + gap
            row_height = 0.0
        dx = cursor_x - min_x
        dy = cursor_y - min_y
        for box in comp:
            box.x += dx
            box.y += dy
            placed.append(box)
        cursor_x += width + gap
        row_height = max(row_height, height)
    return placed


def _leads(description: str, name: str) -> bool:
    """True when ``description`` opens with ``name`` as its grammatical subject."""
    return description.startswith(f"'{name}'") or description.startswith(f"{name} ")


def orient_edge(
    u: str, v: str, type_u: str | None, type_v: str | None, description: Any
) -> bool:
    """Return True when the edge should be drawn from ``v`` to ``u``.

    NetworkX graphs are undirected, so a deterministic direction is derived in
    order of preference from: the relationship description naming exactly one
    endpoint as its subject (``"'A' cites 'B'"`` -> A to B), the paper-first
    rule for bibliographic edges, and finally alphabetical order.  The result
    never depends on storage iteration order.
    """
    first = next(iter(split_field(description)), "")
    u_leads, v_leads = _leads(first, u), _leads(first, v)
    if u_leads != v_leads:
        return v_leads
    paper_u = normalize_type(type_u) == "paper"
    paper_v = normalize_type(type_v) == "paper"
    if paper_u != paper_v:
        return paper_v
    return u > v


# --------------------------------------------------------------------------- #
# Graph loading
# --------------------------------------------------------------------------- #


def build_graph(nodes: list[dict[str, Any]], edges: list[dict[str, Any]]) -> nx.Graph:
    """Rebuild an undirected graph from ``get_all_nodes``/``get_all_edges`` output."""
    graph = nx.Graph()
    for node in sorted(nodes, key=lambda d: str(d.get("id"))):
        node_id = node.get("id")
        if node_id is None:
            continue
        graph.add_node(str(node_id), **{k: v for k, v in node.items() if k != "id"})
    for edge in sorted(
        edges, key=lambda d: (str(d.get("source")), str(d.get("target")))
    ):
        u, v = str(edge.get("source")), str(edge.get("target"))
        if u == v or u not in graph or v not in graph:
            continue
        graph.add_edge(
            u, v, **{k: v2 for k, v2 in edge.items() if k not in ("source", "target")}
        )
    return graph


# --------------------------------------------------------------------------- #
# Exporter
# --------------------------------------------------------------------------- #


class CanvasExporter:
    """Translate a NetworkX knowledge graph into a JSON Canvas document."""

    def __init__(self, graph: nx.Graph, layout: LayoutConfig | None = None) -> None:
        self.graph = graph
        self.layout = layout or LayoutConfig()

    # -- constructors ------------------------------------------------------- #

    @classmethod
    async def from_storage(
        cls, storage: BaseGraphStorage, layout: LayoutConfig | None = None
    ) -> CanvasExporter:
        """Read every node and edge from a live LightRAG graph storage."""
        nodes = await storage.get_all_nodes()
        edges = await storage.get_all_edges()
        logger.info(
            "Loaded %d nodes and %d edges from graph storage", len(nodes), len(edges)
        )
        return cls(build_graph(nodes, edges), layout)

    @classmethod
    def from_graphml(
        cls, path: Path | str, layout: LayoutConfig | None = None
    ) -> CanvasExporter:
        """Read the persisted GraphML written by ``NetworkXStorage``."""
        file = Path(path)
        if not file.exists():
            raise FileNotFoundError(f"GraphML file not found: {file}")
        graph = NetworkXStorage.load_nx_graph(str(file))
        if graph is None:
            raise FileNotFoundError(f"NetworkXStorage could not load {file}")
        logger.info(
            "Loaded %d nodes and %d edges from %s",
            graph.number_of_nodes(),
            graph.number_of_edges(),
            file,
        )
        return cls(nx.Graph(graph), layout)

    @classmethod
    def from_settings(
        cls, settings: Settings, layout: LayoutConfig | None = None
    ) -> CanvasExporter:
        return cls.from_graphml(graphml_path(settings), layout)

    # -- layout ------------------------------------------------------------- #

    def prune(self, graph: nx.Graph) -> nx.Graph:
        limit = self.layout.max_nodes
        if limit is None or graph.number_of_nodes() <= limit:
            return graph
        ranked = sorted(graph.nodes, key=lambda n: (-graph.degree(n), str(n)))
        keep = ranked[:limit]
        logger.info("Pruning graph to the %d highest-degree nodes", limit)
        return nx.Graph(graph.subgraph(keep))

    def _force_layout(self, component: nx.Graph) -> dict[str, tuple[float, float]]:
        cfg = self.layout
        n = component.number_of_nodes()
        if n == 1:
            return {next(iter(component)): (0.0, 0.0)}
        if cfg.algorithm == FORCEATLAS2 and hasattr(nx, "forceatlas2_layout"):
            pos = nx.forceatlas2_layout(
                component, max_iter=cfg.iterations, seed=cfg.seed, strong_gravity=True
            )
        else:
            pos = nx.spring_layout(
                component,
                k=2.5 / math.sqrt(n),
                iterations=cfg.iterations,
                seed=cfg.seed,
                weight=None,
            )
        pos = nx.rescale_layout_dict(pos, scale=1.0)
        return {node: (float(p[0]), float(p[1])) for node, p in pos.items()}

    def _pixel_scale(
        self, positions: dict[str, tuple[float, float]], sizes: list[tuple[int, int]]
    ) -> float:
        """Isotropic factor mapping unit layout coordinates to pixels.

        The factor makes the component's bounding box ``density_factor`` times
        the total padded box area, so the overlap resolver has little to do
        while the force layout's shape is preserved (no anisotropic stretch).
        """
        cfg = self.layout
        padded_area = sum(
            (width + cfg.box_padding) * (height + cfg.box_padding)
            for width, height in sizes
        )
        target_area = padded_area * cfg.density_factor
        xs = [p[0] for p in positions.values()]
        ys = [p[1] for p in positions.values()]
        extent_x = max(xs) - min(xs)
        extent_y = max(ys) - min(ys)
        if extent_x * extent_y > 1e-9:
            return math.sqrt(target_area / (extent_x * extent_y))
        extent = max(extent_x, extent_y)
        return math.sqrt(target_area) / extent if extent > 1e-9 else 0.0

    def compute_layout(self, graph: nx.Graph) -> dict[str, Box]:
        """Force layout per component, pixel scaling, overlap removal, packing."""
        cfg = self.layout
        if graph.number_of_nodes() == 0:
            return {}
        sizes = {
            node: self._node_size(self._node_text(node, graph.nodes[node]))
            for node in graph
        }

        components = sorted(
            nx.connected_components(graph), key=lambda c: (-len(c), min(c))
        )
        laid_out: list[list[Box]] = []
        for comp_nodes in components:
            component = nx.Graph()
            component.add_nodes_from(sorted(comp_nodes))
            component.add_edges_from(
                sorted(
                    (min(u, v), max(u, v))
                    for u, v in graph.subgraph(comp_nodes).edges()
                )
            )
            positions = landscape(self._force_layout(component))
            scale = self._pixel_scale(positions, [sizes[node] for node in comp_nodes])
            boxes = [
                Box(
                    node=node,
                    x=positions[node][0] * scale - sizes[node][0] / 2,
                    y=positions[node][1] * scale - sizes[node][1] / 2,
                    width=sizes[node][0],
                    height=sizes[node][1],
                    degree=graph.degree(node),
                )
                for node in sorted(comp_nodes)
            ]
            laid_out.append(
                resolve_overlaps(
                    boxes,
                    cfg.box_padding,
                    cfg.max_passes,
                    expand_every=cfg.expand_every,
                    expand_factor=cfg.expand_factor,
                )
            )

        placed = shelf_pack(laid_out, gap=float(cfg.padding * 2))
        min_x = min(b.x for b in placed)
        min_y = min(b.y for b in placed)
        for box in placed:
            box.x = float(cfg.grid * round((box.x - min_x) / cfg.grid))
            box.y = float(cfg.grid * round((box.y - min_y) / cfg.grid))

        overlaps = count_overlaps(placed)
        if overlaps:
            raise RuntimeError(f"layout produced {overlaps} overlapping node boxes")
        logger.info(
            "Layout complete: %d nodes, %d components, 0 overlaps",
            len(placed),
            len(components),
        )
        return {box.node: box for box in placed}

    # -- node rendering ----------------------------------------------------- #

    def _node_text(self, name: str, attrs: dict[str, Any]) -> str:
        cfg = self.layout
        entity_type = normalize_type(attrs.get("entity_type")) or "unknown"
        description = " ".join(split_field(attrs.get("description")))
        if len(description) > cfg.description_chars:
            description = description[: cfg.description_chars - 1].rstrip() + "…"
        parts = [f"# {name}", f"**Type:** {entity_type}"]
        if description:
            parts.append(description)
        sources = split_field(attrs.get("file_path"))
        source = next((s for s in sources if s and s != "unknown_source"), None)
        if source:
            parts.append(f"_Source: {source}_")
        return "\n\n".join(parts)

    def _node_size(self, text: str) -> tuple[int, int]:
        cfg = self.layout
        chars_per_line = max(10, cfg.node_width // 9)
        lines = sum(
            max(1, math.ceil(len(line) / chars_per_line)) for line in text.split("\n")
        )
        height = 60 + 24 * lines
        height = max(cfg.min_height, min(cfg.max_height, height))
        return int(cfg.node_width), int(height)

    # -- serialisation ------------------------------------------------------ #

    @staticmethod
    def _sides(src: Box, dst: Box) -> tuple[str, str]:
        dx = dst.cx - src.cx
        dy = dst.cy - src.cy
        if abs(dx) >= abs(dy):
            return ("right", "left") if dx >= 0 else ("left", "right")
        return ("bottom", "top") if dy >= 0 else ("top", "bottom")

    def build_canvas(self) -> dict[str, list[dict[str, Any]]]:
        """Return the JSON Canvas document as a plain dict."""
        graph = self.prune(self.graph)
        boxes = self.compute_layout(graph)
        taken: set[str] = set()
        node_ids: dict[str, str] = {}
        nodes: list[dict[str, Any]] = []

        for name in sorted(graph.nodes):
            attrs = graph.nodes[name]
            box = boxes[name]
            canvas_id = make_canvas_id(f"node:{name}", taken)
            node_ids[name] = canvas_id
            node: dict[str, Any] = {
                "id": canvas_id,
                "type": "text",
                "x": int(box.x),
                "y": int(box.y),
                "width": int(box.width),
                "height": int(box.height),
                "text": self._node_text(name, attrs),
            }
            color = node_color(attrs.get("entity_type"))
            if color:
                node["color"] = color
            nodes.append(node)

        # Orient first, then sort on the canonical pair so the output never
        # depends on the order in which storage yielded the undirected edges.
        oriented: list[tuple[str, str, Any, Any, dict[str, Any]]] = []
        for u, v, attrs in graph.edges(data=True):
            type_u = graph.nodes[u].get("entity_type")
            type_v = graph.nodes[v].get("entity_type")
            if orient_edge(u, v, type_u, type_v, attrs.get("description")):
                u, v, type_u, type_v = v, u, type_v, type_u
            oriented.append((u, v, type_u, type_v, attrs))

        edges: list[dict[str, Any]] = []
        for u, v, type_u, type_v, attrs in sorted(oriented, key=lambda t: (t[0], t[1])):
            from_side, to_side = self._sides(boxes[u], boxes[v])
            edge: dict[str, Any] = {
                "id": make_canvas_id(f"edge:{u}->{v}", taken),
                "fromNode": node_ids[u],
                "fromSide": from_side,
                "fromEnd": "none",
                "toNode": node_ids[v],
                "toSide": to_side,
                "toEnd": "arrow",
                "color": edge_color(type_u, type_v),
            }
            keywords = ", ".join(split_field(attrs.get("keywords")))
            if keywords:
                edge["label"] = keywords[: self.layout.label_chars]
            edges.append(edge)

        logger.info("Canvas built: %d nodes, %d edges", len(nodes), len(edges))
        return {"nodes": nodes, "edges": edges}

    def export(self, path: Path | str) -> Path:
        """Write the canvas to ``path`` (creating parent directories)."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        canvas = self.build_canvas()
        target.write_text(
            json.dumps(canvas, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        logger.info("JSON Canvas written to %s", target)
        return target


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def main(argv: list[str] | None = None) -> Path:
    parser = argparse.ArgumentParser(
        description="Export the knowledge graph to JSON Canvas."
    )
    parser.add_argument(
        "--working-dir", default=None, help="LightRAG working directory"
    )
    parser.add_argument("--workspace", default=None)
    parser.add_argument("--out", default=None, help="Output .canvas path")
    parser.add_argument("--max-nodes", type=int, default=None)
    parser.add_argument(
        "--algorithm",
        choices=(FRUCHTERMAN_REINGOLD, FORCEATLAS2),
        default=FRUCHTERMAN_REINGOLD,
    )
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)

    setup_logger("lightrag", level=args.log_level, enable_file_logging=False)
    settings = Settings()
    overrides: dict[str, Any] = {}
    if args.working_dir:
        overrides["working_dir"] = Path(args.working_dir)
    if args.workspace is not None:
        overrides["workspace"] = args.workspace
    if args.out:
        overrides["canvas_path"] = Path(args.out)
    settings = settings.with_overrides(**overrides)

    layout = LayoutConfig(max_nodes=args.max_nodes, algorithm=args.algorithm)
    exporter = CanvasExporter.from_settings(settings, layout)
    return exporter.export(settings.canvas_path)


if __name__ == "__main__":
    main()
