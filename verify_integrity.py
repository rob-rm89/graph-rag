"""Standalone integrity verifier for JSON Canvas files.

Loads a ``.canvas`` file, confirms it carries ``nodes`` and ``edges`` arrays, and
checks every structural guarantee this project makes on top of the JSON Canvas
1.0 spec:

* every node/edge id is a unique 16-character lower-case hex string,
* ``x``, ``y``, ``width``, ``height`` are strict integers,
* every ``fromNode`` / ``toNode`` references an existing node id,
* colours, sides and end styles use spec-allowed values,
* no two node bounding boxes overlap.

Usage::

    python verify_integrity.py output/knowledge_graph.canvas
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from lightrag.utils import logger, setup_logger

HEX16_RE = re.compile(r"^[0-9a-f]{16}$")
HEX_COLOR_RE = re.compile(r"^#[0-9A-Fa-f]{6}$")
PRESET_COLORS = frozenset({"1", "2", "3", "4", "5", "6"})
NODE_TYPES = frozenset({"text", "file", "link", "group"})
SIDES = frozenset({"top", "right", "bottom", "left"})
ENDS = frozenset({"none", "arrow"})
GEOMETRY_KEYS = ("x", "y", "width", "height")
MAX_REPORTED_OVERLAPS = 20


@dataclass
class VerificationReport:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    stats: dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.errors

    def render(self) -> str:
        lines = [f"Integrity check: {'PASS' if self.ok else 'FAIL'}"]
        for key, value in self.stats.items():
            lines.append(f"  {key}: {value}")
        if self.warnings:
            lines.append(f"Warnings ({len(self.warnings)}):")
            lines.extend(f"  - {w}" for w in self.warnings)
        if self.errors:
            lines.append(f"Errors ({len(self.errors)}):")
            lines.extend(f"  - {e}" for e in self.errors)
        return "\n".join(lines)


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _valid_color(value: Any) -> bool:
    return isinstance(value, str) and (
        value in PRESET_COLORS or bool(HEX_COLOR_RE.match(value))
    )


def load_canvas(path: Path | str) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def find_overlaps(nodes: list[dict[str, Any]]) -> list[tuple[str, str]]:
    """Return pairs of node ids whose boxes strictly intersect."""
    boxes = [
        (n["id"], n["x"], n["y"], n["width"], n["height"])
        for n in nodes
        if isinstance(n, dict)
        and isinstance(n.get("id"), str)
        and all(_is_int(n.get(k)) for k in GEOMETRY_KEYS)
    ]
    boxes.sort(key=lambda b: (b[1], b[2], b[0]))
    pairs: list[tuple[str, str]] = []
    for i, (id_a, ax, ay, aw, ah) in enumerate(boxes):
        for id_b, bx, by, bw, bh in boxes[i + 1 :]:
            if bx >= ax + aw:
                break
            if ax < bx + bw and bx < ax + aw and ay < by + bh and by < ay + ah:
                pairs.append((id_a, id_b))
    return pairs


def verify_canvas(
    canvas: Any, *, require_hex16: bool = True, forbid_overlap: bool = True
) -> VerificationReport:
    """Validate a parsed canvas document and return a report."""
    report = VerificationReport()
    errors = report.errors
    warnings = report.warnings

    if not isinstance(canvas, dict):
        errors.append("Top-level JSON value must be an object")
        return report

    nodes = canvas.get("nodes")
    edges = canvas.get("edges")
    if not isinstance(nodes, list):
        errors.append("'nodes' must be an array")
        nodes = []
    if not isinstance(edges, list):
        errors.append("'edges' must be an array")
        edges = []

    node_ids: set[str] = set()
    type_counts: Counter[str] = Counter()
    color_counts: Counter[str] = Counter()

    for index, node in enumerate(nodes):
        where = f"nodes[{index}]"
        if not isinstance(node, dict):
            errors.append(f"{where}: node must be an object")
            continue
        node_id = node.get("id")
        if not isinstance(node_id, str) or not node_id:
            errors.append(f"{where}: missing string 'id'")
        else:
            if node_id in node_ids:
                errors.append(f"{where}: duplicate node id {node_id!r}")
            node_ids.add(node_id)
            if require_hex16 and not HEX16_RE.match(node_id):
                errors.append(f"{where}: id {node_id!r} is not 16 lower-case hex chars")
        node_type = node.get("type")
        if node_type not in NODE_TYPES:
            errors.append(f"{where}: invalid type {node_type!r}")
        else:
            type_counts[node_type] += 1
        for key in GEOMETRY_KEYS:
            if not _is_int(node.get(key)):
                errors.append(
                    f"{where}: '{key}' must be an integer (got {node.get(key)!r})"
                )
        if _is_int(node.get("width")) and node["width"] <= 0:
            errors.append(f"{where}: width must be positive")
        if _is_int(node.get("height")) and node["height"] <= 0:
            errors.append(f"{where}: height must be positive")
        if node_type == "text" and not isinstance(node.get("text"), str):
            errors.append(f"{where}: text node requires string 'text'")
        if node_type == "file" and not isinstance(node.get("file"), str):
            errors.append(f"{where}: file node requires string 'file'")
        if node_type == "link" and not isinstance(node.get("url"), str):
            errors.append(f"{where}: link node requires string 'url'")
        color = node.get("color")
        if color is not None:
            if _valid_color(color):
                color_counts[color] += 1
            else:
                errors.append(f"{where}: invalid color {color!r}")

    edge_ids: set[str] = set()
    for index, edge in enumerate(edges):
        where = f"edges[{index}]"
        if not isinstance(edge, dict):
            errors.append(f"{where}: edge must be an object")
            continue
        edge_id = edge.get("id")
        if not isinstance(edge_id, str) or not edge_id:
            errors.append(f"{where}: missing string 'id'")
        else:
            if edge_id in edge_ids or edge_id in node_ids:
                errors.append(f"{where}: duplicate id {edge_id!r}")
            edge_ids.add(edge_id)
            if require_hex16 and not HEX16_RE.match(edge_id):
                errors.append(f"{where}: id {edge_id!r} is not 16 lower-case hex chars")
        for key in ("fromNode", "toNode"):
            ref = edge.get(key)
            if not isinstance(ref, str):
                errors.append(f"{where}: '{key}' must be a string")
            elif ref not in node_ids:
                errors.append(f"{where}: '{key}' references unknown node {ref!r}")
        for key in ("fromSide", "toSide"):
            if key in edge and edge[key] not in SIDES:
                errors.append(f"{where}: invalid {key} {edge[key]!r}")
        for key in ("fromEnd", "toEnd"):
            if key in edge and edge[key] not in ENDS:
                errors.append(f"{where}: invalid {key} {edge[key]!r}")
        if "color" in edge and not _valid_color(edge["color"]):
            errors.append(f"{where}: invalid color {edge['color']!r}")
        if "label" in edge and not isinstance(edge["label"], str):
            errors.append(f"{where}: 'label' must be a string")

    overlaps = find_overlaps([n for n in nodes if isinstance(n, dict)])
    if overlaps:
        message = f"{len(overlaps)} overlapping node pair(s)"
        sink = errors if forbid_overlap else warnings
        sink.append(message)
        for id_a, id_b in overlaps[:MAX_REPORTED_OVERLAPS]:
            sink.append(f"  overlap: {id_a} <-> {id_b}")

    report.stats = {
        "nodes": len(nodes),
        "edges": len(edges),
        "node_types": dict(sorted(type_counts.items())),
        "node_colors": dict(sorted(color_counts.items())),
        "overlapping_pairs": len(overlaps),
    }
    if not nodes:
        warnings.append("canvas has no nodes")
    if not edges:
        warnings.append("canvas has no edges")
    return report


def verify_file(path: Path | str, **options: Any) -> VerificationReport:
    logger.info("Verifying JSON Canvas %s", path)
    try:
        canvas = load_canvas(path)
    except (OSError, json.JSONDecodeError) as exc:
        report = VerificationReport()
        report.errors.append(f"cannot load {path}: {exc}")
        return report
    report = verify_canvas(canvas, **options)
    (logger.info if report.ok else logger.error)(
        "Integrity %s: %d error(s), %d warning(s)",
        "PASS" if report.ok else "FAIL",
        len(report.errors),
        len(report.warnings),
    )
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Verify a JSON Canvas file.")
    parser.add_argument("canvas", help="Path to the .canvas file")
    parser.add_argument(
        "--allow-overlap", action="store_true", help="Report overlaps as warnings"
    )
    parser.add_argument(
        "--no-hex-check", action="store_true", help="Skip the 16-hex id rule"
    )
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)

    setup_logger("lightrag", level=args.log_level, enable_file_logging=False)
    report = verify_file(
        args.canvas,
        require_hex16=not args.no_hex_check,
        forbid_overlap=not args.allow_overlap,
    )
    print(report.render())
    return 0 if report.ok else 1


if __name__ == "__main__":
    sys.exit(main())
