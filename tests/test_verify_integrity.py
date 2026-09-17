"""Tests for the standalone JSON Canvas integrity verifier."""

from __future__ import annotations

import copy
import json
from pathlib import Path

from verify_integrity import main, verify_canvas, verify_file

NODE_A = "a1b2c3d4e5f60718"
NODE_B = "0123456789abcdef"
EDGE_1 = "fedcba9876543210"


def valid_canvas() -> dict:
    return {
        "nodes": [
            {
                "id": NODE_A,
                "type": "text",
                "x": 0,
                "y": 0,
                "width": 400,
                "height": 200,
                "text": "# A",
                "color": "6",
            },
            {
                "id": NODE_B,
                "type": "text",
                "x": 600,
                "y": 0,
                "width": 400,
                "height": 200,
                "text": "# B",
                "color": "#00FF00",
            },
        ],
        "edges": [
            {
                "id": EDGE_1,
                "fromNode": NODE_A,
                "toNode": NODE_B,
                "fromEnd": "none",
                "toEnd": "arrow",
                "fromSide": "right",
                "toSide": "left",
                "color": "4",
                "label": "cites",
            },
        ],
    }


def test_valid_canvas_passes() -> None:
    report = verify_canvas(valid_canvas())
    assert report.ok, report.render()
    assert report.stats["nodes"] == 2 and report.stats["edges"] == 1
    assert report.stats["overlapping_pairs"] == 0


def test_dangling_edge_fails() -> None:
    canvas = valid_canvas()
    canvas["edges"][0]["toNode"] = "deadbeefdeadbeef"
    report = verify_canvas(canvas)
    assert not report.ok
    assert any("unknown node" in e for e in report.errors)


def test_float_and_bool_coordinates_fail() -> None:
    canvas = valid_canvas()
    canvas["nodes"][0]["x"] = 1.5
    canvas["nodes"][1]["y"] = True
    report = verify_canvas(canvas)
    assert sum("must be an integer" in e for e in report.errors) == 2


def test_duplicate_ids_fail() -> None:
    canvas = valid_canvas()
    canvas["nodes"][1]["id"] = NODE_A
    canvas["nodes"][1]["x"] = 1000  # keep boxes apart so only the id error fires
    report = verify_canvas(canvas)
    assert any("duplicate node id" in e for e in report.errors)


def test_invalid_color_and_enums_fail() -> None:
    canvas = valid_canvas()
    canvas["nodes"][0]["color"] = "7"
    canvas["edges"][0]["toEnd"] = "diamond"
    canvas["edges"][0]["fromSide"] = "middle"
    report = verify_canvas(canvas)
    assert any("invalid color" in e for e in report.errors)
    assert any("invalid toEnd" in e for e in report.errors)
    assert any("invalid fromSide" in e for e in report.errors)


def test_overlap_is_error_unless_allowed() -> None:
    canvas = valid_canvas()
    canvas["nodes"][1]["x"] = 100  # overlaps node A
    strict = verify_canvas(canvas)
    assert not strict.ok and strict.stats["overlapping_pairs"] == 1
    relaxed = verify_canvas(canvas, forbid_overlap=False)
    assert relaxed.ok and relaxed.warnings


def test_touching_boxes_do_not_overlap() -> None:
    canvas = valid_canvas()
    canvas["nodes"][1]["x"] = 400  # shares an edge with node A
    assert verify_canvas(canvas).ok


def test_missing_arrays_and_wrong_top_level() -> None:
    report = verify_canvas({})
    assert {"'nodes' must be an array", "'edges' must be an array"} <= set(
        report.errors
    )
    assert not verify_canvas([]).ok


def test_non_hex_ids_fail_unless_disabled() -> None:
    canvas = valid_canvas()
    canvas["nodes"][0]["id"] = "node-1"
    canvas["edges"][0]["fromNode"] = "node-1"
    assert not verify_canvas(canvas).ok
    assert verify_canvas(canvas, require_hex16=False).ok


def test_text_node_requires_text() -> None:
    canvas = valid_canvas()
    del canvas["nodes"][0]["text"]
    assert any("requires string 'text'" in e for e in verify_canvas(canvas).errors)


def test_cli_exit_codes(tmp_path: Path, capsys) -> None:
    good = tmp_path / "good.canvas"
    good.write_text(json.dumps(valid_canvas()), encoding="utf-8")
    assert main([str(good)]) == 0
    assert "PASS" in capsys.readouterr().out

    bad_canvas = copy.deepcopy(valid_canvas())
    bad_canvas["edges"][0]["fromNode"] = "0000000000000000"
    bad = tmp_path / "bad.canvas"
    bad.write_text(json.dumps(bad_canvas), encoding="utf-8")
    assert main([str(bad)]) == 1
    assert "FAIL" in capsys.readouterr().out


def test_verify_file_reports_unreadable(tmp_path: Path) -> None:
    missing = verify_file(tmp_path / "nope.canvas")
    assert not missing.ok and "cannot load" in missing.errors[0]
    broken = tmp_path / "broken.canvas"
    broken.write_text("{not json", encoding="utf-8")
    assert not verify_file(broken).ok
