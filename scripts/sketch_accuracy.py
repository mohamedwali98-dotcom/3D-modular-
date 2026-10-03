"""Accuracy of the sketch reader on the golden sheets. Prints a markdown table for the README.
Usage: uv run python scripts/sketch_accuracy.py [folder ...]   (default: every tests/golden_sketch/*)"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

from s2c.config import load_env
from s2c.sketch.pipeline import analyse


def TOL(v: float) -> float:
    return max(1.0, 0.05 * abs(v))


TRUSTED = ("written", "derived", "edited")


def score(reading, trace, expected: dict) -> dict:
    s = {"values": 0, "read_ok": 0, "link_ok": 0, "silent_errors": 0, "edits": 0,
         "envelope_ok": 0, "holes_ok": 0, "holes": len(expected.get("holes", []))}
    got = [d for d in reading.dimensions if d.readings]
    pool = list(got)
    link_by_id = {L.text.id: L for L in (trace.links or [])}
    for e in expected.get("dimensions", []):
        s["values"] += 1
        match = next((d for d in pool if d.view == e["view"] and d.kind == e["kind"]
                      and d.value is not None and abs(d.value - e["value"]) < 1e-6), None)
        if match is None:
            s["edits"] += 1
            continue
        pool.remove(match)
        s["read_ok"] += 1
        L = link_by_id.get(match.id)
        if "span_mm" in e and L is not None and L.refs and trace.solved:
            span = sorted(trace.solved.mm[r] for r in L.refs)
            if all(abs(a - b) <= 0.5 for a, b in zip(span, e["span_mm"])):
                s["link_ok"] += 1
        elif "span_mm" not in e and match.measures:
            s["link_ok"] += 1
        if match.badge not in TRUSTED:
            s["edits"] += 1
    for d in pool:  # values we produced that match nothing expected
        if d.badge in TRUSTED:
            s["silent_errors"] += 1
        s["edits"] += 1
    env = expected.get("envelope", {})
    ok = 0
    for axis, value in env.items():
        size = reading.envelope.get(axis)
        if size is None:
            continue
        right = abs(size.value - value) <= TOL(value)
        ok += right
        if not right and size.badge in TRUSTED:
            s["silent_errors"] += 1
    s["envelope_ok"] = int(ok == len(env))
    holes = [f for f in reading.features if f.type == "hole"]
    for h in expected.get("holes", []):
        idx = {"x": 0, "y": 1, "z": 2}
        across = [k for k in ("x", "y", "z") if k != h["axis"]]
        hit = next((f for f in holes if f.axis == h["axis"] and f.through == h["through"]
                    and abs((f.diameter or 0) - h["diameter"]) <= TOL(h["diameter"])
                    and all(abs(f.position_mm[idx[a]] - v) <= TOL(v) for a, v in zip(across, h["at"]))),
                   None)
        if hit:
            holes.remove(hit)
            s["holes_ok"] += 1
    return s


def main(folders: list[Path]) -> None:
    load_env()
    rows, total = [], {}
    for folder in folders:
        image = next((p for p in folder.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png")), None)
        if image is None:
            continue
        expected = json.loads((folder / "expected.json").read_text(encoding="utf-8"))
        t0 = time.perf_counter()
        reading, trace = analyse(image.read_bytes())
        s = score(reading, trace, expected)
        s["ms"] = (time.perf_counter() - t0) * 1000
        rows.append((folder.name, s))
        for k, v in s.items():
            total[k] = total.get(k, 0) + v
    print("| Sheet | Values read | Linked | Envelope | Holes | Edits | Silent errors | ms |")
    print("| --- | --- | --- | --- | --- | --- | --- | --- |")
    for name, s in rows:
        print(f"| {name} | {s['read_ok']}/{s['values']} | {s['link_ok']}/{s['values']} | "
              f"{'ok' if s['envelope_ok'] else 'no'} | {s['holes_ok']}/{s['holes']} | {s['edits']} | "
              f"{s['silent_errors']} | {s['ms']:.0f} |")
    if rows:
        n = len(rows)
        print(f"| **all** | {total['read_ok']}/{total['values']} | {total['link_ok']}/{total['values']} | "
              f"{total['envelope_ok']}/{n} | {total['holes_ok']}/{total['holes']} | {total['edits'] / n:.1f} per sheet | "
              f"{total['silent_errors']} | {total['ms'] / n:.0f} |")


if __name__ == "__main__":
    args = [Path(a) for a in sys.argv[1:]]
    main(args or sorted(p for p in Path("tests/golden_sketch").iterdir() if (p / "expected.json").exists()))
