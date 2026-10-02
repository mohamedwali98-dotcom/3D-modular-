"""The golden set: real photos and drawings with their known truth, read by the shipped path (read_drawing, then
the pipeline exactly as the web job runs it), scored, and held against a committed baseline (audit C3).

One folder per sample under tests/golden_sketch: `image.jpg` (or `.png`) and `expected.json` with the views, the
written dimensions, and optionally the projection the sheet is drawn in and the part's true envelope. What is
counted, pooled over every sample:
- faces: the expected views named as such;
- values: the written numbers read exactly (number and kind, each once);
- sizes: the envelope sizes the reader wrote in for the user that are right; a wrong one is a silent error, the
  worst outcome, counted on its own; a size left for the user to type is neither;
- builds: the part builds once the user types what was not read (the true size, or the suggestion)."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2

from s2c.multiview import spec as S
from s2c.multiview.build import BuildError, build
from s2c.multiview.sheet_read import observe_drawing, read_drawing

TOLERANCE = 0.02           # a metric may fall this much below its baseline before the gate fails
MACHINE = ("user_written", "measured")  # sizes the app filled in itself, from the reader or the drawing's scale
METRICS = ("faces", "values", "sizes", "builds")


@dataclass
class Sample:
    name: str
    image: Path
    expected: dict


def samples(root: Path) -> list[Sample]:
    """Every folder holding an image and its truth. A folder with only expected.json is drawn by its own test."""
    out = []
    for folder in sorted(p for p in Path(root).iterdir() if p.is_dir()):
        image = next((folder / n for n in ("image.jpg", "image.png") if (folder / n).is_file()), None)
        truth = folder / "expected.json"
        if image is not None and truth.is_file():
            out.append(Sample(folder.name, image, json.loads(truth.read_text(encoding="utf-8"))))
    return out


def matched_values(wanted: list[dict], got: list[tuple[float, str]]) -> int:
    """Expected numbers read exactly, by value and kind, each read value used once."""
    pool = list(got)
    hits = 0
    for w in wanted:
        hit = next((g for g in pool if g[1] == w["kind"] and g[0] is not None and abs(g[0] - w["value"]) < 1e-6),
                   None)
        if hit is not None:
            pool.remove(hit)
            hits += 1
    return hits


def size_marks(truth: dict, known: dict, prov: dict) -> dict:
    """Each true envelope size: right or silently wrong when the app filled it in, asked when the user types it."""
    marks = {"correct": 0, "wrong": 0, "asked": 0}
    for axis, true in truth.items():
        if prov.get(axis) not in MACHINE:
            marks["asked"] += 1
        elif abs(known[axis] - true) <= max(1.0, 0.05 * true):
            marks["correct"] += 1
        else:
            marks["wrong"] += 1
    return marks


def _envelope(res) -> tuple[dict, dict, dict]:
    """(known sizes, their provenance, suggestions) by axis, from a spec or a fuse abstention."""
    if isinstance(res, S.MvAbstain):
        partial = res.partial if isinstance(res.partial, dict) else {}

        def by_axis(d):
            return {k.split(".")[1][0]: v for k, v in (d or {}).items() if k.startswith("envelope.")}
        return by_axis(partial.get("known")), by_axis(partial.get("provenance")), by_axis(partial.get("suggested"))
    return ({a: getattr(res.envelope, f"{a}_mm") for a in "xyz"},
            {a: res.provenance[f"envelope.{a}_mm"] for a in "xyz"}, {})


def score_sample(sample: Sample, pipe) -> dict:
    truth = sample.expected
    views, wanted, true_env = truth.get("views", []), truth.get("dimensions", []), truth.get("envelope", {})
    row = {"name": sample.name, "views": len(views), "named": 0, "values": len(wanted), "matched": 0,
           "sizes": len(true_env), "correct": 0, "wrong": 0, "asked": 0, "built": False, "note": ""}
    read = read_drawing(cv2.imread(str(sample.image)), truth.get("projection", "auto"), reader=pipe.reader,
                        service=pipe.reading())
    if read is None or isinstance(read, S.MvAbstain):
        row["note"] = "no sheet of views found" if read is None else read.remedy
        return row
    named = [f for f in read.naming.faces if f in S.FACES]
    row["named"] = sum(min(views.count(f), named.count(f)) for f in set(views))
    row["matched"] = matched_values(wanted, [(d.value_mm, d.kind) for d in read.scale.dimensions])
    observed = observe_drawing(read, pipe)
    if isinstance(observed, S.MvAbstain):
        row["note"] = observed.remedy
        return row
    known, prov, suggested = _envelope(pipe.fuse(observed))
    row.update(size_marks(true_env, known, prov))
    typed = {f"envelope.{a}_mm": float(true_env.get(a) or suggested.get(a) or 10.0)
             for a in "xyz" if prov.get(a) not in MACHINE}
    spec = pipe.fuse(observed, typed)
    if isinstance(spec, S.MvAbstain):
        row["note"] = spec.remedy
        return row
    try:
        build(spec)
        row["built"] = True
    except BuildError as e:
        row["note"] = e.remedy
    return row


def _ratio(part: float, whole: float) -> float:
    return round(part / whole, 3) if whole else 1.0  # nothing to judge counts as nothing lost


def summarize(rows: list[dict]) -> dict:
    total = {k: sum(r[k] for r in rows) for k in ("views", "named", "values", "matched", "sizes", "correct", "wrong")}
    return {"samples": len(rows), "faces": _ratio(total["named"], total["views"]),
            "values": _ratio(total["matched"], total["values"]), "sizes": _ratio(total["correct"], total["sizes"]),
            "silent_errors": total["wrong"], "builds": _ratio(sum(r["built"] for r in rows), len(rows))}


def regressions(summary: dict, baseline: dict, tol: float = TOLERANCE) -> list[str]:
    out = [f"{k} fell from {baseline[k]:.2f} to {summary[k]:.2f}" for k in METRICS
           if summary[k] < baseline[k] - tol - 1e-9]
    if summary["silent_errors"] > baseline["silent_errors"]:
        out.append(f"silent errors rose from {baseline['silent_errors']} to {summary['silent_errors']}")
    return out


def reader_ready(cache_dir=None) -> bool:
    """Whether TrOCR's weights are in the Hugging Face cache, so a run offline (the tests) can read at all."""
    import os

    from s2c.reading.trocr import DEFAULT_MODEL
    try:
        from huggingface_hub import try_to_load_from_cache
    except ImportError:
        return False
    found = try_to_load_from_cache(os.environ.get("TROCR_MODEL") or DEFAULT_MODEL, "config.json", cache_dir=cache_dir)
    return isinstance(found, str)


def trocr_pipeline():
    """The readers the gate measures: TrOCR alone, no hosted model, so a run is the same on every machine."""
    from s2c.multiview.pipeline import MvPipeline
    from s2c.reading.trocr import TrocrReader
    return MvPipeline(reader=TrocrReader())


def evaluate(root: Path, pipe=None) -> dict:
    pipe = pipe or trocr_pipeline()
    rows = [score_sample(s, pipe) for s in samples(root)]
    return {"rows": rows, "summary": summarize(rows)}
