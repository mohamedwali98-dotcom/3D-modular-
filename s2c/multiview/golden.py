"""The golden set: real photos and drawings with their known truth, read by the shipped path (read_drawing, then
the pipeline exactly as the web job runs it), scored, and held against a committed baseline (audit C3).

One folder per sample under tests/golden_sketch: `image.jpg` (or `.png`) and `expected.json` with the views, the
written dimensions, and optionally the projection the sheet is drawn in, where each view is (`layout`: its centre
as fractions of the image) and the part's true envelope. What is counted, pooled over every sample:
- faces: the views named right, with the sheet's projection given; by place when the layout is known, so swapped
  names count as wrong; faces_auto: the same on the web app's default projection ("auto");
- values: the written numbers read exactly (number and kind, each once); misreads: numbers read that match none;
- sizes: the envelope sizes the reader wrote in for the user that are right; a wrong one is a silent error, the
  worst outcome, counted on its own; a size left for the user to type is neither;
- builds: the part builds (as /api/model builds it) once the user types what was not read (the true size, or the
  suggestion)."""
from __future__ import annotations

import json
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from s2c.multiview import spec as S
from s2c.multiview.sheet_read import observe_drawing, read_drawing

TOLERANCE = 0.02           # a metric may fall this much below its baseline before the gate fails
MACHINE = ("user_written", "measured")  # sizes the app filled in itself, from the reader or the drawing's scale
METRICS = ("faces", "faces_auto", "values", "sizes", "builds")
COUNTS = (("silent_errors", "silent errors"), ("misreads", "misreads"))  # counts that may not rise


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


def misreads(wanted: list[dict], got: list[tuple[float, str]]) -> int:
    """Numbers read (a value came out) that match no expected one."""
    return sum(1 for g in got if g[0] is not None) - matched_values(wanted, got)


def named_by_position(named: list[tuple[str, tuple[float, float]]], layout: dict) -> int:
    """Named views whose nearest place in the truth's layout carries that name."""
    places = {face: np.asarray(at, float) for face, at in layout.items()}
    return sum(1 for face, at in named
               if min(places, key=lambda f: float(np.linalg.norm(places[f] - np.asarray(at, float)))) == face)


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


def _named(read, image: np.ndarray, views: list[str], layout: dict | None) -> int:
    """How many views the read named right: by place when the layout is known, else as a set of names."""
    sheet_views = read.sheet.drawings[read.naming.drawing].views
    to_photo = read.page.to_photo if read.page is not None else np.eye(3)
    h, w = image.shape[:2]
    named = []
    for view, face in zip(sheet_views, read.naming.faces, strict=True):
        if face in S.FACES:
            x, y, bw, bh = view.box
            cx, cy = cv2.perspectiveTransform(np.float32([[[x + bw / 2, y + bh / 2]]]), to_photo)[0, 0]
            named.append((face, (cx / w, cy / h)))
    if layout:
        return named_by_position(named, layout)
    faces = [f for f, _ in named]
    return sum(min(views.count(f), faces.count(f)) for f in set(views))


def _builds(spec: S.MultiViewSpec) -> str:
    """'' when the part builds as /api/model builds it (with the geometry settings), else the remedy."""
    from s2c.multiview.artifacts import build_part
    from s2c.multiview.settings import GeometrySettings
    root = Path(tempfile.mkdtemp(prefix="golden-"))
    try:
        part = build_part(spec, GeometrySettings(), root)
        return part.remedy if isinstance(part, S.MvAbstain) else ""
    finally:
        shutil.rmtree(root, ignore_errors=True)


def score_sample(sample: Sample, pipe) -> dict:
    truth = sample.expected
    views, wanted, true_env = truth.get("views", []), truth.get("dimensions", []), truth.get("envelope", {})
    projection, layout = truth.get("projection", "auto"), truth.get("layout")
    row = {"name": sample.name, "views": len(views), "named": 0, "named_auto": 0, "values": len(wanted),
           "matched": 0, "misread": 0, "sizes": len(true_env), "correct": 0, "wrong": 0, "asked": 0, "built": False,
           "note": ""}
    image = cv2.imread(str(sample.image))
    read = read_drawing(image, projection, reader=pipe.reader, service=pipe.reading())
    if read is None or isinstance(read, S.MvAbstain):
        row["note"] = "no sheet of views found" if read is None else read.remedy
        return row
    row["named"] = _named(read, image, views, layout)
    if projection == "auto":
        row["named_auto"] = row["named"]
    else:
        auto = read_drawing(image, "auto", reader=pipe.reader, service=pipe.reading())
        row["named_auto"] = 0 if auto is None or isinstance(auto, S.MvAbstain) else _named(auto, image, views, layout)
    got = [(d.value_mm, d.kind) for d in read.scale.dimensions]
    row["matched"], row["misread"] = matched_values(wanted, got), misreads(wanted, got)
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
    row["note"] = _builds(spec)
    row["built"] = not row["note"]
    return row


def _ratio(part: float, whole: float) -> float:
    return round(part / whole, 3) if whole else 1.0  # nothing to judge counts as nothing lost


def summarize(rows: list[dict]) -> dict:
    total = {k: sum(r[k] for r in rows)
             for k in ("views", "named", "named_auto", "values", "matched", "misread", "sizes", "correct", "wrong")}
    return {"samples": len(rows), "faces": _ratio(total["named"], total["views"]),
            "faces_auto": _ratio(total["named_auto"], total["views"]),
            "values": _ratio(total["matched"], total["values"]), "misreads": total["misread"],
            "sizes": _ratio(total["correct"], total["sizes"]), "silent_errors": total["wrong"],
            "builds": _ratio(sum(r["built"] for r in rows), len(rows))}


def regressions(summary: dict, baseline: dict, tol: float = TOLERANCE) -> list[str]:
    if summary["samples"] != baseline.get("samples"):
        return [(f"the set has {summary['samples']} samples and the baseline {baseline.get('samples')}: record a new "
                 "one with --write-baseline")]
    out = [f"{k} fell from {baseline[k]:.2f} to {summary[k]:.2f}" for k in METRICS
           if k in baseline and summary[k] < baseline[k] - tol - 1e-9]
    out += [f"{name} rose from {baseline[k]} to {summary[k]}" for k, name in COUNTS
            if k in baseline and summary[k] > baseline[k]]
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
