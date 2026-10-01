"""Benchmark module and CLI against a synthetic reverse-engineering dataset, plus a real-dataset smoke test."""
import json
import os
import time
from pathlib import Path

import cadquery as cq
import numpy as np
import pytest
import trimesh
from PIL import Image

from s2c.multiview import spec as S
from s2c.multiview.benchmark import RefPart, load_part, part_sheet, score_part, summarize, summary_markdown
from s2c.multiview.pipeline import MvPipeline
from s2c.multiview.raster import Mesh, face_mask, solid_mesh
from s2c.multiview.sheet import Naming, Sheet, split_sheet

RE_DATASET = os.environ.get("RE_DATASET")


def _box_with_hole(x: float, y: float, z: float, dia: float = 6.0) -> cq.Workplane:
    """A box with one through hole along its z (depth) axis, our-frame convention."""
    box = cq.Workplane("XY").box(x, y, z, centered=False)
    return box.faces(">Z").workplane().circle(dia / 2).cutThruAll()


def _face_png(mesh: Mesh, face: str, env: S.Envelope, px: int = 800) -> Image.Image:
    mask, _ = face_mask(mesh, face, env, px=px)
    img = np.full((px, px, 3), 255, np.uint8)
    img[mask > 127] = 128
    return Image.fromarray(img, "RGB")


def make_part(tmp: Path, name: str, solid: cq.Workplane, flip_winding: bool = False) -> Path:
    """Write one dataset-shaped part (model.stl, metadata.json, six PNGs) under tmp/<name>."""
    category, dirname = name.split("/")
    part_dir = tmp / category / dirname
    part_dir.mkdir(parents=True, exist_ok=True)
    mesh = solid_mesh(solid)
    v = mesh.vertices - mesh.vertices.min(axis=0)
    env = S.Envelope(x_mm=float(v[:, 0].max()), y_mm=float(v[:, 1].max()), z_mm=float(v[:, 2].max()))
    # dataset frame is the inverse of the load-side rotation: ds = (x, -z_ours, y_ours)
    v_ds = np.stack([v[:, 0], -v[:, 2], v[:, 1]], axis=1)
    faces = mesh.faces[:, ::-1] if flip_winding else mesh.faces
    trimesh.Trimesh(v_ds, faces, process=False).export(part_dir / "model.stl")
    (part_dir / "metadata.json").write_text(json.dumps({
        "bounds_mm": {"x": float(np.ptp(v_ds[:, 0])), "y": float(np.ptp(v_ds[:, 1])), "z": float(np.ptp(v_ds[:, 2]))},
        "watertight": True, "license": "test",
    }))
    ours_mesh = Mesh(v, mesh.faces)  # renders always drawn from the correctly-wound ours-frame mesh
    for face in S.FACES:
        _face_png(ours_mesh, face, env).save(part_dir / f"{face}.png")
    return part_dir


def test_load_part_maps_the_frame(tmp_path):
    part_dir = make_part(tmp_path, "boxes/box_001", _box_with_hole(40, 20, 10))
    ref = load_part(part_dir)
    assert ref.name == "boxes/box_001"
    assert ref.category == "boxes"
    assert ref.envelope.x_mm == pytest.approx(40, abs=0.01)
    assert ref.envelope.y_mm == pytest.approx(20, abs=0.01)
    assert ref.envelope.z_mm == pytest.approx(10, abs=0.01)
    truth = trimesh.Trimesh(ref.mesh.vertices, ref.mesh.faces, process=False)
    assert truth.volume > 0


def test_score_part_builds_a_box_with_a_hole(tmp_path):
    part_dir = make_part(tmp_path, "boxes/box_001", _box_with_hole(40, 20, 10))
    ref = load_part(part_dir)
    row = score_part(ref, MvPipeline(), tmp_path / "out")
    assert row["result"] == "built", row["result"]
    assert abs(row["vol_err"]) < 0.06
    assert row["voxel_iou"] > 0.9
    assert row["frame_iou"] > 0.95


def test_flipped_stl_winding_still_scores_built_with_positive_volume(tmp_path):
    """Some real STLs have inward-facing normals; a negative raw volume must not leak through as vol_true."""
    part_dir = make_part(tmp_path, "boxes/box_003", _box_with_hole(40, 20, 10), flip_winding=True)
    ref = load_part(part_dir)
    truth = trimesh.Trimesh(ref.mesh.vertices, ref.mesh.faces, process=False)
    assert truth.volume < 0  # the flip is real: the raw signed volume is negative
    row = score_part(ref, MvPipeline(), tmp_path / "out")
    assert row["result"] == "built", row["result"]
    assert row["vol_true"] > 0
    assert abs(row["vol_err"]) < 0.06


def test_score_part_timeout_frees_the_worker(tmp_path, monkeypatch):
    """A real hang must return once timeout_s elapses, not once the hung work finally finishes."""
    ref = RefPart("boxes/box_009", "boxes", None, None, {}, True, 10)

    def hang(*args, **kwargs):
        time.sleep(3)
        return {}

    monkeypatch.setattr("s2c.multiview.benchmark._run", hang)
    t0 = time.time()
    row = score_part(ref, MvPipeline(), tmp_path / "out", timeout_s=0.5)
    elapsed = time.time() - t0
    assert row["result"] == "skipped timeout"
    assert elapsed < 1.5, elapsed


def test_not_watertight_is_skipped(tmp_path):
    part_dir = make_part(tmp_path, "boxes/box_002", _box_with_hole(40, 20, 10))
    meta_path = part_dir / "metadata.json"
    meta = json.loads(meta_path.read_text())
    meta["watertight"] = False
    meta_path.write_text(json.dumps(meta))
    ref = load_part(part_dir)
    assert ref.watertight is False
    row = score_part(ref, MvPipeline(), tmp_path / "out")
    assert row["result"] == "skipped not_watertight"


def test_summary_counts_abstains_and_skips():
    def row(**kw):
        base = {"part": None, "category": "a", "frame_iou": None, "vol_true": None, "vol_built": None,
                "vol_err": None, "voxel_iou": None, "view_iou": None, "features": None, "secs": 1.0}
        base.update(kw)
        return base

    rows = [
        row(part="a/1", result="built", vol_err=0.02, voxel_iou=0.95,
            view_iou={f: 0.9 for f in S.FACES}),
        row(part="a/2", result="built", vol_err=-0.08, voxel_iou=0.80,
            view_iou={f: 0.7 for f in S.FACES}),
        row(part="a/3", result="abstain outline:no_outline"),
        row(part="a/4", result="skipped not_watertight"),
        row(part="a/5", result="skipped too_large"),
        row(part="a/6", result="error ValueError"),
    ]
    summary = summarize(rows)
    overall = summary["overall"]
    assert overall["n"] == 6
    assert overall["built"] == 2
    assert overall["abstained"] == {"outline:no_outline": 1}
    assert overall["skipped"] == 2
    assert overall["errors"] == 1
    assert overall["median_abs_vol_err"] == pytest.approx(0.05)
    assert overall["median_voxel_iou"] == pytest.approx(0.875)
    assert overall["within_5pct_vol_err"] == pytest.approx(0.5)
    assert "a" in summary["categories"]
    assert summary["categories"]["a"]["n"] == 6


def test_summary_markdown_states_clean_renders():
    rows = [{"part": "a/1", "category": "a", "result": "built", "vol_err": 0.01, "voxel_iou": 0.9,
            "view_iou": {f: 0.9 for f in S.FACES}, "frame_iou": 0.98, "vol_true": 10.0, "vol_built": 10.1,
            "features": 0, "secs": 1.0}]
    text = summary_markdown(summarize(rows))
    assert "clean renders, true envelope given as user values" in text
    assert "overall" in text
    assert "named correctly" not in text


@pytest.mark.parametrize("name", ["boxes/box_001", "boxes/box_002"], ids=["unlabelled", "labelled"])
def test_sheet_mode_scores_a_synthetic_part(tmp_path, name):
    """The front, top and right renders as one first-angle line-art sheet split, name and build like per-face
    uploads. Even-numbered parts carry view labels, which must not disturb the split."""
    ref = load_part(make_part(tmp_path, name, _box_with_hole(40, 20, 10)))
    row = score_part(ref, MvPipeline(), tmp_path / "out", sheet=True)
    assert row["result"] == "built", row["result"]
    assert row["named_ok"] is True
    assert row["voxel_iou"] > 0.85


def test_even_numbered_parts_get_view_labels(tmp_path):
    for name, labelled in (("boxes/box_001", False), ("boxes/box_002", True)):
        image, truth = part_sheet(load_part(make_part(tmp_path, name, _box_with_hole(40, 20, 10))))
        assert set(truth) == {"front", "top", "right"}
        views = [v for d in split_sheet(image).drawings for v in d.views]
        assert len(views) == 3, name
        assert all((v.label_box is not None) == labelled for v in views), name


@pytest.mark.parametrize("stage", ["split", "naming"])
def test_a_sheet_without_three_named_views_abstains(tmp_path, monkeypatch, stage):
    ref = load_part(make_part(tmp_path, "boxes/box_001", _box_with_hole(40, 20, 10)))
    if stage == "split":
        monkeypatch.setattr("s2c.multiview.benchmark.split_sheet", lambda image: Sheet([], image.shape[:2]))
    else:
        monkeypatch.setattr("s2c.multiview.benchmark.choose_naming",
                            lambda sheet, image, projection: Naming(0, ["front", "auto", "auto"], "first", "setting"))
    row = score_part(ref, MvPipeline(), tmp_path / "out", sheet=True)
    assert row["result"] == f"abstain sheet:{stage}"
    assert row["named_ok"] is False


def test_sheet_summary_reports_named_correctly():
    def row(result, named_ok, **kw):
        base = {"part": "a/1", "category": "a", "result": result, "named_ok": named_ok, "frame_iou": None,
                "vol_true": None, "vol_built": None, "vol_err": None, "voxel_iou": None, "view_iou": None,
                "features": None, "secs": 1.0}
        base.update(kw)
        return base

    rows = [row("built", True, vol_err=0.01, voxel_iou=0.9, view_iou={f: 0.9 for f in S.FACES}),
            row("built", False, vol_err=0.2, voxel_iou=0.5, view_iou={f: 0.5 for f in S.FACES}),
            row("abstain sheet:split", False),
            row("skipped timeout", None)]
    summary = summarize(rows)
    assert summary["overall"]["named_ok"] == pytest.approx(1 / 3)
    assert summary["overall"]["abstained"] == {"sheet:split": 1}
    text = summary_markdown(summary)
    assert "sheet" in text.splitlines()[0]
    assert "named correctly" in text
    assert "33.3%" in text

def test_cli_writes_results_and_summary(tmp_path):
    from scripts import re_benchmark

    dataset = tmp_path / "dataset"
    make_part(dataset, "boxes/box_001", _box_with_hole(40, 20, 10))
    make_part(dataset, "boxes/box_002", _box_with_hole(30, 30, 8))
    out = tmp_path / "out"
    re_benchmark.main(["--dataset", str(dataset), "--out", str(out), "--jobs", "1"])
    lines = (out / "results.jsonl").read_text().strip().splitlines()
    assert len(lines) == 2
    parts = {json.loads(line)["part"] for line in lines}
    assert parts == {"boxes/box_001", "boxes/box_002"}
    summary_text = (out / "summary.md").read_text()
    assert "clean renders" in summary_text
    assert json.loads((out / "summary.json").read_text())["overall"]["n"] == 2


def test_cli_isolates_a_broken_part_and_still_scores_the_other(tmp_path):
    """A bad metadata.json must not poison the run: it becomes one error row, the other part still builds."""
    from scripts import re_benchmark

    dataset = tmp_path / "dataset"
    make_part(dataset, "boxes/box_001", _box_with_hole(40, 20, 10))
    broken_dir = make_part(dataset, "boxes/box_002", _box_with_hole(30, 30, 8))
    (broken_dir / "metadata.json").write_text("{not valid json")
    out = tmp_path / "out"
    re_benchmark.main(["--dataset", str(dataset), "--out", str(out), "--jobs", "2"])
    rows = {json.loads(line)["part"]: json.loads(line)
            for line in (out / "results.jsonl").read_text().strip().splitlines()}
    assert rows["boxes/box_001"]["result"] == "built"
    assert rows["boxes/box_002"]["result"].startswith("error ")


def test_cli_with_two_jobs_writes_two_rows_and_both_summaries(tmp_path):
    from scripts import re_benchmark

    dataset = tmp_path / "dataset"
    make_part(dataset, "boxes/box_001", _box_with_hole(40, 20, 10))
    make_part(dataset, "boxes/box_002", _box_with_hole(30, 30, 8))
    out = tmp_path / "out"
    re_benchmark.main(["--dataset", str(dataset), "--out", str(out), "--jobs", "2"])
    lines = (out / "results.jsonl").read_text().strip().splitlines()
    assert len(lines) == 2
    assert (out / "summary.md").exists()
    assert (out / "summary.json").exists()


def test_cli_sheet_mode_writes_named_ok(tmp_path):
    from scripts import re_benchmark

    dataset = tmp_path / "dataset"
    make_part(dataset, "boxes/box_002", _box_with_hole(40, 20, 10))
    out = tmp_path / "out"
    re_benchmark.main(["--dataset", str(dataset), "--out", str(out), "--jobs", "1", "--sheet"])
    rows = [json.loads(line) for line in (out / "results.jsonl").read_text().strip().splitlines()]
    assert len(rows) == 1
    assert rows[0]["result"] == "built", rows[0]["result"]
    assert rows[0]["named_ok"] is True
    assert "named correctly" in (out / "summary.md").read_text()


def test_per_category_takes_parts_round_robin(tmp_path):
    from scripts import re_benchmark

    for name in ("a/a_001", "a/a_002", "a/a_003", "b/b_001", "c/c_001", "c/c_002"):
        (tmp_path / name).mkdir(parents=True)
        (tmp_path / name / "model.stl").write_text("")
    picked = re_benchmark._find_parts(tmp_path, None, per_category=2)
    assert [f"{p.parent.name}/{p.name}" for p in picked] == ["a/a_001", "b/b_001", "c/c_001", "a/a_002", "c/c_002"]

@pytest.mark.dataset
@pytest.mark.skipif(not RE_DATASET, reason="set RE_DATASET to run against the real dataset")
def test_real_dataset_smoke():
    root = Path(RE_DATASET)
    pipe = MvPipeline()
    for rel in ("bracket/bracket_001", "washer_spacer/washer_spacer_001"):
        ref = load_part(root / rel)
        row = score_part(ref, pipe, Path("tmp/re_bench_smoke"))
        assert row["result"] == "built", (rel, row["result"])
        assert row["voxel_iou"] > 0.85
