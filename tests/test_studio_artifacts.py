import json
import os
import shutil
import threading
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from s2c.multiview import artifacts, exporters
from s2c.multiview.artifacts import build_part, bundle, export_part, geometry_key, sweep
from s2c.multiview.settings import GeometrySettings, MeshSettings, PrintSettings
from s2c.multiview.spec import MultiViewSpec, MvAbstain
from tests.mv_helpers import make_spec

SPEC = MultiViewSpec.model_validate_json(
    (Path(__file__).parents[1] / "examples" / "mv" / "l_bracket.json").read_text())


@pytest.fixture(autouse=True)
def no_real_slicer(monkeypatch):
    artifacts.clear_cache()
    monkeypatch.setattr("s2c.multiview.slice.find_slicer", lambda: None)


def test_the_key_ignores_warnings_and_provenance_but_not_geometry():
    other = SPEC.model_copy(update={"warnings": ["something"], "confidence": 0.5})
    assert geometry_key(other) == geometry_key(SPEC)
    assert geometry_key(SPEC, GeometrySettings(finish="fillet")) != geometry_key(SPEC)


def test_a_part_is_built_once(tmp_path, monkeypatch):
    calls = []
    real = artifacts.build
    monkeypatch.setattr(artifacts, "build", lambda spec: calls.append(1) or real(spec))
    first = build_part(SPEC, root=tmp_path)
    second = build_part(SPEC.model_copy(update={"warnings": ["x"]}), root=tmp_path)
    assert second.solid is first.solid and second.folder == first.folder and calls == [1]
    assert first.preview.exists() and first.bbox_mm == pytest.approx((50, 30, 20))
    assert set(first.views) == {"front", "back", "left", "right", "top", "bottom"}


def test_a_cached_part_carries_the_latest_provenance_into_the_manifest(tmp_path):
    first = build_part(SPEC, root=tmp_path)
    edited = SPEC.model_copy(update={"provenance": {**SPEC.provenance, "features[0].a_mm": "user_edited"},
                                     "warnings": [*SPEC.warnings, "x"]})
    second = build_part(edited, root=tmp_path)
    assert second.solid is first.solid
    with zipfile.ZipFile(bundle(second, export_part(second, ["stl"]))) as z:
        manifest = json.loads(z.read("manifest.json"))
    assert manifest["spec"]["provenance"]["features[0].a_mm"] == "user_edited" and "x" in manifest["warnings"]


def test_a_finish_that_cannot_be_built_is_a_card(tmp_path):
    res = build_part(SPEC, GeometrySettings(finish="fillet", finish_mm=8.0), root=tmp_path)
    assert isinstance(res, MvAbstain) and res.stage == "build" and res.reason == "fillet_failed"


def test_new_formats_reuse_the_part_and_its_files(tmp_path):
    part = build_part(SPEC, root=tmp_path)
    first = export_part(part, ["stl", "step"], MeshSettings())
    stamp = first.files["stl"].stat().st_mtime_ns
    second = export_part(part, ["stl", "step", "obj"], MeshSettings(), PrintSettings(infill_pct=50))
    assert second.files["stl"].stat().st_mtime_ns == stamp and second.files["obj"].exists()
    fine = export_part(part, ["stl"], MeshSettings(quality="fine"))
    assert fine.files["stl"] != first.files["stl"] and fine.files["stl"].stat().st_size > first.sizes["stl"]


def test_every_format_can_be_exported(tmp_path, monkeypatch):
    monkeypatch.setattr("s2c.multiview.blend.blender_runner", lambda: None)
    part = build_part(SPEC, root=tmp_path)
    res = export_part(part, ["stl", "step", "3mf", "obj", "glb", "ply", "brep", "blend", "dxf", "svg", "pdf"])
    assert set(res.files) == {"stl", "step", "3mf", "obj", "glb", "ply", "brep", "blend", "dxf", "svg", "pdf"}
    assert res.files["blend"].name == "part_blender_kit.zip" and all(s > 0 for s in res.sizes.values())


def test_gcode_problems_never_block_the_other_files(tmp_path):
    part = build_part(SPEC, root=tmp_path)
    res = export_part(part, ["stl", "gcode"])
    assert set(res.files) == {"stl"} and "G-code unavailable: slicer not installed" in res.warnings
    big = build_part(make_spec((150.0, 20.0, 10.0)), root=tmp_path)  # 300 mm long at 200 %: larger than the bed
    res = export_part(big, ["step", "gcode"], printing=PrintSettings(scale_pct=200))
    assert set(res.files) == {"step"} and any("larger than the printer bed" in w for w in res.warnings)


def test_the_zip_holds_the_files_and_a_manifest(tmp_path):
    part = build_part(SPEC, root=tmp_path)
    res = export_part(part, ["stl", "step"])
    path = bundle(part, res, {"mesh": {"quality": "normal"}})
    with zipfile.ZipFile(path) as z:
        assert sorted(z.namelist()) == ["manifest.json", "part.step", "part.stl"]
        manifest = json.loads(z.read("manifest.json"))
    assert manifest["spec"]["provenance"]["envelope.x_mm"] == "user_written"
    assert manifest["settings"] == {"mesh": {"quality": "normal"}} and manifest["part"]["key"] == part.key


def test_each_export_gets_its_own_zip(tmp_path):
    part = build_part(SPEC, root=tmp_path)
    first = bundle(part, export_part(part, ["stl"]), {"formats": ["stl"]})
    second = bundle(part, export_part(part, ["step"]), {"formats": ["step"]})
    assert first != second and first.name.startswith(f"sketch-to-cad-{part.key[:8]}-")
    with zipfile.ZipFile(first) as z:
        assert sorted(z.namelist()) == ["manifest.json", "part.stl"]
    assert not list(part.folder.glob("*.part"))  # the temporary files are gone


def test_concurrent_builds_share_one_build(tmp_path, monkeypatch):
    calls = []
    real = artifacts.build

    def counting(spec):
        calls.append(1)
        time.sleep(0.3)
        return real(spec)

    monkeypatch.setattr(artifacts, "build", counting)
    results = [None, None]

    def run(i):
        results[i] = build_part(SPEC, GeometrySettings(), tmp_path)

    threads = [threading.Thread(target=run, args=(i,)) for i in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(calls) == 1
    first, second = results
    assert first.key == second.key and first.preview.exists() and second.preview.exists()


def test_concurrent_exports_of_the_same_part_do_not_collide(tmp_path, monkeypatch):
    write_calls = {"stl": 0, "step": 0}
    real = {"stl": exporters.WRITERS["stl"], "step": exporters.WRITERS["step"]}

    def counted(fmt):
        def write(solid, path, quality="normal"):
            write_calls[fmt] += 1
            time.sleep(0.05)
            return real[fmt](solid, path, quality)
        return write

    monkeypatch.setitem(exporters.WRITERS, "stl", counted("stl"))
    monkeypatch.setitem(exporters.WRITERS, "step", counted("step"))
    run_calls = []

    def fake_run(cmd, timeout_s, log_path, cwd=None):
        run_calls.append(cmd)
        time.sleep(0.05)
        Path(cmd[cmd.index("--output") + 1]).write_text(
            "G1 X1\n; filament used [g] = 1.0\n; estimated printing time (normal mode) = 1m 0s\n")
        return 0

    monkeypatch.setattr("s2c.multiview.slice.run", fake_run)
    part = build_part(SPEC, root=tmp_path)

    def do_export(_):
        return export_part(part, ["stl", "step", "gcode"], slicer=tmp_path / "s.exe")

    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(do_export, range(2)))
    assert write_calls == {"stl": 1, "step": 1} and len(run_calls) == 1


def test_swept_folder_rebuilds(tmp_path, monkeypatch):
    calls = []
    real = artifacts.build
    monkeypatch.setattr(artifacts, "build", lambda spec: calls.append(1) or real(spec))
    first = build_part(SPEC, root=tmp_path)
    for _ in range(20):  # Windows (indexer, antivirus) can hold a file just written for a moment
        try:
            shutil.rmtree(first.folder)
            break
        except OSError:
            time.sleep(0.1)
    second = build_part(SPEC, root=tmp_path)
    assert second.preview.exists() and len(calls) == 2


def test_the_sweep_deletes_old_folders_only(tmp_path):
    old, fresh = tmp_path / "old", tmp_path / "fresh"
    old.mkdir()
    fresh.mkdir()
    past = time.time() - 7200
    os.utime(old, (past, past))
    sweep(tmp_path)
    assert not old.exists() and fresh.exists()


def test_a_drawing_writer_failure_keeps_the_other_files(tmp_path, monkeypatch):
    """A writer that fails on an unusual solid loses its own format, said in the warnings, never the whole export."""
    from s2c.multiview import artifacts as A

    def broken(*args, **kwargs):
        raise RuntimeError("HLR failed")

    monkeypatch.setattr(A, "write_drawings", broken)
    monkeypatch.setattr("s2c.multiview.slice.find_slicer", lambda: None)
    part = A.build_part(SPEC, None, tmp_path)
    res = A.export_part(part, ["stl", "pdf"])
    assert set(res.files) == {"stl"} and any("PDF" in w and "drawing" in w for w in res.warnings), res.warnings
