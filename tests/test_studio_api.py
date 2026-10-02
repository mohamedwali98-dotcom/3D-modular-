import json
import os
import sys
import time
import zipfile
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from s2c.multiview import artifacts
from s2c.web import api, files

SPEC = json.loads((Path(__file__).parents[1] / "examples" / "mv" / "l_bracket.json").read_text())


def client(tmp_path, monkeypatch):
    artifacts.clear_cache()
    monkeypatch.setattr(api, "ARTIFACT_ROOT", tmp_path)
    monkeypatch.setattr("s2c.multiview.slice.find_slicer", lambda: None)
    app = FastAPI()
    api.install_error_handlers(app)
    app.include_router(api.router)
    return TestClient(app, client=("127.0.0.1", 50000))


def test_export_returns_downloadable_files_and_a_zip(tmp_path, monkeypatch):
    c = client(tmp_path, monkeypatch)
    body = c.post("/api/export", json={"spec": SPEC, "settings": {"export": {"formats": ["stl", "dxf"]}}}).json()
    assert set(body["files"]) == {"stl", "dxf"}
    assert c.get(body["files"]["stl"]["url"]).status_code == 200 and c.get(body["zip_url"]).status_code == 200


def test_export_sweeps_builds_older_than_an_hour(tmp_path, monkeypatch):
    c = client(tmp_path, monkeypatch)
    stale = tmp_path / "stale-build"
    stale.mkdir()
    old = time.time() - 7200
    os.utime(stale, (old, old))
    c.post("/api/export", json={"spec": SPEC, "settings": {"export": {"formats": ["stl"]}}})
    assert not stale.exists()


def test_bad_settings_and_paths_are_refused(tmp_path, monkeypatch):
    c = client(tmp_path, monkeypatch)
    assert c.post("/api/export", json={"spec": SPEC, "settings": {"printing": {"layer_mm": 0.9}}}).status_code == 422
    assert c.get("/api/artifacts/" + "0" * 20 + "/..%2F..%2Fsecret").status_code == 404
    assert c.get("/api/artifacts/nothex/part.stl").status_code == 404


def test_the_cli_exports_formats(tmp_path, monkeypatch):
    from scripts import mv_export
    monkeypatch.setattr("s2c.multiview.slice.find_slicer", lambda: None)
    spec_path = Path(__file__).parents[1] / "examples" / "mv" / "l_bracket.json"
    monkeypatch.setattr(sys, "argv", ["mv_export", str(spec_path),
                                      "--format", "stl", "--format", "step", "--out", str(tmp_path)])
    mv_export.main()
    assert any(p.name == "part.stl" for p in tmp_path.rglob("*")) and list(tmp_path.glob("*.zip"))


def test_the_cli_records_settings_in_the_manifest(tmp_path, monkeypatch):
    from scripts import mv_export
    monkeypatch.setattr("s2c.multiview.slice.find_slicer", lambda: None)
    spec_path = Path(__file__).parents[1] / "examples" / "mv" / "l_bracket.json"
    monkeypatch.setattr(sys, "argv", ["mv_export", str(spec_path), "--format", "stl",
                                      "--quality", "fine", "--material", "PETG", "--out", str(tmp_path)])
    mv_export.main()
    zip_path = next(tmp_path.glob("*.zip"))
    manifest = json.loads(zipfile.ZipFile(zip_path).read("manifest.json"))
    assert manifest["settings"]["mesh"]["quality"] == "fine"
    assert manifest["settings"]["printing"]["material"] == "PETG"


def test_artifact_route_rejects_dotdot(tmp_path, monkeypatch):
    c = client(tmp_path, monkeypatch)
    assert files.NAME.fullmatch("..") is None
    key = "0" * 20
    for url in (f"/api/artifacts/{key}/..%2Fx", f"/api/artifacts/{key}/a/../b"):
        status = c.get(url).status_code
        assert status != 200
        assert status in (400, 404)


def test_export_abstain(tmp_path, monkeypatch):
    c = client(tmp_path, monkeypatch)
    # a fillet too large for the part to hold: tests/test_studio_artifacts.py proves this is a real
    # BuildError("fillet_failed") out of build_part, not a mock.
    body = {"spec": SPEC, "settings": {"geometry": {"finish": "fillet", "finish_mm": 8.0}}}
    r = c.post("/api/export", json=body)
    assert r.status_code == 200
    abstain = r.json()["abstain"]
    assert abstain["stage"] == "build"
    assert abstain["reason"] == "fillet_failed"
    assert abstain["remedy"]


def test_cli_bad_material(tmp_path, capsys):
    from scripts import mv_export
    spec_path = Path(__file__).parents[1] / "examples" / "mv" / "l_bracket.json"
    with pytest.raises(SystemExit) as exc:
        mv_export.main(["--material", "WOOD", str(spec_path), "--out", str(tmp_path)])
    assert exc.value.code == 2
    err = capsys.readouterr().err
    assert "Traceback" not in err
    assert "PLA" in err and "PETG" in err


def test_cli_help_does_not_sweep(monkeypatch):
    from scripts import mv_export

    def boom():
        raise AssertionError("swept before --help was handled")

    monkeypatch.setattr(mv_export, "sweep", boom)
    with pytest.raises(SystemExit) as exc:
        mv_export.main(["--help"])
    assert exc.value.code == 0


def test_cli_bad_layer_does_not_sweep(tmp_path, capsys, monkeypatch):
    from scripts import mv_export

    def boom():
        raise AssertionError("swept before the bad --layer was reported")

    monkeypatch.setattr(mv_export, "sweep", boom)
    spec_path = Path(__file__).parents[1] / "examples" / "mv" / "l_bracket.json"
    with pytest.raises(SystemExit) as exc:
        mv_export.main(["--layer", "5.0", str(spec_path), "--out", str(tmp_path)])
    assert exc.value.code == 2
    err = capsys.readouterr().err
    assert "Traceback" not in err
