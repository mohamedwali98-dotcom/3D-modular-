import sys
import time
from pathlib import Path

import pytest
from pydantic import ValidationError

from s2c.multiview.proc import run, tail
from s2c.multiview.settings import (
    DENSITIES,
    FORMATS,
    MATERIALS,
    AiSettings,
    ExportSettings,
    GeometrySettings,
    MeshSettings,
    PrintSettings,
    StudioSettings,
    filament_metres,
    settings_hash,
)


def test_defaults_are_valid_and_match_the_spec():
    s = StudioSettings()
    assert (s.ai.seed, s.ai.attempts, s.geometry.clearance, s.mesh.quality) == (7, 2, "medium", "normal")
    assert (s.printing.material, s.printing.layer_mm, s.printing.infill_pct, s.printing.supports) == \
        ("PLA", 0.2, 20, "buildplate")
    assert s.export.formats == ["stl", "step", "3mf", "gcode"]
    assert MeshSettings(quality="fine").tolerances == (0.005, 0.1)
    assert MATERIALS["PETG"][0] == 240


def test_ranges_are_enforced():
    for bad in ({"layer_mm": 0.4}, {"nozzle_mm": 0.5}, {"infill_pct": 101}, {"scale_pct": 20}, {"material": "PVC"}):
        with pytest.raises(ValidationError):
            PrintSettings(**bad)
    with pytest.raises(ValidationError):
        GeometrySettings(finish_mm=50)
    with pytest.raises(ValidationError):
        ExportSettings(formats=["stl", "xyz"])


def test_a_layer_must_fit_the_nozzle():
    assert PrintSettings(nozzle_mm=0.6, layer_mm=0.32).layer_mm == 0.32
    with pytest.raises(ValidationError):
        PrintSettings(nozzle_mm=0.2, layer_mm=0.2)


def test_formats_are_kept_in_registry_order_without_duplicates():
    assert ExportSettings(formats=["gcode", "stl", "stl", "pdf"]).formats == ["stl", "pdf", "gcode"]
    assert list(FORMATS)[:3] == ["stl", "step", "3mf"]


def test_the_hash_is_stable_and_changes_with_values():
    assert settings_hash(PrintSettings()) == settings_hash(PrintSettings())
    assert settings_hash(PrintSettings()) != settings_hash(PrintSettings(infill_pct=40))
    assert len(settings_hash(GeometrySettings())) == 16


def test_filament_length_comes_from_the_material_density():
    assert set(DENSITIES) == set(MATERIALS)
    assert filament_metres(12.4, "PLA") == pytest.approx(4.16, abs=0.005)  # 10 cm3 of 1.75 mm filament
    assert filament_metres(12.4, "ABS") > filament_metres(12.4, "PETG")  # lighter plastic, longer strand


def test_run_writes_output_to_the_log(tmp_path):
    code = run([sys.executable, "-c", "print('hello from a child')"], 30, tmp_path / "log.txt")
    assert code == 0 and "hello from a child" in tail(tmp_path / "log.txt")


def test_run_kills_a_program_that_hangs(tmp_path):
    t0 = time.monotonic()
    code = run([sys.executable, "-c", "import time; time.sleep(60)"], 1, tmp_path / "log.txt")
    assert code is None and time.monotonic() - t0 < 15


def test_tail_reads_only_the_end(tmp_path, monkeypatch):
    log = tmp_path / "log.txt"
    log.write_text("\n".join(f"line {i}" for i in range(200_000)) + "\n", encoding="utf-8")

    def _whole_file_read(self, *args, **kwargs):
        raise AssertionError("whole file read")

    monkeypatch.setattr(Path, "read_text", _whole_file_read)
    assert tail(log, 3) == "line 199997\nline 199998\nline 199999"


def test_tail_short_and_missing(tmp_path):
    log = tmp_path / "log.txt"
    log.write_text("one\ntwo\n", encoding="utf-8")
    assert tail(log, 20) == "one\ntwo"
    assert tail(tmp_path / "missing.txt", 20) == ""
    bad = tmp_path / "bad.txt"
    bad.write_bytes(b"\xff\xfeok\n")
    assert tail(bad, 20).endswith("ok")


def test_run_log_is_binary(tmp_path):
    log = tmp_path / "log.txt"
    code = run([sys.executable, "-c", "import sys; sys.stdout.buffer.write(b'\\xff\\xfeok\\n')"], 30, log)
    assert code == 0
    assert log.read_bytes().endswith(b"\xff\xfeok\n")


def test_settings_defaults():
    ai = AiSettings()
    assert (ai.use_reader, ai.use_qwen_image, ai.use_rescue, ai.use_triposr, ai.use_solaria, ai.randomize_seed) == \
        (True, False, False, False, False, False)  # hosted image services are opt-in
    geo = GeometrySettings()
    assert (geo.snap, geo.finish, geo.finish_edges, geo.clearance) == (True, "none", "all_vertical", "medium")
