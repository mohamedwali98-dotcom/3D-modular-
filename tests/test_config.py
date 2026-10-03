"""Settings (audit M2): every variable the code reads is declared, documented and checked."""
import re
from pathlib import Path

from s2c import config

ROOT = Path(__file__).resolve().parents[1]
READ = re.compile(r"""(?:os\.environ\.get|os\.getenv|os\.environ\[|_setting)\(?\s*["']([A-Z][A-Z0-9_]+)["']""")


def test_every_variable_the_code_reads_is_declared_and_documented():
    read = {m for p in (ROOT / "s2c").rglob("*.py") for m in READ.findall(p.read_text(encoding="utf-8"))}
    example = (ROOT / ".env.example").read_text(encoding="utf-8")
    assert read - set(config.VARIABLES) == set()
    assert [v for v in config.VARIABLES if v not in example] == []


def test_bad_values_are_named():
    found = config.problems({"READ_TIMEOUT_S": "soon", "QWEN_IMAGE_BACKEND": "bogus", "TROCR_BATCH": "8"})
    assert len(found) == 2 and any("READ_TIMEOUT_S" in p for p in found) and any("QWEN_IMAGE_BACKEND" in p for p in found)


def test_working_folders_follow_s2c_data_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("S2C_DATA_DIR", str(tmp_path))
    assert config.data_path("tmp", "mv") == tmp_path / "tmp" / "mv"


def test_the_server_refuses_to_start_on_a_bad_value(monkeypatch):
    import pytest
    from fastapi.testclient import TestClient

    from s2c.web import api
    from s2c.web.server import app

    monkeypatch.setenv("READ_TIMEOUT_S", "soon")
    monkeypatch.setattr(api, "default_pipeline", lambda: pytest.fail("the models load before the settings are checked"))
    with pytest.raises(RuntimeError, match="READ_TIMEOUT_S='soon'"), TestClient(app):
        pass


def test_every_documented_line_works_when_uncommented():
    """A line of .env.example uncommented as it stands is a valid value: its explanation is a comment, not part of it."""
    import io

    from dotenv import dotenv_values

    text = (ROOT / ".env.example").read_text(encoding="utf-8")
    uncommented = re.sub(r"^#\s*(?=[A-Z][A-Z0-9_]*=)", "", text, flags=re.MULTILINE)
    values = {k: v or "" for k, v in dotenv_values(stream=io.StringIO(uncommented)).items() if k in config.VARIABLES}
    assert config.problems(values) == []
    assert {k: v for k, v in values.items() if "  " in v or v.startswith("#")} == {}


def test_a_bad_log_level_is_named_by_the_check_not_a_crash(monkeypatch):
    import io
    import logging

    from s2c import obs

    monkeypatch.setenv("S2C_LOG_LEVEL", "verbose")
    handler = obs.configure_logging(stream=io.StringIO(), force=True)
    logging.getLogger().removeHandler(handler)
    assert any("S2C_LOG_LEVEL" in p for p in config.problems())


def test_a_bad_pixel_cap_is_named_by_the_check_not_a_crash():
    import os
    import subprocess
    import sys

    env = {**os.environ, "S2C_MAX_PIXELS": "4e7"}
    out = subprocess.run([sys.executable, "-c", "import s2c; print(s2c.MAX_PIXELS)"], env=env, capture_output=True,
                         text=True, cwd=ROOT, check=False)
    assert out.returncode == 0 and out.stdout.strip() == "40000000", out.stderr
    assert any("S2C_MAX_PIXELS" in p for p in config.problems({"S2C_MAX_PIXELS": "4e7"}))


def test_a_choice_the_check_accepts_is_the_choice_the_code_makes(monkeypatch):
    from s2c.multiview import qwen_image
    from s2c.sketch import text

    for name, value in {"QWEN_IMAGE_BACKEND": "DashScope", "QWEN_IMAGE_BASE_URL": "https://x.example/api",
                        "QWEN_IMAGE_MODEL": "m", "VLM_API_KEY": "k", "SKETCH_TEXT_DETECTOR": "Paddle"}.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setattr(qwen_image, "dashscope_gen", lambda *args: "dashscope")
    monkeypatch.setattr(text, "_paddle_boxes", lambda sheet: "paddle")
    assert config.problems() == [] or all("QWEN" not in p and "SKETCH" not in p for p in config.problems())
    assert qwen_image.default_gen() == "dashscope"
    assert text.detect_text_boxes(None, None, 1.0) == "paddle"
    assert config.problems({"S2C_LOG_LEVEL": "critical"}) == [] and config.problems({"S2C_LOG_LEVEL": "warn"}) == []


def test_the_gradio_apps_may_serve_files_from_the_data_folder(monkeypatch, tmp_path):
    import app_mv_gradio as lab
    from s2c.studio import app as studio

    calls = []

    class Fake:
        def queue(self, **kwargs):
            return self

        def launch(self, **kwargs):
            calls.append(kwargs)

    monkeypatch.setenv("S2C_DATA_DIR", str(tmp_path))
    monkeypatch.setattr(config, "load_env", lambda: None)
    monkeypatch.setattr(studio.obs, "configure_logging", lambda *a, **k: None)
    monkeypatch.setattr(studio.config, "problems", lambda *a: [])
    monkeypatch.setattr(studio, "build_app", lambda *a: Fake())
    studio.launch()
    monkeypatch.setattr(lab, "build_app", lambda *a: Fake())
    monkeypatch.setattr(lab, "default_pipeline", lambda: None)
    lab.main()
    assert [c.get("allowed_paths") for c in calls] == [[str(tmp_path)], [str(tmp_path)]]


def test_import_time_and_runtime_settings_come_from_the_same_env(tmp_path, monkeypatch):
    """The .env read at import (S2C_DATA_DIR, S2C_MAX_PIXELS) is the one the entry points load, wherever the app is
    started from: the first one up from the package, never the working directory's."""
    import os

    project, elsewhere = tmp_path / "project", tmp_path / "elsewhere"
    (project / "s2c").mkdir(parents=True)
    elsewhere.mkdir()
    (project / ".env").write_text("S2C_DATA_DIR=from-project\nS2C_TEST_SENTINEL=loaded\n", encoding="utf-8")
    (elsewhere / ".env").write_text("S2C_DATA_DIR=from-cwd\n", encoding="utf-8")
    monkeypatch.setattr(config, "HERE", project / "s2c")
    monkeypatch.chdir(elsewhere)
    monkeypatch.delenv("S2C_DATA_DIR", raising=False)
    assert config.dotenv_path() == project / ".env"
    assert config.setting("S2C_DATA_DIR", "") == "from-project"
    try:
        config.load_env()
        assert os.environ.get("S2C_TEST_SENTINEL") == "loaded"
    finally:
        os.environ.pop("S2C_TEST_SENTINEL", None)
        os.environ.pop("S2C_DATA_DIR", None)


def test_every_entry_point_loads_the_env_through_config():
    entry = [*(ROOT / "s2c").rglob("*.py"), *(ROOT / "scripts").glob("*.py"), *ROOT.glob("app_*.py")]
    assert [p.name for p in entry if "load_dotenv(" in p.read_text(encoding="utf-8") and p.name != "config.py"] == []
