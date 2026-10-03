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
