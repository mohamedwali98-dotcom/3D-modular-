"""Every file, module and command the README and CLAUDE.md name exists (audit H9): the docs describe the code
that runs."""
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PATH = re.compile(r"`((?:s2c|scripts|tests|web|docs|examples|profiles)/[\w./-]*\w|app_\w+\.py)`")
APP = re.compile(r"uvicorn ([\w.]+):app")
RUN = re.compile(r"uv run python ((?:scripts/)?\w+\.py)")


@pytest.mark.parametrize("doc", ["README.md", "CLAUDE.md"])
def test_every_named_path_exists(doc):
    text = (ROOT / doc).read_text(encoding="utf-8")
    missing = [p for p in PATH.findall(text) if "<" not in p and not (ROOT / p).exists()]
    missing += [m for m in APP.findall(text) if not (ROOT / (m.replace(".", "/") + ".py")).exists()]
    missing += [p for p in RUN.findall(text) if not (ROOT / p).exists()]
    assert missing == []


def test_the_trocr_extra_asks_for_the_transformers_it_is_tested_with():
    import tomllib
    extras = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]["optional-dependencies"]
    assert "transformers>=5" in extras["trocr"] and len({d for e in extras.values() for d in e if d.startswith("torch")}) == 1


def test_the_triposr_setup_reaches_its_commit_even_from_an_old_shallow_clone():
    script = (ROOT / "scripts" / "setup_triposr.ps1").read_text(encoding="utf-8")
    assert "fetch --depth 1 origin $Commit" in script and "$LASTEXITCODE" in script


def test_the_readme_says_what_the_lock_installs_for_triposr():
    import tomllib
    lock = tomllib.loads((ROOT / "uv.lock").read_text(encoding="utf-8"))
    versions = {p["version"] for p in lock["package"] if p["name"] == "transformers"}
    if all(int(v.split(".")[0]) >= 5 for v in versions):
        assert "local TripoSR does not run in the locked environment" in (ROOT / "README.md").read_text(encoding="utf-8")


def test_the_readme_names_the_metrics_and_the_log_folder():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "/api/metrics" in readme and "default `<S2C_DATA_DIR>/logs`" in readme


def test_the_readme_tells_an_operator_how_to_keep_the_job_id():
    assert "%(job_id)s" in (ROOT / "README.md").read_text(encoding="utf-8")
