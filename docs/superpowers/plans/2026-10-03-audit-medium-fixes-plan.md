# Audit Medium-priority fixes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Land the audit's Medium items M2–M12 (M1 is done).

**Architecture:** Local refactors behind tests: public helper names, a typed edit parser, a fuse that locks its own state, one pipeline per process, an observability module, a settings registry, honest dependency floors, generated TS models, strict xfails.

**Tech Stack:** Python 3.11, FastAPI, pydantic v2, pytest, React/TS (vitest, tsc), GitHub Actions.

**Spec:** `docs/superpowers/specs/2026-10-03-audit-medium-fixes-design.md`

## Global Constraints

- Never add `Co-Authored-By`, "Generated with", or any AI attribution to a commit (CLAUDE.md).
- Never read or commit `.env`; never commit logs, the "Reverce engineering" dataset or user images.
- Write the failing test first, then the code. ruff (line length 120) clean before every commit.
- No new runtime dependency.
- Browser-facing errors stay plain sentences.

## Review Focus

- A refused edit path must answer 400 at `/api/merge`, and every path the web app and the Studio send must still be accepted.
- Two concurrent merges of one analysis must never run fuse at the same time, now that jobs do not lock.
- The settings check must not refuse a valid `.env` (every documented variable, commented presets included).
- The generated TS models must type-check against every use in the web app.
- Logging configured by the server must not break uvicorn's own access and error logs.

---

### Task 1: Public module boundaries (M7)

**Files:** Modify `s2c/multiview/outline.py`, `pipeline.py`, `build.py`, `turned.py`, `label.py`, `qwen_reader.py`, `sheet.py`, `sheet_read.py`; create `tests/builders.py`; modify the tests importing `_block`, `_cut`, `_pins`, `_pockets`, `_spec`, `_page` (from `tests/test_mv_relief.py`) and `_service` (from `tests/test_mv_sheet_read.py`). Test: `tests/test_boundaries.py` (new).

- [ ] **Step 1: Write the failing test**

```python
"""No module imports another module's private name (audit M7): a helper used elsewhere is public."""
import ast
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _private_imports(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return [f"{node.module}.{a.name}" for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)
            and node.module and node.module.split(".")[0] in ("s2c", "tests")
            for a in node.names if a.name.startswith("_") and not a.name.startswith("__")]


@pytest.mark.parametrize("folder", ["s2c", "tests", "scripts"])
def test_no_module_imports_a_private_name(folder):
    found = {str(p.relative_to(ROOT)): names for p in sorted((ROOT / folder).rglob("*.py"))
             if (names := _private_imports(p))}
    assert found == {}
```

- [ ] **Step 2: Run** `uv run pytest tests/test_boundaries.py -q` — Expected: FAIL listing `_stroke`, `_clean`, `_strip_fences`, `_remove_border`, `_block`, …
- [ ] **Step 3: Implement.** Rename `outline._stroke` → `stroke_width`, `build._clean` → `clean_polygon`, `label._strip_fences` → `strip_fences`, `sheet._remove_border` → `remove_border` (all uses). Move `_page`, `_block`, `_cut`, `_spec`, `_pockets`, `_pins` to `tests/builders.py` as `page`, `block`, `cut`, `relief_spec`, `pockets`, `pins`, and `_service` as `word_service`; `test_mv_relief.py` and `test_mv_sheet_read.py` import them from there.
- [ ] **Step 4: Run** `uv run pytest tests/test_boundaries.py tests/test_mv_relief.py tests/test_mv_sheet_read.py tests/test_mv_sketch_read.py tests/test_mv_sheet_dims.py tests/test_mv_relief_review.py tests/test_web_api.py tests/test_studio_sheet.py tests/test_mv_turned.py tests/test_mv_build.py -q` — Expected: pass.
- [ ] **Step 5: Commit** `Helpers used across modules are public, and shared test builders live in tests/builders.py`.

### Task 2: Typed edits (M6)

**Files:** Create `s2c/multiview/edits.py`. Modify `s2c/multiview/spec.py` (`feature_path`, `finish_path`), `fuse.py`, `build.py`, `drawing.py`, `finish.py`, `relief.py`, `s2c/web/describe.py`, `s2c/web/api.py` (`/api/merge`). Test: `tests/test_mv_edits.py` (new), `tests/test_web_api.py`.

**Interfaces:** `edits.parse(user_values: dict[str, float]) -> dict[str, float]` (the same map, every path checked) raising `edits.EditError(path)`; `spec.feature_path(i: int, name: str) -> str`; `spec.finish_path(i: int, name: str) -> str`.

- [ ] **Step 1: Write the failing tests**

```python
"""The user's edits (audit M6): one parser says which paths exist; nothing else reaches a spec."""
import pytest

from s2c.multiview import edits
from s2c.multiview.spec import feature_path


def test_known_paths_pass():
    values = {"envelope.x_mm": 60, "features[0].diameter_mm": 6, "features[2].keep": 0, "features[1].angle_deg": 30}
    assert edits.parse(values) == values


@pytest.mark.parametrize("path", ["features[0].type", "envelope.w_mm", "features[a].a_mm", "views.front.outer",
                                  "features[0].face", "finishes[0].radius_mm "])
def test_any_other_path_is_refused(path):
    with pytest.raises(edits.EditError):
        edits.parse({path: 1})


def test_provenance_keys_have_one_spelling():
    assert feature_path(3, "diameter_mm") == "features[3].diameter_mm"
```
and in `tests/test_web_api.py`:

```python
def test_an_edit_to_a_field_that_does_not_exist_is_refused():
    job = analyze()
    r = c.post("/api/merge", json={"request_id": job["job_id"], "user_values": {"features[0].type": 1}})
    assert r.status_code == 400 and "features[0].type" in r.json()["error"]
```

- [ ] **Step 2: Run** `uv run pytest tests/test_mv_edits.py tests/test_web_api.py -k "path or edit" -q` — Expected: FAIL.
- [ ] **Step 3: Implement.** `edits.py`: `ENVELOPE = re.compile(r"envelope\.[xyz]_mm")`, `FEATURE = re.compile(r"features\[(\d+)\]\.(\w+)")` with field in `FIELDS = {"a_mm", "b_mm", "diameter_mm", "depth_mm", "width_mm", "length_mm", "angle_deg", "height_mm", "keep"}`; `parse` returns the map or raises `EditError(path)`. `/api/merge` calls it and answers 400 `f"{path} cannot be edited."`. `feature_path`/`finish_path` replace every f-string key on the served path.
- [ ] **Step 4: Run** `uv run pytest tests/test_mv_edits.py tests/test_web_api.py tests/test_mv_fuse.py tests/test_mv_build.py tests/test_studio_ui.py -q` — Expected: pass.
- [ ] **Step 5: Commit** `Edits are parsed once: a path that names no editable value is refused, and provenance keys have one spelling`.

### Task 3: Fuse guards its own state (M5)

**Files:** Modify `s2c/multiview/pipeline.py` (`Observed.lock`, `fuse`, `forget_images`), `s2c/web/jobs.py` (drop `merge_lock`). Test: `tests/test_mv_pipeline.py`.

- [ ] **Step 1: Write the failing test**

```python
def test_two_fuses_of_one_analysis_never_run_at_once(monkeypatch):
    """The web job and a merge (or two merges) may fuse the same Observed together: fuse locks it itself."""
    import threading
    import time
    from s2c.multiview import pipeline as P
    pipe = MvPipeline()
    observed = pipe.observe([ImageInput(sketch(), "front", "sketch")])
    inside, most = [0], [0]
    real = P.complete

    def slow(*a, **k):
        inside[0] += 1
        most[0] = max(most[0], inside[0])
        time.sleep(0.2)
        inside[0] -= 1
        return real(*a, **k)
    monkeypatch.setattr(P, "complete", slow)
    threads = [threading.Thread(target=pipe.fuse, args=(observed, {"envelope.x_mm": 60, "envelope.y_mm": 40,
                                                                  "envelope.z_mm": 5})) for _ in range(3)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert most[0] == 1
```

- [ ] **Step 2: Run** `uv run pytest tests/test_mv_pipeline.py -k never_run_at_once -q` — Expected: FAIL (`most` 3).
- [ ] **Step 3: Implement.** `Observed.lock: threading.RLock` (field, excluded from comparison); `fuse` runs its body under it; `forget_images` takes it; `jobs` drops `merge_lock` and its `with` blocks.
- [ ] **Step 4: Run** `uv run pytest tests/test_mv_pipeline.py tests/test_web_api.py tests/test_web_jobs.py tests/test_studio_ui.py -q` — Expected: pass.
- [ ] **Step 5: Commit** `Fuse locks the analysis it works on, so no caller has to`.

### Task 4: One pipeline per process, a calm sweep (M3, M4)

**Files:** Modify `s2c/web/api.py`, `s2c/web/server.py` (lifespan). Test: `tests/test_web_limits.py`.

**Interfaces:** `api.SWEEP_EVERY_S = 60`.

- [ ] **Step 1: Write the failing tests**

```python
def test_the_pipeline_is_built_once_however_many_requests_race(monkeypatch):
    import threading
    import time
    built = []
    monkeypatch.setattr(api, "_PIPELINE", None)
    monkeypatch.setattr(api, "default_pipeline", lambda: (time.sleep(0.2), built.append(1), MvPipeline())[-1])
    threads = [threading.Thread(target=api.get_pipeline) for _ in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert built == [1]


def test_the_sweep_runs_at_most_once_a_minute(monkeypatch):
    sweeps = []
    monkeypatch.setattr(api.jobs, "sweep_jobs", lambda ttl: sweeps.append(ttl))
    monkeypatch.setattr(api, "_last_sweep", 0.0)
    for _ in range(5):
        c.get("/api/status")
    assert len(sweeps) == 1
```

- [ ] **Step 2: Run** `uv run pytest tests/test_web_limits.py -k "built_once or once_a_minute" -q` — Expected: FAIL.
- [ ] **Step 3: Implement.** `get_pipeline` keeps `_PIPELINE` built under `_PIPELINE_LOCK` (double-checked); `_sweep` returns at once when `time.monotonic() - _last_sweep < SWEEP_EVERY_S`; `server.py` gets a lifespan that builds the pipeline at startup.
- [ ] **Step 4: Run** `uv run pytest tests/test_web_limits.py tests/test_web_api.py tests/test_web_guard.py tests/test_studio_api.py -q` — Expected: pass (tests relying on a sweep reset `_last_sweep`).
- [ ] **Step 5: Commit** `One pipeline per process, built at startup; the sweep runs at most once a minute`.

### Task 5: Logs, metrics and counted fallbacks (M8, M9)

**Files:** Create `s2c/obs.py`. Modify `s2c/web/server.py`, `s2c/web/api.py` (`/api/metrics`), `s2c/web/jobs.py` (job context, job and stage metrics), `s2c/web/guard.py` (refusals), `s2c/studio/app.py` (configure logging), and the provider fallbacks: `complete.py`, `hf3d.py`, `pipeline.py`, `qwen_image.py`, `qwen_reader.py`, `sheet.py`, `reading/service.py`, `reading/vlm.py`, `reading/env.py`, `reading/paddle.py`, `sketch/text.py`, `studio/handlers.py`, `web/api.py`. Test: `tests/test_obs.py` (new).

**Interfaces:** `obs.configure_logging()`, `obs.job_scope(job_id)` (context manager), `obs.count(name, **labels)`, `obs.time_spent(name, seconds, **labels)`, `obs.fallback(name, exc)`, `obs.render() -> str`, `obs.reset()`.

- [ ] **Step 1: Write the failing tests**

```python
"""Logs and metrics (audit M8, M9)."""
import json
import logging

from s2c import obs


def test_a_record_inside_a_job_carries_its_id(capsys):
    obs.configure_logging(force=True)
    with obs.job_scope("abc123"):
        logging.getLogger("s2c.test").warning("inside")
    line = json.loads(capsys.readouterr().err.strip().splitlines()[-1])
    assert line["job_id"] == "abc123" and line["message"] == "inside" and line["level"] == "WARNING"


def test_a_fallback_is_logged_and_counted():
    obs.reset()
    obs.fallback("triposr", RuntimeError("Space down"))
    obs.fallback("triposr", RuntimeError("Space down"))
    assert 's2c_fallbacks_total{name="triposr"} 2' in obs.render()


def test_metrics_render_in_prometheus_text():
    obs.reset()
    obs.count("s2c_jobs_total", mode="photos", outcome="done")
    obs.time_spent("s2c_stage_seconds", 1.5, stage="fuse")
    text = obs.render()
    assert 's2c_jobs_total{mode="photos",outcome="done"} 1' in text
    assert 's2c_stage_seconds_sum{stage="fuse"} 1.5' in text and 's2c_stage_seconds_count{stage="fuse"} 1' in text
```
and in `tests/test_web_api.py`: after `analyze()`, `c.get("/api/metrics").text` contains `s2c_jobs_total{mode="photos",outcome=`.

- [ ] **Step 2: Run** `uv run pytest tests/test_obs.py tests/test_web_api.py -k "metrics or fallback or carries" -q` — Expected: FAIL.
- [ ] **Step 3: Implement.** `obs.py`: a JSON formatter (`time`, `level`, `logger`, `message`, `job_id`, `exc`), `S2C_LOG_FORMAT=text` for plain lines, `S2C_LOG_LEVEL` (INFO); a `contextvars` job id added by a filter; a thread-safe registry of counters and sums; `render` in Prometheus text format. `server.py` configures logging at import (uvicorn keeps its own handlers). `jobs.run`/`run_sheet` run inside `job_scope(job.job_id)`, count `s2c_jobs_total{mode, outcome}` and time each stage on its done event. The guard counts `s2c_refused_total{reason}` (token, rate). Each provider fallback calls `obs.fallback(name, exc)` in place of its own warning.
- [ ] **Step 4: Run** `uv run pytest tests/test_obs.py tests/test_web_api.py tests/test_web_guard.py tests/test_mv_pipeline.py tests/reading -q` — Expected: pass.
- [ ] **Step 5: Commit** `Structured logs with the job id, Prometheus metrics at /api/metrics, and every provider fallback logged and counted`.

### Task 6: One settings registry, checked at startup (M2)

**Files:** Create `s2c/config.py`. Modify `s2c/web/server.py` (check at startup), `s2c/logdir.py`, `s2c/multiview/artifacts.py`, `app_mv_gradio.py`, `s2c/studio/handlers.py` / `app.py` (data folders), `.env.example`. Test: `tests/test_config.py` (new).

**Interfaces:** `config.VARIABLES: dict[str, Var]`; `config.problems(env) -> list[str]`; `config.data_dir() -> Path`; `config.data_path(*parts) -> Path`.

- [ ] **Step 1: Write the failing tests**

```python
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
```

- [ ] **Step 2: Run** `uv run pytest tests/test_config.py -q` — Expected: FAIL.
- [ ] **Step 3: Implement.** The registry (kinds `str`, `int`, `float`, `bool`, `path`, `url`, `choice(...)`); `problems`; `data_dir` = `S2C_DATA_DIR`, else the project folder when it holds `pyproject.toml`, else the working directory; `logdir`, `artifacts.ROOT`, the lab app's and the Studio's folders resolve through `data_path`; `server.py` lifespan raises `RuntimeError("Settings: …")` listing every problem; `.env.example` documents each variable (commented).
- [ ] **Step 4: Run** `uv run pytest tests/test_config.py tests/test_web_api.py tests/test_studio_artifacts.py tests/test_mv_gradio.py -q` — Expected: pass.
- [ ] **Step 5: Commit** `Every setting is declared, documented and checked at startup; working folders follow S2C_DATA_DIR`.

### Task 7: Honest dependencies (M10)

**Files:** Modify `pyproject.toml`, `uv.lock` (relock), `scripts/setup_triposr.ps1`, `s2c/multiview/hf3d.py`, `README.md`; create `.github/dependabot.yml`. Test: `tests/test_docs.py` (dependency floors).

- [ ] **Step 1: Write the failing test** (in `tests/test_docs.py`)

```python
def test_the_trocr_extra_asks_for_the_transformers_it_is_tested_with():
    import tomllib
    extras = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]["optional-dependencies"]
    assert "transformers>=5" in extras["trocr"] and len({d for e in extras.values() for d in e if d.startswith("torch")}) == 1
```

- [ ] **Step 2: Run** `uv run pytest tests/test_docs.py -q` — Expected: FAIL.
- [ ] **Step 3: Implement.** `trocr = ["transformers>=5", "torch>=2.4"]`, `ai` uses the same torch floor; `uv lock`; `setup_triposr.ps1` checks out `107cefdc244c39106fa830359024f6a2f1c78871` and its comment says CUDA 12.6; `hf3d` appends the TripoSR folder to `sys.path`; `.github/dependabot.yml` (uv at `/`, npm at `/web`, github-actions at `/`, weekly); README: local TripoSR needs transformers below 5 (the Space fallback works).
- [ ] **Step 4: Run** `uv lock --check && uv run pytest tests/test_docs.py tests/test_mv_hf3d.py -q` — Expected: pass.
- [ ] **Step 5: Commit** `Dependency floors say what is tested; TripoSR is pinned and never shadows installed packages; Dependabot watches the locks`.

### Task 8: Frontend models generated from the backend (M11)

**Files:** Create `scripts/gen_ts_models.py`, `web/src/api/models.gen.ts`. Modify `web/src/api/types.ts`. Test: `tests/test_ts_models.py` (new).

- [ ] **Step 1: Write the failing test**

```python
"""The web app's copies of the shared models are generated from the Pydantic models (audit M11)."""
from pathlib import Path

from scripts.gen_ts_models import render

GEN = Path(__file__).resolve().parents[1] / "web" / "src" / "api" / "models.gen.ts"


def test_the_generated_models_are_current():
    assert GEN.read_text(encoding="utf-8") == render(), "run: uv run python scripts/gen_ts_models.py"
```

- [ ] **Step 2: Run** `uv run pytest tests/test_ts_models.py -q` — Expected: FAIL (no module).
- [ ] **Step 3: Implement.** `render()` turns the JSON schemas of `MultiViewSpec`, `MvAbstain`, `AiSettings`, `GeometrySettings`, `MeshSettings`, `PrintSettings`, `ExportSettings` into TS interfaces (every property present, `null` unions, literal enums, tuples, records, discriminated unions); `main()` writes the file. `types.ts` re-exports `Spec`, `Outline`, `Hole`, `Slot`, `Pocket`, `Boss`, `AiSettings`, `GeometrySettings`, `PrintSettings` from it.
- [ ] **Step 4: Run** `uv run pytest tests/test_ts_models.py -q` and `cd web && npm run build && npm test` — Expected: pass.
- [ ] **Step 5: Commit** `The web app's spec and settings types are generated from the Pydantic models, and a test keeps them current`.

### Task 9: Strict expected failures (M12)

**Files:** Modify `tests/sketch/test_golden.py:18`, `tests/sketch/test_pipeline.py:19`.

- [ ] **Step 1:** Set `strict=True` on both marks.
- [ ] **Step 2: Run** `uv run pytest tests/sketch -q -rx` three times — Expected: the same xfails each run, no XPASS.
- [ ] **Step 3: Commit** `Known-limit tests are strict: a fix that makes one pass fails the run until its mark goes`.
