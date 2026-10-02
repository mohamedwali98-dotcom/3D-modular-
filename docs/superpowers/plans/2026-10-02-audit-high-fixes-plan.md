# Audit High-priority fixes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Land the audit's High items H1–H5, H7–H9 and M1 so the served app is safe to leave running and its docs are true.

**Architecture:** Small, independent changes at the edges: the upload path (`s2c/web/api.py`), the job runner (`s2c/web/jobs.py`), the TrOCR reader, the AI defaults, the log paths, CI and the docs. No contract change except new optional limits.

**Tech Stack:** Python 3.11, FastAPI/Starlette, pytest, OpenCV, Pillow, React/TS (vitest), GitHub Actions.

**Spec:** `docs/superpowers/specs/2026-10-02-audit-high-fixes-design.md`

## Global Constraints

- Never add `Co-Authored-By`, "Generated with", or any AI attribution to a commit (CLAUDE.md).
- Never read or commit `.env`; never commit `logs/*.jsonl`, the "Reverce engineering" dataset or user images.
- Write the failing test first, then the code (CLAUDE.md).
- ruff line length 120; `uv run ruff check s2c scripts tests` clean before every commit.
- Every browser-facing error stays a plain sentence; no exception text reaches the browser.

## Review Focus

- A PNG of many megapixels that compresses to a few kilobytes must be refused before any full decode, on every upload path.
- A chunked upload (no Content-Length) must be cut at the same limit as a declared one.
- A job a user is still reviewing must survive a busy hour of other jobs.
- Flipping the hosted-service defaults must not change what an explicit `ai` setting asks for.
- Moving the log directory must leave every existing log writer working (VLM, chat, Qwen-Image, Solaria, reading).

---

### Task 1: Images are capped before decode (H1)

**Files:** Modify `s2c/__init__.py`, `s2c/web/api.py`. Test: `tests/test_web_limits.py` (new).

**Interfaces:** Produces `s2c.MAX_PIXELS: int` (40_000_000) and `api.TOO_MANY_PIXELS: str`.

- [ ] **Step 1: Write the failing tests**

```python
"""Upload limits (audit H1): pixels before decode, and a body without Content-Length."""
import io
import os

import cv2
import numpy as np
from fastapi.testclient import TestClient
from PIL import Image

import s2c
from s2c.multiview.pipeline import MvPipeline
from s2c.web import api
from s2c.web.api import get_pipeline
from s2c.web.server import app

app.dependency_overrides[get_pipeline] = lambda: MvPipeline()
c = TestClient(app, client=("127.0.0.1", 50000), raise_server_exceptions=False)


def _big_png() -> bytes:
    buf = io.BytesIO()
    Image.new("L", (8000, 6000)).save(buf, "PNG")  # 48 MP of black: a few kB on disk
    return buf.getvalue()


def test_an_image_with_too_many_pixels_is_refused_before_decoding():
    r = c.post("/api/analyze", files=[("files", ("big.png", _big_png(), "image/png"))],
               data={"faces": '["front"]', "kinds": '["sketch"]'})
    assert r.status_code == 413 and r.json()["error"] == api.TOO_MANY_PIXELS


def test_opencv_never_decodes_past_the_cap():
    assert os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] == str(s2c.MAX_PIXELS)
    assert cv2.imdecode(np.frombuffer(_big_png(), np.uint8), cv2.IMREAD_COLOR) is None


def test_a_body_without_content_length_is_cut_at_the_limit(monkeypatch):
    monkeypatch.setattr(api, "MAX_BODY", 1000)
    r = c.post("/api/analyze", content=(b"x" * 500 for _ in range(4)),
               headers={"Content-Type": "multipart/form-data; boundary=zz"})
    assert r.status_code == 413
```

- [ ] **Step 2: Run** `uv run pytest tests/test_web_limits.py -q` — Expected: 3 FAIL (no 413, no env var, no cut).
- [ ] **Step 3: Implement.** `s2c/__init__.py`: `MAX_PIXELS = int(os.environ.get("S2C_MAX_PIXELS", 40_000_000))`; `os.environ.setdefault("OPENCV_IO_MAX_IMAGE_PIXELS", str(MAX_PIXELS))`. `api.py`: in `analyze`, after the magic check, `_pixels(data) > MAX_PIXELS` → `HTTPException(413, TOO_MANY_PIXELS)` where `_pixels` opens the header with PIL (`Image.open` is lazy) and returns `width * height` (0 when unreadable). `install_error_handlers.body_limit`: when an `/api` request has no Content-Length, wrap `request._receive` (Starlette ASGI receive) to count `http.request` body bytes and answer 413 past `MAX_BODY` (MAX_JSON for JSON).
- [ ] **Step 4: Run** the same command — Expected: 3 passed; `tests/test_web_api.py tests/test_web_guard.py` still pass.
- [ ] **Step 5: Commit** `Uploads are capped in pixels before any decode, and a body without a length is counted as it streams`.

### Task 2: Library errors never reach the browser (H2)

**Files:** Modify `s2c/web/jobs.py`. Test: `tests/test_web_api.py`.

**Interfaces:** Produces `jobs.UserFacing(Exception)` with `.remedy: str`.

- [ ] **Step 1: Write the failing test**

```python
def test_a_library_error_in_a_sheet_job_never_reaches_the_browser(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB in C:/secret/path")
    monkeypatch.setattr("s2c.multiview.sheet_read.observe_drawing", boom)
    job = _sheet_job(monkeypatch)
    assert job["status"] == "failed" and job["error"] == jobs.FAILED
    assert "CUDA" not in json.dumps(job)
```
(`_sheet_job` posts `examples/mv/sheet/sheet.png` in sheet mode and polls to the end, as `test_analyze_sheet_mode_splits_a_drawn_sheet_and_builds` does.)

- [ ] **Step 2: Run** `uv run pytest tests/test_web_api.py -k library_error -q` — Expected: FAIL, the error is the CUDA text.
- [ ] **Step 3: Implement.** `class UserFacing(Exception)` (one `remedy` arg); `run_sheet` raises it for the two user messages (`page.remedy…`, the sketch reader's abstention) and catches `UserFacing` (not `RuntimeError`) to set `job.error = e.remedy`; every other exception keeps `FAILED`.
- [ ] **Step 4: Run** `uv run pytest tests/test_web_api.py -q` — Expected: all pass.
- [ ] **Step 5: Commit** `A sheet job shows only its own messages: a library error gives the fixed sentence`.

### Task 3: No silent switch to the other sketch reader (H7)

**Files:** Modify `s2c/web/jobs.py`, `tests/test_web_api.py:324`. Test: `tests/test_web_api.py`.

- [ ] **Step 1: Write the failing tests**

```python
def test_a_crash_in_the_drawing_reader_fails_the_job_instead_of_switching_readers(monkeypatch):
    called = []
    monkeypatch.setattr("s2c.multiview.sheet_read.read_drawing", lambda *a, **k: 1 / 0)
    monkeypatch.setattr(jobs, "_sheet_reading", lambda data: called.append(1))
    job = _sheet_job(monkeypatch)
    assert job["status"] == "failed" and job["error"] == jobs.FAILED and not called
```
and change the existing assertion at line 324 to `"2 views found by the sketch reader"` in `job["stages"][0]["detail"]`.

- [ ] **Step 2: Run** `uv run pytest tests/test_web_api.py -k "crash_in_the_drawing or reads_one_sheet" -q` — Expected: 2 FAIL.
- [ ] **Step 3: Implement.** `_drawn_sheet` drops its `try/except Exception`; the fallback's views detail reads `f"{n} views found by the sketch reader (no drawing sheet found)"`.
- [ ] **Step 4: Run** `uv run pytest tests/test_web_api.py -q` — Expected: all pass.
- [ ] **Step 5: Commit** `A crash in the drawing reader fails the job; only "no sheet found" falls back, and the job says so`.

### Task 4: Jobs are evicted by last use (H3, partial)

**Files:** Modify `s2c/web/jobs.py`, `s2c/web/api.py` (`_job`), `Dockerfile` (CMD), `README.md`. Test: `tests/test_web_jobs.py` (new).

**Interfaces:** `Job.used: float`; `jobs.touch(job)`; `MAX_JOBS = 200`.

- [ ] **Step 1: Write the failing tests**

```python
"""The in-memory job registry (audit H3): a job being reviewed outlives newer ones."""
import time

from s2c.multiview.pipeline import MvPipeline
from s2c.web import jobs


def _done(monkeypatch=None):
    job = jobs.new_job(1, MvPipeline())
    job.status = "done"
    return job


def test_the_least_recently_used_job_goes_first(monkeypatch):
    monkeypatch.setattr(jobs, "JOBS", {})
    monkeypatch.setattr(jobs, "MAX_JOBS", 3)
    first = _done()
    _done(), _done()
    jobs.touch(first)  # the user is still on Review
    _done()
    assert jobs.get_job(first.job_id) is first


def test_the_time_to_live_counts_from_last_use(monkeypatch):
    monkeypatch.setattr(jobs, "JOBS", {})
    job = _done()
    job.created = time.time() - 2 * jobs.TTL_S
    jobs.touch(job)
    jobs.sweep_jobs()
    assert jobs.get_job(job.job_id) is job
```

- [ ] **Step 2: Run** `uv run pytest tests/test_web_jobs.py -q` — Expected: FAIL (`touch` missing).
- [ ] **Step 3: Implement.** `Job.used` (default now); `touch(job)` sets it; `_register` evicts idle jobs by `used`; `sweep_jobs` by `now - used`; `api._job` and `merge` call `jobs.touch`; `MAX_JOBS = 200`; Dockerfile CMD adds `"--workers", "1"`; README Docker section: jobs live in this one process.
- [ ] **Step 4: Run** `uv run pytest tests/test_web_jobs.py tests/test_web_api.py -q` — Expected: all pass.
- [ ] **Step 5: Commit** `Jobs live an hour from their last use and the busiest hour evicts the least recently used; one worker`.

### Task 5: A per-job time budget, and a TrOCR lock that cannot be held forever (H4, partial)

**Files:** Modify `s2c/web/jobs.py`, `s2c/reading/trocr.py`. Test: `tests/test_web_jobs.py`, `tests/reading/test_trocr_lock.py` (new).

**Interfaces:** `jobs.JOB_BUDGET_S = 900`, `jobs.TOO_LONG: str`; `trocr.READ_LOCK_S = 120.0`.

- [ ] **Step 1: Write the failing tests**

```python
def test_a_job_past_its_budget_stops_with_a_plain_message(monkeypatch):
    monkeypatch.setattr(jobs, "JOB_BUDGET_S", 0)
    job = jobs.new_job(1, MvPipeline())
    jobs.run(job, MvPipeline(), [ImageInput(PNG, "front", "sketch")], None)
    assert job.status == "failed" and job.error == jobs.TOO_LONG
```
(`PNG` = `examples/mv/sketches/front.png` bytes.)

```python
"""A TrOCR read abandoned by a timed-out caller must not block every later read (audit H4)."""
import pytest

from s2c.reading import trocr


def test_a_held_read_lock_times_out(monkeypatch):
    monkeypatch.setattr(trocr, "READ_LOCK_S", 0.05)
    trocr._READ_LOCK.acquire()
    try:
        with pytest.raises(TimeoutError):
            trocr._run(None, None, "cpu", [object()], 4)
    finally:
        trocr._READ_LOCK.release()
```

- [ ] **Step 2: Run** `uv run pytest tests/test_web_jobs.py tests/reading/test_trocr_lock.py -q` — Expected: 2 FAIL.
- [ ] **Step 3: Implement.** `run`/`run_sheet` progress callbacks raise `JobTimeout` once `time.time() - job.created > JOB_BUDGET_S`; both runners catch it: `status="failed"`, `error=TOO_LONG` ("The analysis took too long. Try again with fewer images or with the AI helpers off."). `_run` takes `_READ_LOCK.acquire(timeout=READ_LOCK_S)` or raises `TimeoutError`; the reading service already turns a reader error into "error".
- [ ] **Step 4: Run** the same command plus `tests/reading -q` — Expected: all pass.
- [ ] **Step 5: Commit** `A job stops past its time budget, and a TrOCR read waits for the model a bounded time`.

### Task 6: Hosted image services are opt-in (H5)

**Files:** Modify `s2c/multiview/settings.py`, `s2c/web/api.py`, `web/src/state/store.tsx`, `web/src/screens/Capture.tsx`, `tests/test_studio_settings.py:107-108`, `README.md`, `CLAUDE.md`. Test: `tests/test_web_api.py`, `web/src/state/store.test.ts`.

- [ ] **Step 1: Write the failing tests**

```python
def test_without_ai_settings_no_image_goes_to_a_hosted_service():
    job = analyze()
    pipe = jobs.get_job(job["job_id"]).pipe
    assert not pipe.draw_faces and not pipe.rescue_enabled and pipe.mesh_provider is None and pipe.depth is None
```
`tests/test_studio_settings.py`: the default tuple becomes `(True, False, False, False, False, False)`.
`web/src/state/store.test.ts`: `expect(initialAi).toMatchObject({ use_qwen_image: false, use_rescue: false, use_triposr: false, use_solaria: false })`.

- [ ] **Step 2: Run** `uv run pytest tests/test_web_api.py -k hosted tests/test_studio_settings.py -q` and `npx vitest run src/state` — Expected: FAIL.
- [ ] **Step 3: Implement.** `AiSettings` defaults False for the four; `/api/analyze` always applies `AiSettings()` when no `ai` field is sent; `initialAi` follows; each of the four toggles gets the hint "Sends your images to a hosted service"; README "Responsible AI and data" and CLAUDE.md "Responsible AI positions" say which services receive images and when.
- [ ] **Step 4: Run** the same — Expected: pass; full web tests and `npm run build` pass.
- [ ] **Step 5: Commit** `Images go to hosted services (Qwen-Image, TripoSR, Solaria) only when the user turns them on`.

### Task 7: CI checks what ships; call logs stay out of git (H8, M1)

**Files:** Create `s2c/logdir.py`. Modify `s2c/vision/client.py`, `s2c/multiview/label.py`, `s2c/multiview/depth.py`, `s2c/multiview/qwen_image.py`, `s2c/pipeline.py`, `s2c/web/chat.py`, `tests/conftest.py`, `pyproject.toml`, `.github/workflows/ci.yml`, `.gitignore`; untrack `logs/vlm.jsonl`, `logs/vlm_fake.jsonl`. Test: `tests/test_vision_client.py`.

**Interfaces:** `s2c.logdir.log_file(name: str) -> Path` (`S2C_LOG_DIR`, default `logs`, read at call time).

- [ ] **Step 1: Write the failing test**

```python
def test_the_default_log_goes_to_the_log_directory(tmp_path, monkeypatch):
    monkeypatch.setenv("S2C_LOG_DIR", str(tmp_path / "calls"))
    VLMClient(chat=lambda m: "{}", model="fake-model").complete_json("hi", b"\x89PNG")
    assert (tmp_path / "calls" / "vlm.jsonl").is_file()
```
(use the client's real call method name, as `test_every_call_is_logged` does.)

- [ ] **Step 2: Run** `uv run pytest tests/test_vision_client.py -q` — Expected: FAIL (written under ./logs).
- [ ] **Step 3: Implement.** `log_file`; every default log path becomes `None` resolved through `log_file` at call time; `chat.LOG_PATH` becomes a function; `conftest` sets `S2C_LOG_DIR` to a temp folder for every test; `.gitignore` gets `logs/*` and `!logs/.gitkeep`; `git rm --cached logs/vlm.jsonl logs/vlm_fake.jsonl`. `pyproject.toml` `addopts = "-m 'not gpu and not network and not slicer and not blender and not dataset'"`. CI: `uv sync --locked`; a `web` job (`actions/setup-node@v4` node 20, `npm ci`, `npm run build`, `npm test` in `web/`); a `docker` job (`docker build .`); an `audit` job (`uv export --frozen --no-dev --no-hashes -o req.txt && uvx pip-audit -r req.txt`, `npm audit --audit-level=high` in `web/`) with `continue-on-error: true`.
- [ ] **Step 4: Run** `uv run pytest -q tests/test_vision_client.py tests/test_web_chat.py tests/test_mv_qwen_image.py tests/test_mv_depth.py` — Expected: pass; `git status` shows no `logs/` change after the run; `uv run pytest tests/test_mv_hf3d.py -q` shows the GPU test deselected.
- [ ] **Step 5: Commit** `CI builds and tests the web app and the image on a locked environment; call logs go to S2C_LOG_DIR, out of git`.

### Task 8: The docs describe the code that runs (H9)

**Files:** Modify `README.md`, `CLAUDE.md`. Test: `tests/test_docs.py` (new).

- [ ] **Step 1: Write the failing test**

```python
"""Every file, module and command the README and CLAUDE.md name exists (audit H9)."""
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PATH = re.compile(r"`((?:s2c|scripts|tests|web|docs|examples|profiles)/[\w./-]*\w|app_\w+\.py)`")
APP = re.compile(r"uvicorn ([\w.]+):app")


@pytest.mark.parametrize("doc", ["README.md", "CLAUDE.md"])
def test_every_named_path_exists(doc):
    text = (ROOT / doc).read_text(encoding="utf-8")
    missing = [p for p in PATH.findall(text) if not (ROOT / p).exists() and "<" not in p]
    missing += [m for m in APP.findall(text) if not (ROOT / (m.replace(".", "/") + ".py")).exists()]
    assert missing == []
```

- [ ] **Step 2: Run** `uv run pytest tests/test_docs.py -q` — Expected: FAIL listing `s2c/api.py`, `app_gradio.py`, `s2c.api`, …
- [ ] **Step 3: Implement.** README: "How it works" shows the served multi-view path (capture or one sheet → observe → fuse → MultiViewSpec → Review → build → export); "Run it" drops `s2c.api` and `app_gradio.py`; "Repository layout" lists `s2c/web`, `s2c/multiview`, `s2c/reading`, `s2c/sketch`, `s2c/studio`, and the single-view modules as "not on the served path"; the status section is dated today. CLAUDE.md: "Pipeline and ownership" and "Stack and commands" match; the rules stay.
- [ ] **Step 4: Run** `uv run pytest tests/test_docs.py -q` — Expected: pass.
- [ ] **Step 5: Commit** `The README and CLAUDE.md describe the served path and name only files that exist`.
