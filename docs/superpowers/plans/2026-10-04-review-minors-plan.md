# Review Minors (Medium round) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:executing-plans. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Fix the eight minors the final review of the audit's Medium round deferred, then keep a measure → fix → test loop running until 08:00 on 2026-10-04.

**Architecture:** Each fix is local to the module the review named. Tests first (RED), then the change (GREEN), then the full suite. One commit per task.

**Tech Stack:** Python 3.11, FastAPI, pytest, ruff (line length 120).

**Spec:** the final review of 5feaf64..37032ae (minor findings M3–M10), recorded in the user's release notes ("Still open"). The user asked on 2026-10-03: "fix the deferred minors ... don't stop and for any question pick the recommended one".

## Global Constraints

- No AI attribution in commits. Never read or write `.env`. No secrets or personal data in `tests/`.
- Push with `git push modular geometry/complex-parts:main` at the end of each loop cycle.
- `uv run pytest -q -p no:cacheprovider` must stay green; `uv run ruff check .` clean.

## Review Focus

- A refused merge path must not change any analysis state (the job stays mergeable).
- Metrics stay valid Prometheus text: one `# TYPE` per family, family lines contiguous.
- A job id must never leak to a record logged outside that analysis (contexts copied per task, never shared).
- The `.env` the settings read at import is the file the entry points load.
- Logging with an operator's own root handler must still give the app's lines a job id.

---

### Task 1: Merge refuses what it cannot write (review M3)

**Files:** `s2c/multiview/fuse.py` (assemble), `s2c/web/api.py` (merge). Test: `tests/test_web_api.py`.

- [ ] Test: on a done analysis, `features[0].length_mm` on a hole and `features[99].a_mm` answer 400 naming the path; the job still merges afterwards.
- [ ] `assemble` raises `edits.EditError` for an index past the feature list or a field the feature does not have; `/api/merge` maps it to 400.

### Task 2: Metrics that add up (review M4, M6, M7)

**Files:** `s2c/web/jobs.py` (reduce), `s2c/obs.py` (render), `s2c/reading/service.py`, `s2c/reading/trocr.py`, `s2c/multiview/hf3d.py`. Tests: `tests/test_obs.py`, `tests/test_web_jobs.py`, `tests/reading/test_service.py`.

- [ ] Test: a per-image stage is timed from its own image's start (two images: samples 1 s and 2 s, not 1 s and 3 s).
- [ ] Test: `render()` prints `# TYPE <family> counter|summary` once per family, before its lines.
- [ ] Test: a reader timeout counts `reader_<name>_timeout`; TrOCR's move to CPU counts `trocr_gpu`; a TripoSR failure counts `triposr_local` (local → Space) and the caller's `triposr`, never `triposr_space`.

### Task 3: Every record of an analysis carries its id (review M5)

**Files:** `s2c/obs.py` (`carry`), `s2c/web/jobs.py` (merge), `s2c/web/api.py` (model), `s2c/reading/service.py`, `s2c/multiview/hf3d.py`. Tests: `tests/test_obs.py`, `tests/test_web_api.py`.

- [ ] Test: a record logged inside a merge, inside a model build for a known request id, and inside a reader thread carries the job id; one logged after does not.
- [ ] `obs.carry(fn)` runs fn in a copy of the caller's context; the reader pool and hf3d submit through it; merge and model run inside `job_scope`.

### Task 4: One `.env` for import-time and runtime settings (review M8, M9)

**Files:** `s2c/config.py` (`dotenv_path`), `s2c/web/server.py`, `s2c/studio/app.py`, `app_mv_gradio.py`, `scripts/mv.py`, `scripts/sketch_accuracy.py`, `README.md`. Tests: `tests/test_config.py`, `tests/test_docs.py`.

- [ ] Test: with the working directory elsewhere (its own `.env`), `config.setting` reads the project's `.env`, the same file the entry points load.
- [ ] Test: README names `/api/metrics` and gives the log folder as `<data dir>/logs`.

### Task 5: An operator's logging config wins (review M10)

**Files:** `s2c/obs.py`. Test: `tests/test_obs.py`.

- [ ] Test: when root already has a handler, `configure_logging()` adds none, and that handler's records carry `job_id`.
