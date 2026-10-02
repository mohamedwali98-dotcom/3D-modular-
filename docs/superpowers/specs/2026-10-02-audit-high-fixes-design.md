# Audit High-priority fixes — design

Date: 2026-10-02. Source: the codebase audit of 2026-10-02 (High items H1–H5, H7–H9, plus M1). The Critical items
(C1–C3) are fixed in `730652a`, `21afd9d` and `8fb970b`. Approved under the user's standing instruction ("if there is
a question pick the recommended one").

## Goal

Make the served app (`s2c.web.server`) safe to leave running: bounded memory per upload, no internal error text in
the browser, no silent algorithm switch, jobs that survive a busy hour, a per-job time budget, images sent to hosted
services only when the user opts in, a CI that checks what ships, logs that stay out of git, and docs that describe
the code that runs.

## Out of scope (rulings)

- **H6, one application service for all UIs.** A restructure of the web job, the Studio and the lab app; too large to
  land unattended. The shared `sheet_read.observe_drawing` (C3) is the first step.
- **H3, durable job state (Redis, workers).** Needs infrastructure. This round keeps jobs in memory, evicts by last use
  instead of age, and pins one worker.
- **Deleting the single-view modules** (`s2c/pipeline.py`, `merge.py`, `fakes/`, `store.py`). They are team-owned
  contracts (CLAUDE.md): documented as "not on the served path", not removed.

## Design

1. **Images (H1).** `s2c/__init__.py` sets `OPENCV_IO_MAX_IMAGE_PIXELS` (default 40 MP) before any `cv2` import, so
   every decode in the process is capped. `/api/analyze` reads each upload's header with PIL and answers 413 above
   `MAX_PIXELS` before decoding. A request body without `Content-Length` (chunked) is counted as it streams and cut
   at the same limit as a declared one.
2. **Errors (H2).** A `UserFacing(remedy)` exception carries every message meant for the browser in a sheet job. Any
   other exception, `RuntimeError` included (a CUDA out-of-memory error is one), gives the fixed `FAILED` sentence.
3. **No silent reader switch (H7).** `_drawn_sheet` no longer catches everything: a crash in `read_drawing` fails the
   job (logged with its traceback). Only "no sheet of views found" (None) falls back to the team's sketch reader, and
   the views stage then says so.
4. **Jobs (H3, partial).** A job's `used` time is refreshed on every status poll and merge; the registry evicts the
   least recently used finished jobs (`MAX_JOBS` raised to 200) and the TTL counts from last use. The Docker command
   pins `--workers 1`; the README says why.
5. **Time budget (H4, partial).** A job past `JOB_BUDGET_S` (900 s) stops at its next progress event with "The analysis
   took too long…". TrOCR takes its read lock with a timeout, so a read abandoned by a timed-out caller cannot block
   every later read.
6. **Hosted services opt-in (H5).** `AiSettings` defaults `use_qwen_image`, `use_rescue`, `use_triposr` and
   `use_solaria` to False (the web store and the Studio follow); an `/api/analyze` call without AI settings uses those
   defaults too. Each toggle says the images go to a hosted service. The Responsible AI lines in README and CLAUDE.md
   say what is true.
7. **CI and logs (H8, M1).** CI: `uv sync --locked`, a web job (`npm ci`, build, test), a Docker build, and a
   dependency audit. pytest deselects `gpu`, `network`, `slicer`, `blender` and `dataset` tests by default
   (`-m gpu` still runs them), so `uv run pytest` is green on a GPU machine. Call logs go to `S2C_LOG_DIR` (default
   `logs/`), resolved at call time; the tests send them to a temp folder; `logs/*.jsonl` leave git.
8. **Docs (H9).** README "How it works", run commands, layout and status, and CLAUDE.md's surfaces, pipeline and
   commands describe the served multi-view path; the single-view modules are listed as not served.

## Testing

Each item lands test-first: a 40 MP+ PNG refused before decode and a chunked body cut; a library `RuntimeError` never
reaching `job.error`; a crashing `read_drawing` failing the job instead of switching readers; a polled job surviving
eviction; a job past its budget stopping; a held TrOCR lock timing out; the AI defaults and the HTTP default; the
log directory honoured. The full suite, the web tests and the golden gate stay green.
