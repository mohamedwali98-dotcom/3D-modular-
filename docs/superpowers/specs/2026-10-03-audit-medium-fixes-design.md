# Audit Medium-priority fixes — design

Date: 2026-10-03. Source: the codebase audit of 2026-10-02, Medium items M2–M12 (M1, call logs out of git, landed
with the High fixes in `1754518`). Approved under the user's standing instruction ("if there is a question pick the
recommended one").

## Goal

Pay down the routine-refactoring debt the audit listed: one validated place for settings, one pipeline per process,
no filesystem scan on every poll, a fuse that guards its own state, typed edits, public module boundaries, counted
fallbacks and real logs and metrics, honest dependency floors, frontend models generated from the backend, and
strict expected failures.

## Design

1. **Public boundaries (M7).** Helpers used across modules get public names: `outline.stroke_width`,
   `build.clean_solid`, `label.strip_fences`, `sheet.remove_border`. Test builders shared between test modules
   move to `tests/builders.py`. A test fails when any module imports another module's private name.
2. **Typed edits (M6).** `s2c/multiview/edits.py` parses the `user_values` map once: `envelope.{x,y,z}_mm`,
   `features[k].<numeric field>` and `features[k].keep`. Anything else is refused at `/api/merge` with a plain
   sentence, never silently ignored or written into a feature. `spec.feature_path(i, name)` and
   `spec.finish_path(i, name)` build every provenance key on the served path.
3. **Fuse guards its own state (M5).** `Observed` owns a lock; `fuse` and `forget_images` take it, so no caller
   needs `merge_lock` (dropped from jobs). The image retention policy is `MvPipeline.needs_images` (High round).
4. **One pipeline, a calm sweep (M3, M4).** `api.get_pipeline` builds the pipeline once under a lock and the server
   builds it at startup (lifespan). The `/api` sweep of old jobs and artifact folders runs at most once a minute.
5. **Observability (M8, M9).** `s2c/obs.py`: logging configured once (JSON lines, or text with
   `S2C_LOG_FORMAT=text`; `S2C_LOG_LEVEL`), every record of an analysis carrying its `job_id`; counters and timings
   (`s2c_jobs_total`, `s2c_stage_seconds`, `s2c_fallbacks_total`, `s2c_refused_total`) rendered in Prometheus text at
   `/api/metrics` (guarded like the rest of `/api`). Every provider fallback goes through `obs.fallback(name, exc)`,
   which logs and counts it. Catches at provider boundaries stay broad (third-party clients raise anything).
6. **Settings (M2).** `s2c/config.py` declares every environment variable the code reads (name, kind, default,
   meaning) and checks the environment at server start: a bad number or an unknown choice stops the server with
   one message listing each problem. A test fails when the code reads a variable the registry or `.env.example` does
   not list. Working folders (`tmp/…`, `logs/`) resolve under `S2C_DATA_DIR`, by default the project folder in a
   source checkout and the working directory otherwise.
7. **Dependencies (M10).** Floors say what is tested: `transformers>=5` (TrOCR), one torch floor. The TripoSR clone
   is pinned to the tested commit and appended to `sys.path` instead of shadowing installed packages. Dependabot
   watches uv, npm and the workflow actions. The README says local TripoSR needs transformers below 5.
8. **Generated frontend models (M11).** `scripts/gen_ts_models.py` writes `web/src/api/models.gen.ts` from the
   Pydantic models the API shares with the web app (`MultiViewSpec` and its parts, `MvAbstain`, the settings);
   `types.ts` re-exports them. A test fails when the generated file is stale.
9. **Strict expected failures (M12).** The two known-limit `xfail` marks become `strict=True`.

## Out of scope (rulings)

- **Server-sent events instead of polling (M4).** The poll stays at 400 ms; with the sweep throttled each poll is a
  dict lookup and a small JSON. Cost if wrong: more requests than needed under many users.
- **A fully immutable `Observed` (M5).** The lock makes callers safe; splitting the caches out is churn without a
  failure mode. Cost if wrong: fuse still mutates what it is given, behind its own lock.
- **Generated types for the dict-built responses** (job, analysis, model and export results, chat). They need
  response models on every route first. Cost if wrong: those few types can still drift.

## Testing

Each item lands test-first; the full suite, the web tests and build, and the golden gate stay green.
