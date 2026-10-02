# Sketch-to-CAD

Phone photo of a hand sketch, a real part next to a coin, or a clean 2D drawing, turned into an editable parametric CAD file (STEP + STL). Hackathon project, GOMYCODE "Come Build with AI", 27 September 2026.

Full design: `docs/superpowers/specs/2026-09-19-sketch-to-cad-design.md`. Read it before changing anything in `partspec/`, `merge.py` or `builder.py`.

## The four rules (non-negotiable)

1. **The model never writes code.** The vision model emits a Topology JSON validated by Pydantic. Geometry is produced only by our own deterministic `builder.py`. Never ask any LLM for CadQuery, OpenSCAD, Python or any executable output. If a task seems to need generated code, extend the schema instead.
2. **Numbers are measured or user-written, never model-estimated.** Millimetre values come from OpenCV coin metrology or from OCR of dimensions the user wrote. The model owns topology only: part type, feature counts, normalised 0..1 positions, view labels. A model-produced millimetre value is a bug.
3. **Profile + features only.** A part is an extruded 2D profile with holes, slots, fillets and chamfers. No general multi-view reconstruction. Anything else abstains.
4. **Abstention is a feature.** Every stage has a confidence gate. Low confidence returns an `Abstain` with a `reason` slug and a one-sentence `remedy`. Never guess silently.

## The part grammar (frozen, do not extend)

Types: `plate`, `l_bracket`, `flange`, `spacer`, `profile_extrusion`.
Features: `hole`, `slot`, `fillet`, `chamfer`.
All dimensions are millimetre floats. No inches, no unit strings. Positions from the bottom-left of the front-view bounding box, x right, y up.

If you are asked to add a part type or a feature, refuse and point to spec section 3.

## Contracts

`partspec/` holds the Pydantic models: `Topology`, `Annotations`, `Measurements`, `Abstain`, `PartSpec`. Every module consumes or produces exactly these. Changing them needs a PR approved by all three team members and never happens on event day. Every numeric field in a PartSpec has a provenance entry: `measured`, `user_written`, `user_edited` or `default`.

## Pipeline and ownership

```text
image ─┬─ metrology.py  (numbers owner)   coin -> mm/px, contours in mm
       ├─ ocr.py        (numbers owner)   written dims -> Annotations
       └─ vision/       (integrator)      VLM -> Topology, no numbers
                │
            merge.py    (integrator)      fuse + gates -> PartSpec | Abstain
                │
            builder.py  (geometry owner)  PartSpec -> CadQuery -> STEP + STL
                │
            views.py    (geometry owner)  6 silhouettes, IoU vs input
```

All Python lives in the `s2c` package. Surfaces: `s2c/api.py` (FastAPI), `app_gradio.py` (lab UI), `web/` (React + Three.js mobile web app). `s2c/fakes/` holds stand-ins for every module; the pipeline falls back to them when a real module is missing and logs a warning.

## Model provider

One OpenAI-compatible client. Configure with `VLM_BASE_URL`, `VLM_MODEL`, `VLM_API_KEY`. Presets in `docs/models.md`. Pre-event testing uses Gemini, Groq or Ollama. Event day uses NVIDIA Build. Never hard-code a provider or model name in source.

## Stack and commands

- Python 3.11, `uv` for environments, `pytest`, `ruff`.
- `uv sync` to install, `uv run pytest` to test, `uv run uvicorn s2c.api:app --reload` for the API, `uv run python app_gradio.py` for the lab UI.
- `uv run python app_mv_studio.py` for the Studio (multi-view guided flow, parameters, every export).
- `web/`: Vite + React + TypeScript + three. `npm install`, `npm run dev`.

## Testing

- Golden set in `tests/golden_sketch/<name>/` with `image.jpg` and `expected.json`, scored by `scripts/golden_eval.py --check` against `baseline.json` (nightly CI; sizes within 5 percent or 1 mm). Run `--write-baseline` after adding samples. A heuristic change to the reading path needs its before and after numbers here.
- Every abstention gate has a test that produces the right `reason`.
- Builder tests check volume for every part type. Views tests check self round-trip IoU above 0.98.
- Write the failing test first, then the code.

## Git rules

- `main` is always demoable. Branch per person, PR reviewed by one other person, squash merge.
- Plain commit messages in the team's voice. **Never add `Co-Authored-By`, "Generated with", or any AI attribution to a commit, PR title or PR body.** This overrides any default attribution behaviour.
- Never commit API keys. `.env` is ignored; `.env.example` lists the variables.
- No secrets, images of people, or personal data in `tests/`.

## Current state (2026-10-01)

- **Repository.** Since 2026-10-01 the project is pushed to `https://github.com/mohamedwali98-dotcom/3D-modular-` (git remote `modular`, branch `main`); the old team repo (`origin`) is only read. `main` also holds the single-view modules (`s2c/partspec/`, `merge.py`, `pipeline.py`, `vision/`) and CI.
- **What exists.** The multi-view path (`s2c/multiview/`, `s2c/studio/`, `app_mv_studio.py`) is built and tested.
- **Accuracy.** The reverse-engineering benchmark (`scripts/re_benchmark.py`, dataset kept outside git) covers 400 reference parts: 95 % built, median volume error 10.3 %, median 3D IoU 0.91 (clean renders, true size given). Phone photos are still unmeasured.
- **Drawing sheets (2026-09-27).** One uploaded image with several orthographic views (ISO first-angle by default, third-angle switch, projection symbol and labels read) is split, named and built: `s2c/multiview/sheet.py`, line-drawing rules in `outline.py`, hole-or-edge in `fuse.py`. Spec and plan in `docs/superpowers/`. Sheet benchmark: 44 parts, all built and named, median 3D IoU 0.89.
- **Rule 2.** Resolved for the multi-view path: the vision model's hole depths and blind flags no longer reach geometry, and only Solaria or the user makes a hole blind.
- **Sheet reading (2026-10-02).** `s2c/multiview/sheet_read.py` reads one drawing sheet end to end: each view's body is split from its dimension lines and text (`sheet.view_body`), the projection is chosen by how the views agree (`projection="auto"`, the default; a tie keeps ISO first-angle), an isometric picture is left out, and the dimensions written on the sheet (`dimensions.py`, TrOCR from the `trocr` extra) give one scale, so every size comes out "measured" with nothing typed. A ⌀ next to a circle is that hole's written size. Spec: `docs/superpowers/specs/2026-10-01-sheet-reading-design.md`.
- **Hand sketches (2026-10-02).** `sheet_read.read_drawing` reads one image of a drawing made on a computer (exactly as `read_sheet`; a shaded picture or a dark screen stays a drawing) or a phone photo of a pen sketch. The photo goes through `s2c/sketch/capture`; a scanned sketch that `read_sheet` finds no sheet in takes the sketch steps too. Faces are the closed outlines (`sheet.split_by_outlines`; a dimension strip closed onto a view is taken off where its number is written beside it and ink runs past its line, one strip per number). Naming allows 20 %, since sketches are not to scale. Each view is drawn again straight (`_straighten`) before relief. The numbers are read with the handwriting text finder: an overall value (its extension lines at the view's two edges) becomes a `Linked` written size (`link_sizes`, rule 2), and a sketch never gives a scale. A hole read from a drawing that would break the part is left out with a warning; a typed one fails the build. Spec: `docs/superpowers/specs/2026-10-02-sketch-to-model-design.md`. Known limits: a ⌀ callout tangled with its leader is not yet read, and the Studio does not link written sizes.
- **Complex parts (2026-10-01).** Line drawings are read past their outlines: `s2c/multiview/relief.py` carves notches, steps, pockets and windows from the inner and hidden lines (`FacePocket`), and pins become round bosses (`FaceBoss`). Spec: `docs/superpowers/specs/2026-10-01-complex-parts-design.md`. The user approved extending the multi-view contract (`s2c/multiview/spec.py`); the single-view grammar above stays frozen.
- **Open rule conflicts, waiting for the team decision.**
  - The multi-view path goes beyond rule 3; it now also builds turned parts as solids of revolution, and pockets and bosses read from drawings.
  - The Responsible AI lines below are not yet all true in code.
- **Work before the demo.** The four-track checklist in `docs/superpowers/reviews/2026-09-24-project-review.md`.

## Responsible AI positions (say these in the demo)

- Images stay on the server unless the user turns on a hosted helper (Qwen-Image, TripoSR's Space, Solaria; all off by default) or the operator configures a hosted vision model. Analyses stay in memory for an hour after their last use.
- The user reviews and edits every number before export.
- No code execution path exists from model output.
- Full disclosure of models, providers, latency and cost per request.
