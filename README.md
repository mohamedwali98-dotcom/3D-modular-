# Sketch-to-CAD

> **Demo video:** open [`docs/demo/S2C DEMO VD.docx`](docs/demo/S2C%20DEMO%20VD.docx). This file gives you access to the demo video: it holds the link to the 90-second video ([watch on Google Drive](https://drive.google.com/file/d/1F_BJ6ZdxtqHqYxYnIAMBgpvA5OWEruSv/view?usp=sharing)).

Point your phone at a broken part or a hand-drawn sketch of one. Get an editable, parametric CAD file back, ready to print or machine.

Built for the GOMYCODE "Come Build with AI" hackathon, 27 September 2026, on NVIDIA Build.

## The problem

A bracket snaps, a spacer goes missing, a plastic clip breaks. The replacement does not exist or costs more to ship than to print. Modelling it in CAD takes an hour you do not have, and mesh generators give you a blob you cannot edit or trust.

## What it does

Three inputs, one pipeline, one output.

| Input | How | Where the numbers come from |
| --- | --- | --- |
| Hand sketch | Draw it on paper, write the dimensions in mm, photograph it | Your handwriting, read by OCR |
| Real part | Photograph it top-down next to a coin | Coin scale, measured with OpenCV |
| 2D drawing | A clean orthographic view with dimensions | Printed dimensions, read by OCR |

Output: a STEP file for CAD tools and an STL file for slicers, plus a live 3D preview and sliders that re-derive the geometry when you change a number.

## How it works

```text
image ──┬─ metrology   OpenCV: coin -> mm per pixel, contours in mm
        ├─ ocr         written dimensions -> values linked to edges and holes
        └─ vision      vision model -> topology only: part type, hole count, rough positions
                │
             merge      fuse + confidence gates -> PartSpec, or a clear abstention
                │
        PartSpec        the single source of truth, edited by sliders
                │
             builder    our own deterministic CadQuery code -> STEP + STL
                │
             views      six orthographic silhouettes -> round-trip match against the input
```

Four rules shape every design decision:

1. **The model never writes code.** The vision model returns a small JSON describing topology. A schema validates it before anything is built. There is no code execution path from model output.
2. **Numbers are measured or written, never estimated.** Every millimetre comes from the coin scale or from what the user wrote. The model is not allowed to produce a dimension; if it does, the value is discarded.
3. **Profile plus features.** A part is a 2D outline extruded straight, with holes, slots, fillets and chamfers. That covers plates, brackets, flanges, spacers and free-form outlines, and it is honest about what it cannot do.
4. **Abstention is a feature.** Tilted coin, unsupported shape, missing thickness, poor round-trip match: each one stops with a reason and tells you what to do next.

Supported part types: `plate`, `l_bracket`, `flange`, `spacer`, `profile_extrusion`. Features: `hole`, `slot`, `fillet`, `chamfer`. All dimensions in millimetres.

## Run it

Requirements: Python 3.11, [uv](https://docs.astral.sh/uv/), Node 20.

### Docker (recommended)

Docker runs the built React app and FastAPI backend together on one port. The local `.env` is loaded at runtime but is never copied into the image. If `.env` is missing, the offline path still works.

```bash
docker compose up --build
```

Open `http://localhost:8000`. To use another host port, set `APP_PORT` when starting Compose (for example, `APP_PORT=8080 docker compose up --build` in Bash or `$env:APP_PORT=8080; docker compose up --build` in PowerShell).

The container automatically translates an Ollama URL using `localhost` or `127.0.0.1` to `host.docker.internal`, so the same `.env` works both locally and in Docker. Hosted model URLs are unchanged. Generated files and call logs use Docker volumes; the app still removes generated artifacts after one hour.

Useful commands:

```bash
docker compose ps
docker compose logs -f app
docker compose down
```

Use `docker compose down --volumes` only when you also want to delete the generated-file and log volumes.

### Local development

```bash
cp .env.example .env            # add a vision model key, see docs/models.md
uv sync --extra ai --extra trocr  # the extras bring TrOCR, which reads the dimensions written on drawings
uv run pytest                   # everything green before you start
```

A plain `uv sync` leaves the extras out (and removes them if they were there): drawings then still split, name and build, but their dimensions are not read and the sizes must be typed.

**Demo, one port.** Build the web app once, then the API serves it at `/`:

```bash
cd web && npm install && npm run build && cd ..
uv run uvicorn s2c.web.server:app --host 0.0.0.0 --port 8000
```

Open `http://localhost:8000`, or `http://<your-LAN-IP>:8000` on a phone on the same network. Rebuild `web/` after any frontend change.

**Development, two processes, hot reload.** Vite on :5173 proxies `/api` to the API on :8000:

```bash
uv run uvicorn s2c.web.server:app --port 8000 --reload     # terminal 1
cd web && npm run dev                                       # terminal 2, open http://localhost:5173
```

On Windows, `powershell scripts/dev.ps1` starts both (Ctrl+C stops both). Other entry points:

```bash
uv run uvicorn s2c.api:app --port 8002        # the single-image API
uv run python app_gradio.py                   # lab view on :7860
```

Without any model keys the app still runs end to end: the offline path traces the outlines, skips reading, and asks you to type the overall size on the Review screen. `GET /api/status` shows which providers are configured.

The vision model is chosen by three environment variables: `VLM_BASE_URL`, `VLM_MODEL`, `VLM_API_KEY`. Any OpenAI-compatible endpoint works. The table in `docs/models.md` lists NVIDIA Build, Gemini, Groq and Ollama presets.

## Multi-view path (pending team sign-off)

Give one or more images per face, several of the same face if you have them: they are aligned and voted into one cleaner outline. Qwen-VL reads the numbers you wrote. Faces you did not give are drawn by Qwen-Image and kept only if they agree with the faces you did give; otherwise TripoSR, otherwise a rectangle. Solaria's depth map tells through holes from blind ones. The part is the intersection of the three extruded outlines, sliced to G-code. Designs: `docs/superpowers/specs/2026-09-22-multiview-gcode-design.md` and `docs/superpowers/specs/2026-09-23-qwen-solaria-design.md`.

**One image, five steps** (`s2c/multiview/sheet_read.read_drawing`; spec `docs/superpowers/specs/2026-10-02-sketch-to-model-design.md`). Give one drawing made on a computer, one scan or one phone photo of a pen sketch, holding several views:

1. **Page:** a photo becomes a clean page (the sheet found, flattened, the ink binarised).
2. **Faces:** each closed outline is a face. Dimension, miter and centre lines never join two views.
3. **Labels:** the faces are named by the projection symbol, then the labels, then how the views agree, with ISO first-angle when unsure.
4. **The rest:** a sketch's wobbly strokes are drawn again straight before notches, pockets and holes are read.
5. **Numbers:** TrOCR reads them. A value whose dimension line spans a whole view is that size, written by you.

A sketch is never taken as drawn to scale: any size no number gives is asked for in Review. The web app's one-sheet mode and the Studio use this path.

    uv run python app_mv_studio.py                                                           # the Studio on :7860 (guided flow, parameters, every export)
    uv run python app_mv_gradio.py                                                           # the simple lab app
    uv run python scripts/mv_build.py examples/mv/l_bracket.json --out tmp/mv_demo          # spec -> STEP, STL, G-code
    uv run python scripts/mv_export.py examples/mv/l_bracket.json --format stl --format step --format pdf  # spec -> chosen formats + zip
    uv run python scripts/mv.py --image front.jpg@front@sketch --image top.jpg@top@sketch   # images -> the same
    NETWORK_TESTS=1 uv run pytest tests/test_mv_network.py -v                               # live check of the hosted models

Settings are in `.env.example`: Qwen-VL and Qwen-Image on DashScope or Hugging Face, Solaria on Hugging Face. Photos of real parts: shoot top-down with the part lying flat.

G-code needs PrusaSlicer: `winget install --id Prusa3D.PrusaSlicer -e` (needs admin), or unzip the portable zip from the PrusaSlicer GitHub release into `vendor/`. Without it you still get STL and STEP.

Export formats: STL, STEP, 3MF, OBJ, GLB, PLY, BREP, Blender, DXF/SVG/PDF drawing, G-code, and a zip with a manifest. Blender: set `BLENDER_PATH`, or run `scripts/setup_blender.ps1`; without it the download is a Blender kit.

TripoSR (the local fallback when Qwen-Image cannot complete a face): `powershell scripts/setup_triposr.ps1` installs it; without it, that face falls straight to a rectangle.

## Repository layout

```text
s2c/                Python package
  partspec/         frozen contracts: PartSpec, Topology, Annotations, Measurements, Abstain
  vision/           provider-agnostic client, prompts, topology extraction
  merge.py          fuse stage outputs into a PartSpec, confidence gates
  builder.py        PartSpec -> CadQuery solid -> STEP + STL
  views.py          six silhouettes for the round-trip check
  ocr.py            dimension reading and linking
  metrology.py      coin scale and measured contours
  silhouette.py     input silhouette, mask normalisation, IoU
  api.py            FastAPI surface
  fakes/            stand-ins for every stage so the pipeline runs before a module lands
app_gradio.py       lab UI showing every stage output
web/                React + Three.js mobile web app
tests/              pytest suite and the golden set of ground-truth parts
docs/
  superpowers/specs/   the design spec
  superpowers/plans/   one implementation plan per owner
  roles/               one brief per team member
  models.md            provider presets
  disclosure.md        tools, models and data used
```

## Testing and reliability

- **Golden set.** Ten hand-drawn sketches and five coin photos of parts with known dimensions. A test runs the full pipeline and checks every number within 5 percent or 1 mm.
- **Round trip.** The built solid is re-projected and compared with the input silhouette. Below 0.85 IoU the result is shown in amber with a warning.
- **Builder.** Volume tests for every part type and feature.
- **Abstention tests.** Each gate has a test that produces the right reason.
- **Call log.** Every model call records provider, model, latency and tokens for the disclosure.

Accuracy numbers, updated as tests land:

| Metric | Value |
| --- | --- |
| Golden sketches passing | pending, golden set not yet built |
| Coin scale error | pending |
| OCR value accuracy | pending |
| Reference parts, multi-view path (400 parts with known STLs, clean renders, true size given; 2026-09-26) | built 95 % (381 of 400), median volume error 10.3 %, median 3D IoU 0.91, 31 % of parts within 5 % volume. Before the fixes of 2026-09-26: 89 %, 15.6 %, 0.87, 21 %. Not a phone-photo number: that is still unmeasured |
| Drawing sheets (one first-angle line-art sheet per part: front, top and right views, labels on half; 44 reference parts, 4 per category; projection on Auto; 2026-10-02) | built 100 % (44 of 44), views named correctly 100 %, median volume error 13.1 %, median 3D IoU 0.895 (0.889 before notches and pockets were read from the inner lines). Hinges 0.77 -> 0.97. Weakest: brackets (IoU 0.59, round outlines, so the inner lines are not read) and one bearing holder whose rendered shading lines read as steps (0.81 -> 0.75) |
| Real hand sketch (`tests/golden_sketch/real_bracket_1`, a phone photo of a pen sketch, TrOCR only; 2026-10-02) | 3 faces found and named. 3 of the 10 written values read exactly (60, 25, 3), and 2 more with the right number but without their ⌀. Width and depth come from the written numbers; the height is asked for (a misread 7 is flagged, not used). The part builds |
| Median sketch-to-STL latency | pending full-pipeline measurement; one measured vision-model call (`gemma3:4b` via Ollama) took about 17.8 s, logged in `logs/vlm.jsonl` — this is a single sample of model latency only, not a pipeline median |

Reference-part benchmark:
- **Command:** `uv run python scripts/re_benchmark.py --dataset <folder> --jobs 6 --timeout 300`. Add `--sheet --per-category 4` for the drawing-sheet mode.
- **Dataset:** about 400 printable parts with STLs and six renders each. It is kept outside the repository because of its non-commercial licences.
- **What is measured:** the multi-view path gets the six renders, tagged by face, and the true size typed in. Each rebuilt part is scored against its STL, and the output is a per-category table.
- **Main remaining error:** material that no view can see (open boxes and trays, three-plate corner brackets, blind pockets). The visual hull fills it in, which is why brackets (median 24 %) and bearing holders (17 %) are worst, and washers (2.8 %) and shaft collars (2.3 %) best.

## Responsible AI and data

- Images are processed in memory. One silhouette PNG and the exported files live in a temp folder for one hour, then they are deleted.
- The user sees and can edit every number before export. Each value carries a badge saying where it came from: measured, written, edited, or a default guess.
- No model output is ever executed.
- Models, providers and tools are listed in `docs/disclosure.md`.

## Team

Three people, three owners. The integrator owns the contracts, model layer, merge, API and UIs. The geometry owner owns the builder, views and the golden parts. The numbers owner owns coin metrology and dimension OCR. Briefs for each role are in `docs/roles/`.

## Status

Updated 2026-09-26:

- **Single-view path:** Real code exists for the contracts (`s2c/partspec/`), the temp file store (`s2c/store.py`), the vision client and topology extraction (`s2c/vision/`), the sketch-path merge with its abstention gates (`s2c/merge.py`), and silhouette handling (`s2c/silhouette.py`). Fakes in `s2c/fakes/` allow the pipeline to run before all modules land. CI (`.github/workflows/ci.yml`) runs `ruff check` and `pytest` on push and PR.
- **Multi-view path & Studio:** Built and tested with capture, review, modeling, and export across 12 formats, backed by over 290 automated tests and verified end-to-end on benchmark examples.
- **Accuracy (2026-09-26):**
  - The multi-view path is measured on 400 reference parts, with the reference-part row in the table above.
  - Turned parts are now built as solids of revolution.
  - The vision model's hole depths and blind flags no longer reach geometry (P0-3).
  - A fillet that does not fit keeps the last good part and suggests the largest size that builds (P0-11).
- **What to do next:** See the checklist and plans in `docs/superpowers/reviews/2026-09-24-project-review.md` and `docs/superpowers/plans/`.
