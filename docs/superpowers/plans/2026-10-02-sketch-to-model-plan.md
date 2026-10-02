# Sketch to Model Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** One uploaded image, a hand-sketch photo or a clean drawing, goes through five steps (faces, labels, generate the rest, OCR, model) to a draft 3D model the user checks.

**Architecture:** A new entry `sheet_read.read_drawing` adds a page step (the team's `capture` for photos) in front of the existing sheet reader, splits views by closed outlines for sketches, widens the naming tolerances for sketches, straightens wobbly view crops from the team's `vectorize` primitives, and reads a sketch's written overall sizes per axis instead of trusting one scale. Web one-sheet mode and the Studio call it.

**Tech Stack:** Python 3.11, OpenCV, NumPy, SciPy, CadQuery, pytest, uv; React/TypeScript for stage labels.

**Spec:** `docs/superpowers/specs/2026-10-02-sketch-to-model-design.md`

## Global Constraints

- Rule 1: no code from any model; readers only read text.
- Rule 2: every millimetre is written by the user (read), measured on a drawing with the user's own dimensions as the scale, typed, or accepted from a flagged suggestion. A sketch's proportions never become trusted sizes.
- Rule 4: each step abstains with a reason and a remedy.
- No new dependencies; `s2c/multiview/spec.py` unchanged.
- Clean sheets unchanged: the 44-part sheet benchmark stays 44/44 built and named, median IoU ≥ 0.89; every existing test passes.
- Test fixtures are synthetic (or the repo's own `tests/golden_sketch/real_bracket_1`); nothing from `Reverce engineering`; no AI attribution in commits.

## Review Focus

1. A clean CAD drawing must never be sent through `capture` (photo path): it would be re-binarised and lose thin lines. Test in Task 1: a clean drawing sheet keeps kind "drawing" and its exact pixels.
2. A hand sketch whose views touch through a long dimension line or a miter line must still give separate faces. Test in Task 2: hand sheet with the miter line and dimension lines between views → 3 faces.
3. A sketch's numbers must never become a trusted scale: proportions only suggest. Test in Task 5: a hand sheet with one overall dimension per axis gives `user_written` envelope values; with none, the fuse abstains with suggested sizes, never "measured".
4. A value that spans only part of a view (a sub-feature) must not be taken as the envelope. Test in Task 5: a "3" chamfer dimension is listed as "read, not used", not as a size.
5. Straightening must not erase circles (holes, pins) or curves. Test in Task 4: a wobbly view with a hole keeps its circle after straightening.

---

### Task 1: Page step and hand-sketch fixtures

**Files:**
- Create: `tests/hand_views.py` (photo-like hand sketches from exact line drawings)
- Modify: `s2c/multiview/sheet_read.py` (add `Page`, `page_of`)
- Test: `tests/test_mv_sketch_read.py` (new)

**Interfaces:**
- Produces: `@dataclass Page: image: np.ndarray (BGR, black on white); kind: str ("sketch" | "drawing"); to_photo: np.ndarray (3x3 homography page -> photo); stroke_px: float`; `page_of(image_bgr) -> Page | S.MvAbstain`.
- Produces (tests): `hand_photo(part, faces, layout, dims=True, seed=0, miter=False) -> (photo_bgr, words_on_photo)`; words are `(box_on_photo, text)`.

- [ ] **Step 1: fixtures.** `tests/hand_views.py`: start from `line_views.drawing_sheet` (add a `font` parameter there, default `cv2.FONT_HERSHEY_SIMPLEX`; hand sheets use `cv2.FONT_HERSHEY_SCRIPT_SIMPLEX`), optionally draw a dashed 45° miter line between the plan view and the side view, then make it hand-like: a smooth random displacement field (amplitude 2.5 px, Gaussian-blurred noise, seeded), a 1.5° rotation, ink colour dark blue (60, 30, 20) on paper (235, 235, 230); then paste the page onto a 1.3× larger dark desk (90, 70, 50) with a perspective warp (corners moved up to 4 % of the size) and a left-to-right shadow gradient (paper down to 60 % brightness). Word boxes are carried through every transform (their four corners mapped, then the bounding box).
- [ ] **Step 2: failing tests** in `tests/test_mv_sketch_read.py`:

```python
def test_a_clean_drawing_is_used_as_drawn():
    img, _, _ = drawing_sheet(_part(), faces=THIRD, layout="third")
    page = page_of(img)
    assert page.kind == "drawing" and page.image.shape == img.shape and np.array_equal(page.image, img)

def test_a_hand_photo_becomes_a_clean_page():
    photo, _ = hand_photo(_part(), faces=THIRD, layout="third")
    page = page_of(photo)
    assert page.kind == "sketch" and max(page.image.shape[:2]) == 1600
    gray = cv2.cvtColor(page.image, cv2.COLOR_BGR2GRAY)
    assert set(np.unique(gray)) <= {0, 255}          # black ink on white
    corners = cv2.perspectiveTransform(np.float32([[[0, 0]]]), page.to_photo)
    assert 0 <= corners[0, 0, 0] < photo.shape[1]

def test_a_dark_photo_abstains_with_a_remedy():
    page = page_of(np.full((900, 1200, 3), 20, np.uint8))
    assert isinstance(page, S.MvAbstain) and page.remedy
```

- [ ] **Step 3: run** `uv run pytest tests/test_mv_sketch_read.py -q` — Expected: FAIL (ImportError `page_of`).
- [ ] **Step 4: implement** `page_of`: clean when ≥ 97 % of pixels are within 40 grey levels of the median border grey or of the 1st-percentile ink grey and the mean HSV saturation of paper pixels is under 25; clean → `Page(img, "drawing", np.eye(3), stroke)`; else `s2c.sketch.capture.capture(img)`: `SketchAbstain` → `MvAbstain(stage="outline", reason=a.reason, remedy=a.remedy)`; `Captured` → `Page(255 - ink as BGR, "sketch", captured.to_original, captured.stroke_px)`.
- [ ] **Step 5: run** the test file — Expected: PASS. Commit: `git commit -m "Page step: a photo of a sketch becomes a clean page; a clean drawing is used as drawn"`.

### Task 2: Faces from closed outlines

**Files:**
- Modify: `s2c/multiview/sheet.py` (add `split_by_outlines`)
- Test: `tests/test_mv_sketch_read.py`

**Interfaces:**
- Consumes: `Page` (Task 1).
- Produces: `split_by_outlines(image_bgr, stroke_px: float) -> Sheet` with `View(box=body box, label_box, line_art=True)`; grouped by `_drawings`.

- [ ] **Step 1: failing tests:**

```python
def test_views_joined_by_dimension_and_miter_lines_are_three_faces():
    photo, _ = hand_photo(_part(), faces=THIRD, layout="third", miter=True)
    page = page_of(photo)
    sheet = split_by_outlines(page.image, page.stroke_px)
    big = [v for d in sheet.drawings for v in d.views if v.box[2] * v.box[3] > 0.02 * page.image.size / 3]
    assert len(big) == 3

def test_the_real_sketch_has_three_faces():
    page = page_of(cv2.imread(str(GOLDEN / "real_bracket_1" / "image.jpg")))
    sheet = split_by_outlines(page.image, page.stroke_px)
    big = [v for d in sheet.drawings for v in d.views if v.box[2] * v.box[3] > 0.01 * page.image.shape[0] * page.image.shape[1]]
    assert len(big) == 3
```

- [ ] **Step 2: run** — Expected: FAIL (ImportError).
- [ ] **Step 3: implement:** ink = `ink_mask(image)`; `_remove_border`; mend gaps with a closing of `k = 2 * stroke + 1` px; `ndimage.binary_fill_holes`; open with `k` to drop open lines; components of area ≥ 1 % of the page are bodies; a body's box is its bbox; label boxes: the existing label rule (text component within 1.5 heights above/below, overlapping) on the ink outside bodies; `Sheet(_drawings(views, []), shape, [])`.
- [ ] **Step 4: run** — Expected: PASS. Commit: `git commit -m "Faces from closed outlines: lines and text between views never join them"`.

### Task 3: `read_drawing` and sketch-tolerant labels

**Files:**
- Modify: `s2c/multiview/sheet.py` (`name_views(..., scale_tol=SCALE)` threaded to `_name`/`_fits`/`_close`; `is_sheet(sheet, align=ALIGN, extent=EXTENT)`)
- Modify: `s2c/multiview/sheet_read.py` (`read_drawing`; `SheetRead.kind`, `SheetRead.page`)
- Test: `tests/test_mv_sketch_read.py`

**Interfaces:**
- Produces: `read_drawing(image_bgr, projection="auto", reader=None, service=None, keep_unnamed=False) -> SheetRead | S.MvAbstain | None`. Sketch: `split_by_outlines` first (dilation split when it finds under 2 bodies), `scale_tol=0.2`, `is_sheet(align=0.2, extent=0.4)`. Drawing: exactly `read_sheet`.

- [ ] **Step 1: failing tests:**

```python
def test_a_hand_sketch_is_read_into_named_faces():
    photo, _ = hand_photo(_part(), faces=THIRD, layout="third", miter=True)
    read = read_drawing(photo, "third")
    assert read.kind == "sketch" and sorted(c.face for c in read.crops) == sorted(THIRD)

def test_a_clean_drawing_reads_exactly_as_before():
    img, _, words = drawing_sheet(_part(), faces=THIRD, layout="third", iso=True)
    a = read_sheet(img, "auto", service=_service(words))
    b = read_drawing(img, "auto", service=_service(words))
    assert b.kind == "drawing" and [c.face for c in a.crops] == [c.face for c in b.crops]
    assert a.scale.mm_per_px == b.scale.mm_per_px
```

- [ ] **Step 2: run** — Expected: FAIL. **Step 3: implement.** **Step 4: run** — PASS, plus `tests/test_mv_sheet_*` unchanged. Commit: `git commit -m "read_drawing: hand sketches get a page, outline faces and wider naming tolerances"`.

### Task 4: Straightened sketch views

**Files:**
- Modify: `s2c/multiview/sheet_read.py` (`_straighten(crop_bgr, stroke) -> np.ndarray`, used for sketch crops)
- Test: `tests/test_mv_sketch_read.py`

**Interfaces:**
- Consumes: `s2c.sketch.vectorize.vectorize(ink, stroke_px, view) -> list[Prim]` (kinds line, circle, arc, curve).
- Produces: sketch `SheetCrop.png` redrawn from primitives (lines straight, circles round, arcs and curves as drawn) at the crop's own scale and line width; kept as drawn when primitives cover under 80 % of the ink.

- [ ] **Step 1: failing tests:**

```python
def test_a_wobbly_notched_block_reads_its_notch():
    part = _cut(_block(), 0, 40, 50, 15, 60, 70)
    photo, words = hand_photo(part, faces=THIRD, layout="third")
    read = read_drawing(photo, "third")
    pipe = MvPipeline()
    spec = pipe.fuse(pipe.observe(read.inputs()), {"envelope.x_mm": 80, "envelope.y_mm": 60, "envelope.z_mm": 70})
    assert [f for f in spec.features if f.type == "pocket"]

def test_straightening_keeps_a_hole_round():
    part = _block().faces(">Z").workplane(centerOption="CenterOfBoundBox").hole(12)
    photo, _ = hand_photo(part, faces=THIRD, layout="third")
    read = read_drawing(photo, "third")
    obs = MvPipeline().observe(read.inputs())
    front = next(o for o in obs.observations if o.face == "front")
    assert len(front.outline.circles) == 1
```

- [ ] **Step 2–5:** run (FAIL), implement, run (PASS), commit `"Sketch views are redrawn from fitted lines and circles before they are read"`.

### Task 5: A sketch's numbers, per axis

**Files:**
- Modify: `s2c/multiview/dimensions.py` (`read_dimensions(..., kind="drawing")`: on sketches, text boxes from `s2c.sketch.text.find_text_boxes(ink, stroke)`, dimension lines as straight segments within 10° of horizontal or vertical (`cv2.HoughLinesP` on the annotation ink), no scale confirmation)
- Modify: `s2c/multiview/sheet_read.py` (`link_sizes(read, observed)`: overall values -> `Linked` written values on the view's axis; others -> notes "Read 3 near the front view; not used")
- Test: `tests/test_mv_sketch_read.py`

**Interfaces:**
- Produces: `Dimension.view: int | None` and `Dimension.axis: str | None` ("a" | "b" of that view's face frame) for values whose line spans the view's full extent (15 % on sketches, 8 % on drawings); `link_sizes(read, observed) -> None`.

- [ ] **Step 1: failing tests:**

```python
def test_a_sketch_builds_from_its_written_sizes():
    part = _block()
    photo, words = hand_photo(part, faces=THIRD, layout="third")
    read = read_drawing(photo, "third", service=_service(words))
    pipe = MvPipeline()
    observed = pipe.observe(read.inputs())
    link_sizes(read, observed)
    spec = pipe.fuse(observed)
    assert (spec.envelope.x_mm, spec.envelope.y_mm, spec.envelope.z_mm) == (80, 60, 70)
    assert spec.provenance["envelope.x_mm"] == "user_written"

def test_a_sketch_without_numbers_suggests_and_never_measures():
    photo, _ = hand_photo(_block(), faces=THIRD, layout="third", dims=False)
    read = read_drawing(photo, "third")
    result = MvPipeline().fuse(MvPipeline().observe(read.inputs()))
    assert isinstance(result, S.MvAbstain) and result.stage == "dimensions"

def test_a_part_dimension_is_read_but_not_used_as_a_size():
    part = _cut(_block(), 0, 40, 50, 15, 60, 70)
    # one more dimension, over the notch's 15 mm width only (front view, along a, 40 px above the view)
    photo, words = hand_photo(part, faces=THIRD, layout="third", extra=[("front", 0, 15, "a", "15")])
    read = read_drawing(photo, "third", service=_service(words))
    pipe = MvPipeline()
    observed = pipe.observe(read.inputs())
    link_sizes(read, observed)
    spec = pipe.fuse(observed)
    assert spec.envelope.x_mm == 80
    assert any("Read 15" in w and "not used" in w for w in spec.warnings)
```

  (`hand_photo(..., extra=[(face, start_mm, end_mm, axis, text)])` draws extra dimensions between two points of a view edge, given in that face's frame.)

- [ ] **Step 2–5:** run (FAIL), implement, run (PASS), commit `"A sketch's written sizes set the envelope per axis; part dimensions are listed, never a scale"`.

### Task 6: Wiring into the web app and the Studio

**Files:**
- Modify: `s2c/web/jobs.py` (`_drawn_sheet` -> `read_drawing`; `link_sizes` after `observe`; an `MvAbstain` from the page step fails the job with its remedy), `web/src/screens/Analyzing.tsx` (stage names: views "Finding the faces", lines "Labelling and completing the faces", values "Reading your numbers")
- Modify: `s2c/studio/handlers.py` (`_split` -> `read_drawing(..., keep_unnamed=True)`)
- Test: `tests/test_web_api.py`

- [ ] **Step 1: failing test:** a hand photo with dims posted in sheet mode with a `WordReader` pipeline → job done, faces front/top/right, envelope `user_written`.
- [ ] **Step 2–5:** run (FAIL), implement, run (PASS), commit `"Web app and Studio read hand sketches through the five steps"`.

### Task 7: The real sketch, the benchmark, docs

**Files:**
- Test: `tests/test_mv_sketch_read.py` (`test_the_real_sketch_end_to_end`, skipped without the `trocr` extra)
- Modify: `README.md`, `CLAUDE.md`

- [ ] **Step 1:** the real sheet with `default_pipeline()` readers: 3 faces; names top/front/right, or bottom/front/left with projection source "setting" (ISO when the drawing does not decide); at least 8 of the 10 values of `expected.json` read; the part builds.
- [ ] **Step 2:** run the 44-part sheet benchmark; 44/44 built and named, median IoU ≥ 0.89.
- [ ] **Step 3:** docs, full suite, final review (opus), fix pass, push to `modular main`.
