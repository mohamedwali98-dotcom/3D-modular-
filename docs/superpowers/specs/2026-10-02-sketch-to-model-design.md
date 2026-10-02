# One Sketch to a 3D Model in Five Steps

Date 2026-10-02. Asked by the user: "the model has to break down this 2D sketch to create the 3D model: 1. find the faces, 2. label them, 3. auto detect and generate the rest, 4. read with the OCR, 5. make the 3D model". Choices made in chat: both hand-drawn photos and clean drawings; the result is a draft the user checks (every label and number shown, unsure ones flagged); approach A (one pipeline, two front ends); standing instruction: take the recommended option.

## 1. Where it stands

- **Clean drawings** (printed, CAD, screenshots): `sheet_read.read_sheet` already runs the five steps.
- **Hand sketches** (a phone photo of a pen sketch): nothing works end to end. On the real test sheet `tests/golden_sketch/real_bracket_1` (top, front and right views, ten written values, a 45° miter line, hidden lines, a page shadow and a notebook spiral):
  - `read_sheet` finds no sheet: dimension lines in the gap join the front and right views into one blob after dilation;
  - the team's `read_sketch` finds 2 of 3 views and reads almost no number right ("10", "il", "l" for 60, 25, Ø6), and guesses an envelope of 122 × 68 × 74 mm.
- The team's `s2c/sketch/capture.capture` turns the photo into a clean ink page (sheet found, rectified, shadows flattened, spiral and desk removed). It works well on the real sheet.

## 2. Success

- On `real_bracket_1`: 3 views found and named top, front, right; at least 8 of the 10 written values read correctly, the others flagged, never silently wrong; the envelope from the written sizes; the part builds.
- Synthetic hand sketches (wobbly, slightly slanted strokes, hand-like digits, a shadow): the same, measured in tests.
- Clean sheets unchanged: the 44-part sheet benchmark stays 44/44 built and named, median IoU ≥ 0.89; every existing test passes.

## 3. Design

One entry point, `sheet_read.read_drawing(image, projection="auto", reader=None, service=None) -> SheetRead | None`, used by the web one-sheet mode and the Studio. Its steps:

### 3.0 Page
- **Photo or drawing.** An image is a clean drawing when nearly all of it is paper or ink (≥ 97 % of pixels within 40 grey levels of the paper or of the darkest ink) and it has no colour cast; otherwise it is a photo.
- A photo goes through `capture`: the result is the ink page (black on white) plus the homography back to the photo, kept on `SheetRead.to_photo` for overlays. `capture` abstaining (no sheet, too dark, blurred) ends the reading with its remedy (rule 4).
- A drawing is used as it is (identity mapping).
- `SheetRead.kind` is "sketch" or "drawing"; later steps adapt their tolerances to it.

### 3.1 Faces: views are closed outlines
- The ink, with small gaps mended (a closing of about two line widths, so a hand-drawn corner that does not quite meet still closes), is filled; lines up to about two line widths thick are opened away; every remaining blob of at least 1 % of the page is a view body. A view body's box is the view.
- Dimension, extension, centre and miter lines, leaders, text, and the picture's thin lines cannot join two views or enlarge one, because they enclose no area.
- Everything outside the bodies is annotation (text and dimension lines for step 4).
- When the outline split finds fewer than two bodies, the existing dilation split is used (clean sheets keep today's behaviour exactly when the outline split is not needed).

### 3.2 Labels
Unchanged: the projection symbol, then consistent labels (read in step 4's pass), then how the views agree (`relief.mismatch`), ISO first-angle when unsure; a view that cannot be named (the 3D picture) is left out with a note. For sketches the scale check tolerance is widened from 8 % to 20 % (hand sketches are not drawn to scale).

### 3.3 Generate the rest
Unchanged machinery, run on the named views: mirrored faces, the second side view of a turned part, notches, pockets and windows from inner lines, round pins. For sketches, relief carving needs the lines straightened first: each view crop is drawn again, at its own scale and line width, from the primitives `s2c/sketch/vectorize.vectorize` fits to its strokes (lines, circles, arcs; polylines kept as drawn), so wobbly strokes read as the straight lines they mean. A view whose primitives cover under 80 % of its ink is kept as drawn.

### 3.4 Numbers
- **Text** is found with the team's handwriting-tuned finder (`s2c/sketch/text.find_text_boxes`) on sketches and with `dimensions._words` on drawings; each box is read by the reading service (TrOCR, plus the vision model when configured), upright, and also turned both ways when it sits along a vertical line; `ocr.parse_value` keeps numbers, Ø (also a handwritten "o" or "ϕ" before digits) and R.
- **Dimension line**: the nearest straight thin annotation segment within 1.6 text heights, parallel to the text within 10° (hand lines slant), running under the text's middle; its span is tip to tip.
- **Use of each value**:
  - a linear value whose span covers a view's full width or height (within 8 % on drawings, 15 % on sketches) is that envelope size, written by the user: it is attached to that view as a written value on that axis (`Linked`, axis a or b), so `fuse_envelope` takes it as `user_written` and reports conflicts;
  - a Ø next to a circle is that hole's written diameter (existing `link_diameters` rule, with the 10 % check only on drawings);
  - every other value is listed in Review as "read, not used" with its text, so the user sees it was read.
- **Scale**: on drawings, as today (two agreeing dimensions confirm it). On sketches, a scale is never trusted: proportions only suggest the sizes no value gives, as checks.

### 3.5 Model and Review
The existing build. Review shows the five steps' results: faces with their labels, the numbers read (used, flagged, or not used), and the warnings. The Analyzing screen names the stages Page, Faces, Labels, Generate, Numbers, Model.

## 4. Rules
- Rule 1: no code from any model; the readers only read text.
- Rule 2: every millimetre is written by the user (read), measured on a drawing with the user's own dimensions as the scale, typed, or accepted from a flagged suggestion. A sketch's proportions never become trusted sizes.
- Rule 4: each step abstains with a reason and a remedy (no page, fewer than two faces, no readable number: "type the sizes").
- No new dependencies; `spec.py` unchanged.

## 5. Out of scope
- Solving sub-feature sizes from every written value (the team's solve stage): a "read, not used" value does not move geometry yet.
- Section views, detail views, title blocks, perspective sketches without orthographic views.

## 6. Testing
- Synthetic hand sketches from exact HLR line drawings, made hand-like: strokes jittered and slanted by up to 3°, line ends that overshoot or stop short, hand-like digits (cv2 Hershey script), a shadow gradient and a perspective warp; a stand-in reader for unit tests.
- The real `real_bracket_1` sheet end to end with the real TrOCR when the `trocr` extra is installed (skipped otherwise): views and names, values read, envelope, build.
- Regression: all existing tests; the 44-part sheet benchmark.
