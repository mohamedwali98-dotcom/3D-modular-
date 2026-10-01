# Sheet Reading: Faces Labelled, Dimensions Read, Part Built

Date 2026-10-01. Asked by the user: "break down the 2D projection into the faces, label the faces even if the user didn't, read the dimensions if there are, and at the end make the 3D model". Convention stated: the middle view is the front, the view on its right is the left view, the view above it is the bottom view (ISO first-angle). An isometric view is the target part and is ignored for labelling. Standing instruction: pick the recommended option, do not stop.

## 1. What fails today

- **Dimensions break the split.** Dimension lines, extension lines, arrows and numbers touch or nearly touch the views. After dilation they merge into the view, so the view box grows, the scale check fails, views go "auto", and the crop's outline grows with the extension lines.
- **No sizes.** The sheet is read for proportions only; the user types three sizes. Numbers written on the sheet are read per crop, and only when the crop's margin happens to include them.
- **Projection.** Without a symbol or labels the setting decides. A third-angle sheet read as first-angle builds a mirrored or wrong part (the user's own sample sheet is third-angle: its pins show as visible circles in the view above the front).

## 2. Design

### 2.1 Geometry and annotations are separated (sheet.py)
- A view's **body** is its filled closed region: the ink component with its holes filled. Open lines (dimension and extension lines, leaders, centre-line tails) and text enclose no area.
- **Annotations** are the ink outside every body (dilated by a line width). They are kept per sheet as `Sheet.marks`, never as views.
- View boxes are body boxes. Crops are cut from the ink of the view's body and its own inner lines only, so the outline stage never sees a dimension line.
- An isometric view is a body that aligns with no other view, or does not pass the scale check: it stays out of the part (as today), with the note "the 3D view was used only as a picture".

### 2.2 Dimensions read once per sheet (new `s2c/multiview/dimensions.py`)
- **Text.** Annotation components shaped like text (a line of glyphs), grouped into words; vertical words are read rotated. Every word is read by the existing reading service (TrOCR when configured, plus any VLM reader); `ocr.parse_value` keeps numbers, ⌀ and R.
- **Dimension line.** For a linear value: the straight thin annotation line parallel to its reading direction, within 1.5 text heights of it and under its middle. Its span, arrow tip to arrow tip (or extension line to extension line), in sheet pixels.
- **Scale.** Each linear dimension gives mm per sheet pixel = value / span. A ⌀ near a drawn circle gives value / circle diameter. The sheet has one scale: the median of the largest group of dimensions that agree within 3 %. One dimension is enough; disagreeing ones are listed in Review ("Ø12 does not fit the drawing's scale; check it").
- **Result.** `SheetScale(mm_per_px, used, rejected, warnings)`, and the values themselves (for hole diameters near circles).

### 2.3 Into the pipeline
- `ImageInput` gains `mm_per_px` (a known scale in the image's own pixels). `observe` converts it to the resized crop and sets `Observation.mm_per_px`, so the envelope and every drawn size come out "measured" (rule 2: read from the user's own numbers and measured on the drawing; Review shows them as measured, and the user can still edit them).
- A ⌀ read next to a circle is linked to that circle as a written diameter (`Linked` with its hole index) in the crop's frame.
- With a scale the Review screen opens ready to build; without one, nothing changes (the user types the sizes).

### 2.4 Projection chosen by the drawing (`projection="auto"`, the new default)
- The symbol wins, then consistent labels (as today).
- Otherwise both namings are tried on the same crops: each view's outline and lines are extracted once, the faces are assigned per hypothesis, and the relief reader's line mismatch after carving is compared. The clearly better one (by 10 % of the mismatch) is used, with the note "Read as third-angle: the views agree with each other that way". A tie keeps first-angle (ISO), the user's stated convention.
- The web switch gains "Auto (recommended)" as default; First-angle and Third-angle still force it. The Studio radio likewise.

## 3. Rules that stay
- Rule 1: no code from a model; the reader only reads text.
- Rule 2: every millimetre is written by the user on the sheet (read), measured on the drawing with the user's own dimension as the scale, typed or accepted.
- `spec.py` unchanged by this work.

## 4. Testing
- Synthetic sheets from exact HLR line drawings (tests/line_views.py) with drawn dimension lines, arrows, extension lines and numbers (cv2 text); a fake reader that returns the true text for each word box.
- Named cases: dimensions do not change the split or the names; the scale is found from one dimension and from several with one misread; a vertical dimension; a ⌀ on a circle; an isometric view on the sheet; a third-angle sheet without symbol or labels read as third-angle under Auto; a symmetric part stays first-angle; the end-to-end build volume within 3 % with no typed size.
- The web sheet mode test with Auto and a dimensioned sheet builds without typed sizes.
