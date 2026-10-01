# Complex Parts: Notches, Steps, Pockets and Pins From Line Drawings

Date 2026-10-01. Approved in chat: extend the multi-view contract (not the single-view grammar).

## 1. Problem

The multi-view build intersects three extruded outer outlines (a visual hull), then cuts holes and slots. A drawing of a real part carries much more than its outlines:
- a corner notch, a step or a pocket changes no outline; it shows only as inner lines and hidden lines;
- a pin standing on a face shows as a circle in that face's view, and comes out square (the hull of two bumps) or is read as a hole.

Reviewed on a user sheet (block, two front corner notches, two pins, a side hole in a notch): the build was a plain box with square pins, a hole on the wrong wall, and no notch.

## 2. Contract (s2c/multiview/spec.py)

Two new face features, alongside `FaceHole` and `FaceSlot`, same face frames and provenance rules:
- `FacePocket(face, a_mm, b_mm, width_mm, height_mm, depth_mm | None)`: an axis-aligned rectangular cut from the envelope face inward, centred at (a, b); `depth_mm` None is through. A corner notch is a pocket that runs off the outline.
- `FaceBoss(face, a_mm, b_mm, diameter_mm, height_mm)`: a round pin standing on the face: within `height_mm` of the envelope face, the material around the cylinder (its square, slightly enlarged) is removed.

Rule 2 holds: every millimetre is scaled from the typed envelope ("scaled"), as drawn holes already are. The single-view grammar in `s2c/partspec/` is untouched. CLAUDE.md records the decision under the open rule-3 conflict.

## 3. Reading the drawing

- **Visible lines.** The line-art outline stage also returns its long axis-parallel visible line segments (`PixelOutline.lines`, `Observation.lines`), in the same format as hidden lines: ("h" | "v", pos, start, end) in fractions of the bbox, image orientation. Circles and dashed lines are excluded.
- **Relief reconstruction (`s2c/multiview/relief.py`).**
  1. A grid per global axis from every outline vertex and line coordinate of the line-art views, merged within 1.5 % of the axis.
  2. Cells start as the hull: a cell is solid when its centre projects inside each canonical outline of the spec.
  3. Each view predicts its drawing from the cells: a visible line where neighbouring columns show different front depths (or one is empty); a hidden line where a 3D edge exists but is not visible.
  4. Score: per grid edge, a visible line drawn but not predicted (or predicted but not drawn) costs its length; a drawn hidden line that the model has no edge for costs half. Hidden lines the model has but the drawing omits cost nothing (drawings often leave them out).
  5. Search: greedy carving. A move takes one view, one region of that view (cells between drawn lines) and one depth plane, and empties every cell of the region nearer the viewer than that plane. The best improving move is applied until none improves.
  6. The removed cells are merged into boxes; each becomes a `FacePocket` from the nearest face it is open to (nothing solid between it and that face). A box that runs through is through.
- **Pins.** A circle that `classify_drawn_circles` reads as an edge, whose neighbour silhouette shows a bump of its width standing above the rest of the part on that face, becomes a `FaceBoss` with the bump's height.
- **Gates.** Relief runs only when every observation is line art and at least two views are present; a reconstruction that does not improve the score is discarded; a warning lists what was found ("2 notches or pockets read from the inner lines; check them").

## 4. Surfaces

- Builder: pockets are box cuts, bosses are (box − cylinder) cuts, applied with the holes.
- Studio and web Review: the new features list with their fields; the DXF drawing notes them.
- The sheet benchmark is re-run; brackets and hinges are expected to gain.

## 5. Out of scope

Slanted faces, curved pockets, pockets on curved surfaces, internal voids, and filled renders (the rule needs drawn lines; Canny edges of renders are the benchmark's way in).

## 6. Testing

- Exact line drawings from CadQuery parts with OCC hidden-line removal (test helper), not hand-drawn fixtures.
- Named cases: a block with a corner notch, a step block, a top pocket, a through window, a block with two pins, the user's sheet layout reproduced synthetically; each built volume within 3 % of the true part.
- Plain parts (no inner lines) are unchanged: same spec as before.
