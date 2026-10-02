"""Dimensions written on a drawing sheet -> the sheet's one scale (sheet-reading spec 2026-10-01, section 2.2).

The numbers are the user's own (rule 2): the reader only reads them. Each linear value sits over its dimension
line; value / line length is the sheet's millimetres per pixel. All views share that scale, so one dimension sizes
every view, and several that agree confirm it."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import cv2
import numpy as np

from s2c.multiview.ocr import decide
from s2c.reading import Crop, ReadingService

log = logging.getLogger(__name__)

Box = tuple[int, int, int, int]
LINE_MIN = 0.025     # a dimension line is at least this share of the sheet's long side (30 px at least),
                     # longer than a digit's strokes, which must stay text
WORD_GROW = 0.004    # glyphs closer than this share of the long side are one word
REACH = 1.6          # a dimension line lies within this many text heights of its value
AGREE = 0.03         # dimensions agreeing within this share give one scale
MAX_WORDS = 24


@dataclass
class Dimension:
    value_mm: float
    kind: str                      # linear | diameter | radius
    text: str
    box: Box                       # the text, sheet pixels
    span_px: float | None = None   # linear: its dimension line's length
    confirmed: bool = True
    line: tuple[str, float, int, int] | None = None  # the dimension line: axis "h"|"v", position across, start, end
    view: int | None = None        # the view whose whole width or height the line spans (an overall size)...
    axis: str | None = None        # ...and which: "a" (its width) or "b" (its height)


@dataclass
class SheetScale:
    mm_per_px: float | None
    used: list[Dimension] = field(default_factory=list)
    rejected: list[Dimension] = field(default_factory=list)
    dimensions: list[Dimension] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    confirmed: bool = False  # two or more dimensions agree and none disagrees: the sizes are measured, not checks


def _segments(lines: np.ndarray, axis: str) -> list[tuple[float, int, int]]:
    """(position across, start, end) of each straight run of a line mask: rows for "h", columns for "v"."""
    _, _, stats, _ = cv2.connectedComponentsWithStats(lines, connectivity=8)
    out = []
    for x, y, w, h, _ in stats[1:]:
        out.append((y + h / 2, int(x), int(x + w - 1)) if axis == "h" else (x + w / 2, int(y), int(y + h - 1)))
    return out


def _words(text_ink: np.ndarray, long: int) -> list[Box]:
    g = max(3, round(WORD_GROW * long))
    grown = cv2.dilate(text_ink, np.ones((g, g), np.uint8))
    _, _, stats, _ = cv2.connectedComponentsWithStats(grown, connectivity=8)
    boxes = []
    for x, y, w, h, area in stats[1:]:
        short, tall = min(w, h), max(w, h)
        if 0.006 * long <= short <= 0.06 * long and tall <= 0.25 * long:
            boxes.append((int(x + g // 2), int(y + g // 2), int(w - 2 * (g // 2)), int(h - 2 * (g // 2))))
    return boxes


def annotation_ink(ink: np.ndarray, bodies: list[tuple[Box, np.ndarray]], grow: int = 5) -> np.ndarray:
    """The sheet's ink outside every view body: dimension lines, extension lines, arrows, numbers, notes."""
    inside = np.zeros_like(ink)
    for (x, y, w, h), mask in bodies:
        inside[y: y + h, x: x + w] |= mask.astype(np.uint8) * 255
    inside = cv2.dilate(inside, np.ones((grow, grow), np.uint8))
    return cv2.bitwise_and(ink, cv2.bitwise_not(inside))


def read_dimensions(image_bgr: np.ndarray, ink: np.ndarray, bodies: list[tuple[Box, np.ndarray]],
                    service: ReadingService | None, kind: str = "drawing", stroke: float | None = None,
                    to_source: np.ndarray | None = None) -> SheetScale:
    """The dimensions written around the views and the scale they agree on. No reader, or no dimension whose
    line is found: no scale (the user types the sizes, as before).

    A hand sketch (`kind` "sketch", with its line width) is read with the handwriting text finder and dimension
    lines that may slant; it is not drawn to scale, so its dimensions never give one (sketch-to-model spec 3.4).
    `to_source` maps the image's pixels to the upload's, for the readers' crop boxes."""
    if service is None:
        return SheetScale(None)
    long = max(ink.shape)
    notes = annotation_ink(ink, bodies)
    length = max(30, round(LINE_MIN * long))
    if kind == "sketch":
        from s2c.sketch.text import find_text_boxes
        segs, lines = _slanted(notes, length, stroke or 2.0)
        text_ink = cv2.subtract(notes, cv2.dilate(lines, np.ones((3, 3), np.uint8)))
        boxes = find_text_boxes(notes, stroke or 2.0)
    else:
        h_lines = cv2.morphologyEx(notes, cv2.MORPH_OPEN, np.ones((1, length), np.uint8))
        v_lines = cv2.morphologyEx(notes, cv2.MORPH_OPEN, np.ones((length, 1), np.uint8))
        segs = {"h": _segments(h_lines, "h"), "v": _segments(v_lines, "v")}
        text_ink = cv2.subtract(notes, cv2.dilate(cv2.bitwise_or(h_lines, v_lines), np.ones((3, 3), np.uint8)))
        boxes = _words(text_ink, long)
    words = []  # (box, the dimension line (axis, position, start, end) it is written on, or None)
    for box in boxes:
        if _solid(text_ink, box):
            continue  # an arrowhead or a filled mark: never a number
        line = _line_for(box, segs)
        if line is not None or _near_view(box, bodies, long):  # a value on its line, or a Ø / R callout
            words.append((box, line))
    words = sorted(words, key=lambda w: (w[1] is None, -w[0][2] * w[0][3]))[:MAX_WORDS]
    if not words:
        return SheetScale(None)
    image = image_bgr if image_bgr.ndim == 3 else cv2.cvtColor(image_bgr, cv2.COLOR_GRAY2BGR)
    crops, owner = [], []
    for k, ((x, y, w, h), line) in enumerate(words):
        m = max(3, round(0.15 * min(w, h)))
        tile = image[max(0, y - m): y + h + m, max(0, x - m): x + w + m]
        upright = line is None or line[0] == "h"
        turned = not upright or (line is None and h > 1.3 * w)
        turns = ([None] if upright else []) + ([cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE]
                                                if turned else [])
        source = _mapped((x, y, w, h), to_source)
        for turn in turns:  # along a vertical line the text is turned: read it both ways round
            crops.append(Crop(tile if turn is None else cv2.rotate(tile, turn), source))
            owner.append(k)
    runs = service.read(crops)
    found: dict[int, Dimension] = {}
    for j, k in enumerate(owner):
        decided = decide(runs, j)
        if decided is None or k in found:
            continue
        text, (value, what), _, confirmed = decided
        box, line = words[k]
        linear = line is not None and what == "linear"
        found[k] = Dimension(value, what, text, box, float(line[3] - line[2] + 1) if linear else None, confirmed,
                             line if linear else None)
    if kind == "sketch":  # a sketch is not drawn to scale: its values are sizes, never a scale (rule 2)
        return SheetScale(None, dimensions=list(found.values()))
    return _agree(list(found.values()))


def _slanted(notes: np.ndarray, length: int, stroke: float):
    """Straight annotation lines that may slant and wobble a little (a hand-drawn dimension line), as
    {"h"|"v": [(position across, start, end)]}, and their ink: the ink thickened across by a line width either side
    keeps a run of `length` px along the line where a thin hand line drifts by a pixel or two."""
    t = 2 * max(1, round(stroke)) + 1
    segs, lines = {}, np.zeros_like(notes)
    for axis, grow, run in (("h", (t, 1), (1, length)), ("v", (1, t), (length, 1))):
        band = cv2.morphologyEx(cv2.dilate(notes, np.ones(grow, np.uint8)), cv2.MORPH_OPEN, np.ones(run, np.uint8))
        found = cv2.bitwise_and(notes, band)
        segs[axis] = _segments(found, axis)
        lines |= found
    return segs, lines


def _mapped(box: Box, to_source: np.ndarray | None) -> Box:
    """The box on the upload (through the homography `to_source`), or as it is."""
    if to_source is None:
        return box
    x, y, w, h = box
    pts = cv2.perspectiveTransform(np.float32([[[x, y], [x + w, y], [x + w, y + h], [x, y + h]]]), to_source)[0]
    x0, y0 = pts.min(0)
    x1, y1 = pts.max(0)
    return int(x0), int(y0), int(x1 - x0), int(y1 - y0)


def _solid(text_ink: np.ndarray, box: Box) -> bool:
    """A filled mark (an arrowhead, a dot): its ink is thick for its size, where a written glyph is a thin stroke."""
    x, y, w, h = box
    sub = (text_ink[y: y + h, x: x + w] > 0).astype(np.uint8)
    if not sub.any():
        return True
    dist = cv2.distanceTransform(sub, cv2.DIST_L2, 3)
    return float(dist.max()) > 0.3 * min(w, h)  # a triangle is a third of its size thick; bold text a quarter


def _near_view(box: Box, bodies, long: int) -> bool:
    x, y, w, h = box
    reach = 0.08 * long
    for (bx, by, bw, bh), _ in bodies:
        if bx - reach <= x + w / 2 <= bx + bw + reach and by - reach <= y + h / 2 <= by + bh + reach:
            return True
    return False


def _line_for(box: Box, segs) -> tuple[str, float, int, int] | None:
    """The dimension line a value is written on, as (axis, position across, start, end): parallel to the text,
    within REACH text heights of it, running past the text's middle. Both orientations are tried; the nearer, in
    text heights, wins, so a narrow single digit is read the right way round."""
    x, y, w, h = box
    best = None
    for axis in ("h", "v"):
        size = h if axis == "h" else w           # the text's height across its line
        mid = x + w / 2 if axis == "h" else y + h / 2
        near, far = (y, y + h) if axis == "h" else (x, x + w)
        for pos, s, e in segs[axis]:
            if not (s - size <= mid <= e + size) or near < pos < far:
                continue
            gap = min(abs(pos - near), abs(pos - far))
            if gap <= REACH * size and (best is None or gap / size < best[0]):
                best = (gap / size, (axis, float(pos), int(s), int(e)))
    return best[1] if best else None


def _agree(dims: list[Dimension]) -> SheetScale:
    """The scale the dimensions agree on. Trusted (confirmed) only when two or more agree and none disagrees; one
    lone dimension, or a group with a dissenter, gives sizes for the user to confirm; dimensions that do not agree
    at all give no scale: a misread number never scales the part on its own (rule 2)."""
    linear = [d for d in dims if d.kind == "linear" and d.span_px]
    if not linear:
        return SheetScale(None, dimensions=dims)
    scales = np.array([d.value_mm / d.span_px for d in linear])
    support = [int(np.sum(np.abs(scales / s - 1) <= AGREE)) for s in scales]
    best = int(np.argmax(support))
    if support[best] < 2 and len(linear) > 1:
        texts = ", ".join(d.text for d in linear)
        return SheetScale(None, dimensions=dims, warnings=[
            f"The dimensions read ({texts}) do not agree with each other; type the sizes."])
    group = np.abs(scales / scales[best] - 1) <= AGREE
    used = [d for d, g in zip(linear, group, strict=True) if g]
    rejected = [d for d, g in zip(linear, group, strict=True) if not g]
    scale = float(np.median(scales[group]))
    warnings = [f"The dimension {d.text} does not fit the drawing's scale; check it and the sizes."
                for d in rejected]
    if len(used) == 1:
        warnings.append(f"Sizes are scaled from one dimension ({used[0].text}); confirm them.")
    return SheetScale(scale, used, rejected, dims, warnings, confirmed=len(used) >= 2 and not rejected)
