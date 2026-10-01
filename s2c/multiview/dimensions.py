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
LINE_MIN = 0.012     # a dimension line is at least this share of the sheet's long side
WORD_GROW = 0.004    # glyphs closer than this share of the long side are one word
REACH = 1.6          # a dimension line lies within this many text heights of its value
AGREE = 0.03         # dimensions agreeing within this share give one scale
MAX_WORDS = 40


@dataclass
class Dimension:
    value_mm: float
    kind: str                      # linear | diameter | radius
    text: str
    box: Box                       # the text, sheet pixels
    span_px: float | None = None   # linear: its dimension line's length
    confirmed: bool = True


@dataclass
class SheetScale:
    mm_per_px: float | None
    used: list[Dimension] = field(default_factory=list)
    rejected: list[Dimension] = field(default_factory=list)
    dimensions: list[Dimension] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


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
                    service: ReadingService | None) -> SheetScale:
    """The dimensions written around the views and the scale they agree on. No reader, or no dimension whose
    line is found: no scale (the user types the sizes, as before)."""
    if service is None:
        return SheetScale(None)
    long = max(ink.shape)
    notes = annotation_ink(ink, bodies)
    length = max(15, round(LINE_MIN * long))
    h_lines = cv2.morphologyEx(notes, cv2.MORPH_OPEN, np.ones((1, length), np.uint8))
    v_lines = cv2.morphologyEx(notes, cv2.MORPH_OPEN, np.ones((length, 1), np.uint8))
    segs = {"h": _segments(h_lines, "h"), "v": _segments(v_lines, "v")}
    text_ink = cv2.subtract(notes, cv2.dilate(cv2.bitwise_or(h_lines, v_lines), np.ones((3, 3), np.uint8)))
    words = sorted(_words(text_ink, long), key=lambda b: -b[2] * b[3])[:MAX_WORDS]
    if not words:
        return SheetScale(None)
    image = image_bgr if image_bgr.ndim == 3 else cv2.cvtColor(image_bgr, cv2.COLOR_GRAY2BGR)
    crops, owner = [], []
    for k, (x, y, w, h) in enumerate(words):
        m = max(3, round(0.15 * min(w, h)))
        tile = image[max(0, y - m): y + h + m, max(0, x - m): x + w + m]
        if h > 1.3 * w:  # written along a vertical dimension line: read it both ways round
            for turn in (cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE):
                crops.append(Crop(cv2.rotate(tile, turn), (x, y, w, h)))
                owner.append(k)
        else:
            crops.append(Crop(tile, (x, y, w, h)))
            owner.append(k)
    runs = service.read(crops)
    found: dict[int, Dimension] = {}
    for j, k in enumerate(owner):
        decided = decide(runs, j)
        if decided is None or k in found:
            continue
        text, (value, kind), _, confirmed = decided
        x, y, w, h = words[k]
        span = _span(words[k], segs) if kind == "linear" else None
        found[k] = Dimension(value, kind, text, (x, y, w, h), span, confirmed)
    dims = list(found.values())
    return _agree(dims)


def _span(box: Box, segs) -> float | None:
    """The length of the dimension line this value is written on: parallel to the text, within REACH text heights
    of it, running under the text's middle."""
    x, y, w, h = box
    vertical = h > 1.3 * w
    axis = "v" if vertical else "h"
    size = w if vertical else h               # the text's height across its line
    mid = y + h / 2 if vertical else x + w / 2
    near, far = (x, x + w) if vertical else (y, y + h)
    best = None
    for pos, s, e in segs[axis]:
        if not (s - size <= mid <= e + size):
            continue
        gap = min(abs(pos - near), abs(pos - far))
        if gap > REACH * size or (near < pos < far):
            continue
        if best is None or gap < best[0]:
            best = (gap, e - s + 1)
    return float(best[1]) if best else None


def _agree(dims: list[Dimension]) -> SheetScale:
    linear = [d for d in dims if d.kind == "linear" and d.span_px]
    if not linear:
        return SheetScale(None, dimensions=dims)
    scales = np.array([d.value_mm / d.span_px for d in linear])
    support = [int(np.sum(np.abs(scales / s - 1) <= AGREE)) for s in scales]
    best = int(np.argmax(support))
    group = np.abs(scales / scales[best] - 1) <= AGREE
    used = [d for d, g in zip(linear, group, strict=True) if g]
    rejected = [d for d, g in zip(linear, group, strict=True) if not g]
    scale = float(np.median(scales[group]))
    warnings = [f"The dimension {d.text} does not fit the drawing's scale; check it." for d in rejected]
    if len(used) == 1:
        warnings.append(f"Sizes are scaled from one dimension ({used[0].text}); check them.")
    return SheetScale(scale, used, rejected, dims, warnings)
