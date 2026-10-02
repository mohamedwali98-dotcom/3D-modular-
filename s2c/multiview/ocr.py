"""Handwritten dimension values: find the text, read it, link it to a face axis or a hole. Spec section 4.2.
The reader only reads what the user wrote; it never estimates a size."""
from __future__ import annotations

import logging
import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import cv2
import numpy as np

from s2c.multiview.outline import PixelOutline, foreground
from s2c.reading import MIN_CONFIDENCE, Crop, ReaderRun, ReadingService

log = logging.getLogger(__name__)

Reader = Callable[[np.ndarray], tuple[str, float]]  # BGR crop -> (text, confidence)
BatchReader = Callable[[list[np.ndarray]], list[tuple[str, float]] | None]  # every crop of one image; None on failure
_DIAMETER_SIGNS = "⌀ØøΦφ∅"
_VALUE = re.compile(r"^(⌀|D|R)?(\d+(?:\.\d+)?)$")
STROKE_PX = 25
MAX_CROPS = 16    # the text boxes nearest the part; the rest of a sheet is rarely a dimension
NEAR_HOLE = 0.25  # a ⌀ or R links to a hole within this share of the part's longer side, plus the hole's radius


@dataclass
class Reading:
    value_mm: float
    kind: Literal["linear", "diameter", "radius"]
    bbox: tuple[int, int, int, int]
    confidence: float
    text: str
    confirmed: bool = True  # False: the readers did not agree, so the user checks the value before it is trusted


@dataclass
class Linked:
    reading: Reading
    axis: Literal["a", "b", "ab"] | None  # face axis; "ab" is the diameter of a round outline
    hole_index: int | None


STRAY = "-'\",:;`"  # marks a reader returns for a tick or a dash beside the number


def parse_value(text: str) -> tuple[float, str] | None:
    tokens = [tok for tok in text.split() if tok.strip(STRAY)]  # a tick or a dash beside a number is no glyph
    text = " ".join(tokens)
    if len(tokens) > 1 and all(len(tok) == 1 for tok in tokens):
        if re.fullmatch(r"0\d+", "".join(tokens)):
            return None  # "0 1" is no number
        text = "".join(tokens)  # a handwriting reader spells a written number out glyph by glyph: "2 5" is 25
    if re.search(r"\d\s+\d", text):
        return None  # "12 48" is two values or a misread, never 1248
    t = text.strip().replace(" ", "").replace(",", ".")
    for sign in _DIAMETER_SIGNS:
        t = t.replace(sign, "⌀")
    if t.lower().endswith("mm"):
        t = t[:-2]
    t = t.rstrip(".")
    if len(t) > 1 and t[0] in "oO" and t[1].isdigit():
        t = "⌀" + t[1:]  # a handwritten ⌀ is often read as o
    if len(t) > 1:
        t = t[0] + t[1:].replace("O", "0").replace("o", "0")
    m = _VALUE.match(t)
    if not m:
        return None
    value = float(m.group(2))
    if not 0 < value < 2000:
        return None
    return value, {"⌀": "diameter", "D": "diameter", "R": "radius"}.get(m.group(1) or "", "linear")


def text_regions(image_bgr: np.ndarray, outline: PixelOutline) -> list[tuple[int, int, int, int]]:
    """Boxes of written text: ink left after erasing the outline, the openings and long straight lines."""
    ink = foreground(image_bgr)
    erase = np.zeros_like(ink)
    cv2.drawContours(erase, [outline.outer.reshape(-1, 1, 2).astype(np.int32)], -1, 255, STROKE_PX)
    for loop in outline.inner:
        cv2.drawContours(erase, [loop.reshape(-1, 1, 2).astype(np.int32)], -1, 255, STROKE_PX)
    for c in outline.circles:
        cv2.circle(erase, (round(c.cx), round(c.cy)), round(c.d / 2), 255, STROKE_PX)
    text = cv2.bitwise_and(ink, cv2.bitwise_not(erase))
    lines = cv2.bitwise_or(cv2.morphologyEx(text, cv2.MORPH_OPEN, np.ones((1, 60), np.uint8)),
                           cv2.morphologyEx(text, cv2.MORPH_OPEN, np.ones((60, 1), np.uint8)))
    words = cv2.dilate(cv2.subtract(text, lines), np.ones((9, 25), np.uint8))
    n, _, stats, _ = cv2.connectedComponentsWithStats(words)
    boxes = []
    for i in range(1, n):
        x, y, w, h = (int(v) for v in stats[i, :4])
        if 15 <= h <= 220 and w >= 12 and w / h <= 8:
            boxes.append((x, y, w, h))
    return boxes


def _gap(box: tuple[int, int, int, int], bbox: tuple[int, int, int, int]) -> float:
    """Distance from a text box's centre to the part's bounding box; 0 inside it."""
    x, y, w, h = box
    bx, by, bw, bh = bbox
    cx, cy = x + w / 2, y + h / 2
    return float(np.hypot(max(bx - cx, 0, cx - bx - bw), max(by - cy, 0, cy - by - bh)))


def crops_for(image_bgr: np.ndarray, outline: PixelOutline) -> list[Crop]:
    """At most MAX_CROPS text crops, nearest the part first, with a 6 px margin."""
    boxes = sorted(text_regions(image_bgr, outline), key=lambda b: _gap(b, outline.bbox))[:MAX_CROPS]
    return [Crop(image_bgr[max(y - 6, 0): y + h + 6, max(x - 6, 0): x + w + 6], (x, y, w, h))
            for x, y, w, h in boxes]


def decide(runs: list[ReaderRun], k: int) -> tuple[str, tuple[float, str], float, bool] | None:
    """Crop k's reads -> (text, (value, kind), confidence, confirmed), or None when no read is usable.
    Usable: parses as a value, and a calibrated reader is at least MIN_CONFIDENCE sure.
    Confirmed: two or more usable reads give the same number, or the only configured reader is calibrated.
    The first usable read in reader order gives the text: the VLM goes first because it keeps ⌀ and R."""
    usable = []
    for run in runs:
        if run.results is None:
            continue
        r = run.results[k]
        parsed = parse_value(r.text)
        if parsed is None or (run.calibrated and r.confidence < MIN_CONFIDENCE):
            continue
        usable.append((r, parsed))
    if not usable:
        return None
    first, parsed = usable[0]
    agree = len(usable) >= 2 and len({p[0] for _, p in usable}) == 1
    alone = len(runs) == 1 and runs[0].calibrated
    return first.text, parsed, first.confidence, agree or alone


def read_values(image_bgr: np.ndarray, outline: PixelOutline, service: ReadingService | None) -> list[Reading]:
    """Every reader reads every crop in parallel; decide() says which values the user must check."""
    if service is None:
        return []
    crops = crops_for(image_bgr, outline)
    runs = service.read(crops)
    out = []
    for k, crop in enumerate(crops):
        decided = decide(runs, k)
        if decided is None:
            log.info("no usable read at %s", crop.box)
            continue
        text, (value, kind), confidence, confirmed = decided
        out.append(Reading(value, kind, crop.box, float(confidence), text, confirmed))
    return out


def link(readings: list[Reading], outline: PixelOutline) -> list[Linked]:
    """Below or above the outline: axis a. Left or right: axis b. Diameters: the nearest hole, if it is near."""
    bx, by, bw, bh = outline.bbox
    reach = NEAR_HOLE * max(bw, bh)
    out = []
    for r in readings:
        x, y, w, h = r.bbox
        cx, cy = x + w / 2, y + h / 2
        inside = bx <= cx <= bx + bw and by <= cy <= by + bh
        if r.kind != "linear":
            dist = [float(np.hypot(c.cx - cx, c.cy - cy)) - c.d / 2 for c in outline.circles]
            k = min(range(len(dist)), key=dist.__getitem__) if dist else None
            if k is not None and (inside or not outline.circular) and dist[k] <= reach:
                out.append(Linked(r, None, k))
            else:
                out.append(Linked(r, "ab" if outline.circular else None, None))
        elif cy < by or cy > by + bh:
            out.append(Linked(r, "a", None))
        elif cx < bx or cx > bx + bw:
            out.append(Linked(r, "b", None))
        else:
            out.append(Linked(r, None, None))  # written inside the part: kept for the lab view, not linked
    return out


def trocr_reader(model_name: str | None = None) -> Reader:
    """One crop at a time over the shared TrOCR reader (s2c.reading.trocr), for callers that need a function.
    The pipeline passes the shared reader itself, so every crop of an image is read in one batch."""
    from s2c.reading.trocr import TrocrReader
    shared = TrocrReader(model_name)

    def read(crop_bgr: np.ndarray) -> tuple[str, float]:
        out = shared.read([Crop(crop_bgr, (0, 0, crop_bgr.shape[1], crop_bgr.shape[0]))])
        return (out[0].text, out[0].confidence) if out else ("", 0.0)

    return read
