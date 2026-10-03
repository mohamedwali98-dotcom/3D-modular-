"""Stage 2: find text boxes, read them with every reader, decide what each text is."""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Literal

import cv2
import numpy as np

from s2c import obs
from s2c.reading import MIN_CONFIDENCE, Crop, Reader, ReadingService
from s2c.sketch.grammar import Parsed, match_label, parse_text
from s2c.sketch.models import Reading, SketchAbstain, ViewName

log = logging.getLogger(__name__)
PAD = 4


@dataclass
class TextItem:
    id: str
    box: tuple[int, int, int, int]
    readings: list[Reading]
    parsed: Parsed | None
    label: ViewName | None
    role: Literal["label", "dimension", "other"]
    badge: Literal["written", "uncertain"] | None
    confidence: float
    candidates: list[float] = field(default_factory=list)


def _line_like(w: int, h: int, stroke: float) -> bool:
    return min(w, h) <= 1.8 * stroke and max(w, h) >= 3 * min(w, h)


def _dash_members(stats: np.ndarray, idx: list[int], stroke: float) -> set[int]:
    """Line-like pieces with at least two similar pieces on the same line: pieces of a dashed line."""
    pieces = [i for i in idx if _line_like(stats[i, 2], stats[i, 3], stroke)]
    out = set()
    for i in pieces:
        xi, yi, wi, hi = stats[i, :4]
        horizontal = wi > hi
        near = 0
        for j in pieces:
            if j == i:
                continue
            xj, yj, wj, hj = stats[j, :4]
            if (wj > hj) != horizontal:
                continue
            if horizontal and abs((yi + hi / 2) - (yj + hj / 2)) < 2 * stroke \
                    and abs(xi - xj) < 3 * max(wi, wj):
                near += 1
            if not horizontal and abs((xi + wi / 2) - (xj + wj / 2)) < 2 * stroke \
                    and abs(yi - yj) < 3 * max(hi, hj):
                near += 1
        if near >= 2:
            out.add(i)
    return out


def _strip_lines(ink: np.ndarray, length: int) -> tuple[np.ndarray, np.ndarray]:
    """Remove straight horizontal and vertical runs of at least `length` px. Returns what is left and
    the removed band (the runs grown by one pixel)."""
    lines = cv2.morphologyEx(ink, cv2.MORPH_OPEN, np.ones((1, length), np.uint8)) | \
        cv2.morphologyEx(ink, cv2.MORPH_OPEN, np.ones((length, 1), np.uint8))
    band = cv2.dilate(lines, np.ones((3, 3), np.uint8))
    return cv2.bitwise_and(ink, cv2.bitwise_not(band)), band


def _cut_from_line(pts: np.ndarray, ink: np.ndarray, band: np.ndarray, near_band: np.ndarray,
                   stroke: float) -> bool:
    """A thin piece touching a removed line was cut out of line geometry when it runs along that line
    (an arrowhead half) or when its ink goes on past the line (an extension line or a leader crossing
    it). A glyph stroke that only ends at the line (a "1" standing on it) is neither. The walk past the
    end is long enough to cross a thick outline at 45 degrees."""
    if near_band[pts[:, 1], pts[:, 0]].mean() >= 0.3:
        return True
    centre = pts.mean(0)
    _, _, vt = np.linalg.svd(pts - centre, full_matrices=False)
    u, v = vt[0], vt[1]
    t = (pts - centre) @ u
    h, w = ink.shape
    for end, sign in ((pts[np.argmin(t)], -1), (pts[np.argmax(t)], 1)):
        crossed = False
        for step in range(1, int(5 * stroke) + 1):
            q = np.round(end + sign * step * u + np.outer((-1, 0, 1), v)).astype(int)
            q = q[(q[:, 0] >= 0) & (q[:, 0] < w) & (q[:, 1] >= 0) & (q[:, 1] < h)]
            if band[q[:, 1], q[:, 0]].any():
                crossed = True
            elif crossed and ink[q[:, 1], q[:, 0]].any():
                return True
    return False


def _line_debris(labels: np.ndarray, stats: np.ndarray, idx: list[int], ink: np.ndarray, band: np.ndarray,
                 stroke: float) -> set[int]:
    """Thin pieces cut out of line geometry by the line removal: arrowhead halves left when the line
    through the head is removed, extension-line or outline stubs cut off by a crossing line, and the
    end of a slanted leader cut off by the outline it crosses (thin in its own direction, not in x or
    y)."""
    near_band = cv2.dilate(band, np.ones((5, 5), np.uint8)) > 0
    touching = set(np.unique(labels[cv2.dilate(band, np.ones((3, 3), np.uint8)) > 0]).tolist())
    out = set()
    for i in idx:
        if i not in touching:
            continue
        x, y, w, h = stats[i, :4]
        ys, xs = np.nonzero(labels[y:y + h, x:x + w] == i)
        pts = np.stack([xs + x, ys + y], 1)
        (_, _), (a, b), _ = cv2.minAreaRect(pts.astype(np.float32))
        thin = _line_like(stats[i, 2], stats[i, 3], stroke) or _line_like(max(a, b) + 1, min(a, b) + 1, stroke)
        if thin and _cut_from_line(pts, ink, band > 0, near_band, stroke):
            out.add(i)
    return out


def _pieces(ink: np.ndarray, length: int, char_max: float, stroke: float):
    """Components left after removing straight runs of at least `length` px, split into glyph
    candidates and the ones that are not glyphs (dash pieces, line debris)."""
    rest, band = _strip_lines(ink, length)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(rest, connectivity=8)
    idx = [i for i in range(1, n) if stats[i, 4] >= 10 and stats[i, 3] <= char_max
           and stats[i, 2] <= 1.5 * char_max]
    drop = _dash_members(stats, idx, stroke) | _line_debris(labels, stats, idx, ink, band, stroke)
    return labels, stats, [i for i in idx if i not in drop]


def find_text_boxes(ink: np.ndarray, stroke_px: float) -> list[tuple[int, int, int, int]]:
    h, w = ink.shape
    long_side = max(h, w)
    L = max(40, int(0.04 * long_side))
    char_max = 0.06 * long_side
    labels, stats, cand = _pieces(ink, L, char_max, stroke_px)
    heights = [stats[i, 3] for i in cand]
    glyph_h = float(np.median(heights)) if heights else 0.0
    if heights and 2 * glyph_h < L:
        # Deliberate second pass, from the original ink: now the glyph height is known, straight runs
        # longer than two glyphs are lines too (the lines of a short dimension, the extension lines a
        # digit touches), all shorter than L.
        labels, stats, cand = _pieces(ink, int(2 * glyph_h), char_max, stroke_px)
    glyphs = np.isin(labels, cand).astype(np.uint8) * 255
    k = max(5, int(0.6 * glyph_h)) if heights else 5
    glued = cv2.dilate(glyphs, np.ones((k, k), np.uint8))
    m, _, st, _ = cv2.connectedComponentsWithStats(glued, connectivity=8)
    boxes = []
    for i in range(1, m):
        x, y, bw, bh = (int(v) for v in st[i, :4])
        x, y, bw, bh = x + k // 2, y + k // 2, bw - k, bh - k  # undo the dilation margin
        if bw < 6 or bh < 6 or max(bw, bh) / max(min(bw, bh), 1) > 10:
            continue
        boxes.append((max(0, x - PAD), max(0, y - PAD), bw + 2 * PAD, bh + 2 * PAD))
    return boxes


def _crops(sheet: np.ndarray, box) -> list[Crop]:
    x, y, w, h = box
    img = sheet[y:y + h, x:x + w]
    out = [Crop(img, box)]
    if h > 1.3 * w:  # maybe sideways text on a vertical dimension
        out += [Crop(cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE), box),
                Crop(cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE), box)]
    return out


def _best(results) -> tuple[str, float]:
    """Among one reader's readings of the orientations of one box: parseable first, then labels,
    then the most confident."""
    ranked = sorted(results, key=lambda r: (parse_text(r.text) is not None,
                                            match_label(r.text) is not None, r.confidence),
                    reverse=True)
    return ranked[0].text, ranked[0].confidence


def read_texts(sheet_bgr: np.ndarray, boxes, readers: list[Reader]) -> list[TextItem] | SketchAbstain:
    if not boxes:
        return []
    groups = [_crops(sheet_bgr, b) for b in boxes]
    flat = [c for g in groups for c in g]
    runs = ReadingService(readers).read(flat)
    per_reader = [(run.name, run.results) for run in runs if run.results is not None]
    if not per_reader:
        return SketchAbstain(stage="text", reason="readers_unavailable",
                             remedy="Reading service unavailable. Retry in a minute.")
    items: list[TextItem] = []
    start = 0
    for n, (box, group) in enumerate(zip(boxes, groups)):
        readings = []
        for name, res in per_reader:
            text, conf = _best(res[start:start + len(group)])
            readings.append(Reading(reader=name, text=text, confidence=conf))
        start += len(group)
        items.append(_decide(f"t{n}", box, readings, len(readers)))
    return items


def _decide(tid: str, box, readings: list[Reading], n_readers: int) -> TextItem:
    parses = [(r, parse_text(r.text)) for r in readings]
    parsed = [(r, p) for r, p in parses if p is not None]
    if parsed:
        values: list[float] = []
        for _, p in parsed:
            if p.value not in values:
                values.append(p.value)
        agree = (len(parsed) == n_readers >= 2 and len(values) == 1
                 and all(r.confidence >= MIN_CONFIDENCE for r, _ in parsed))
        return TextItem(tid, box, readings, parsed[0][1], None, "dimension",
                        "written" if agree else "uncertain",
                        max(r.confidence for r, _ in parsed), values)
    for r in readings:
        name = match_label(r.text)
        if name:
            return TextItem(tid, box, readings, None, name, "label", None, r.confidence)
    return TextItem(tid, box, readings, None, None, "other", None,
                    max((r.confidence for r in readings), default=0.0))


_DETECTOR = None


def _paddle_boxes(sheet_bgr: np.ndarray) -> list[tuple[int, int, int, int]]:
    """PaddleOCR's pretrained text detector (PP-OCRv5 det). Checked against PaddleOCR 3.x
    (`TextDetection(...).predict(...)` returning `dt_polys`)."""
    global _DETECTOR
    if _DETECTOR is None:
        from paddleocr import TextDetection

        _DETECTOR = TextDetection(
            model_name=os.environ.get("SKETCH_PADDLE_DET", "PP-OCRv5_server_det"))
    res = next(iter(_DETECTOR.predict(sheet_bgr, batch_size=1)))
    data = res.json.get("res", res.json) if hasattr(res, "json") else dict(res)
    boxes = []
    for poly in data.get("dt_polys", []):
        p = np.asarray(poly, float)
        x0, y0 = p.min(0)
        x1, y1 = p.max(0)
        boxes.append((max(0, int(x0) - PAD), max(0, int(y0) - PAD),
                      int(x1 - x0) + 2 * PAD, int(y1 - y0) + 2 * PAD))
    return boxes


def detect_text_boxes(sheet_bgr: np.ndarray, ink: np.ndarray, stroke_px: float):
    """Pretrained detector first; the classical detector when PaddleOCR is missing or fails."""
    choice = os.environ.get("SKETCH_TEXT_DETECTOR", "auto")
    if choice in ("auto", "paddle"):
        try:
            return _paddle_boxes(sheet_bgr)
        except Exception as exc:  # noqa: BLE001 - missing optional dependency, or a model error
            if choice == "paddle":  # asked for by name: said out loud; on auto a missing PaddleOCR is normal
                obs.fallback("paddle_text_detector", exc)
            else:
                obs.count("s2c_fallbacks_total", name="paddle_text_detector")
    return find_text_boxes(ink, stroke_px)


def erase_mask(texts: list[TextItem], shape, pad: int = 2) -> np.ndarray:
    mask = np.zeros(shape[:2], np.uint8)
    for t in texts:
        if t.role in ("label", "dimension"):
            x, y, w, h = t.box
            mask[max(0, y - pad): y + h + pad, max(0, x - pad): x + w + pad] = 255
    return mask
