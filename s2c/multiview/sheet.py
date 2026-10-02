"""One image holding several orthographic views -> drawings and views (spec 3.1), and the face each view shows
(spec 3.2)."""
from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass, field
from itertools import combinations
from typing import TYPE_CHECKING

import cv2
import numpy as np
from scipy import ndimage

if TYPE_CHECKING:
    from s2c.multiview.ocr import Reader

log = logging.getLogger(__name__)

Box = tuple[int, int, int, int]  # x, y, w, h in image pixels

INK_DELTA = 40          # grey levels away from the median border grey
BORDER_NEAR = 0.05      # a frame "nearly touches" a side: within this share of the image size
BORDER_FILL = 0.05      # ... and inks under this share of its box
BORDER_SIDE = 0.9       # ... and inks this share of each side of its box, which a round or L-shaped view does not
BORDER_BAND = 0.03
BORDER_EDGE = 0.02      # ... or keeps this share of its ink within an edge band this wide (share of its short side)
BORDER_EDGE_INK = 0.9
SEPARATOR_INK = 0.9     # a row or column this full of ink is a separator band
DILATE = 0.008
SPECK = 0.0002
LABEL_HEIGHT = 0.06
LABEL_ASPECT = 2.5
LABEL_REACH = 1.5
TEXT_GAP = 1.0          # words and letters of one label line join across gaps up to this many heights
DRAWING_GAP = 3.0       # views farther apart than this many median view sizes are separate drawings
ALIGN = 0.10            # centres of two views in one row (column) within this share of their height (width)
EXTENT = 0.25           # ... and their heights (widths) within this share: views in a row share the front's height
MIN_VIEW = 0.15         # a view shorter than this share of the drawing's largest cannot make a sheet (dimension text)
LINE_ART = 0.35         # a view is line art when its ink fills under this share of its filled outline (spec 3.3)
SCALE = 0.08            # a name's implied size and the view's size agree within this share (spec 3.2)
SLACK = 0.004           # ... or within this share of the long side, for thin views and stroke widths
SPUR = 0.006            # filled rows or columns this thin (share of the long side) are chain lines, not the view
BODY = 0.01             # a closed outline filling at least this share of the page is a view (sketches)
MEND = 3.0              # ... after gaps up to this many line widths are closed (a hand corner that does not meet)
STRIP_REACH = 1.6       # a dimension's number sits within this many text heights of its dimension line
STRIP_SHARE = 0.2       # erasing a dimension line takes under this share off the view; a real edge opens it all
STRIP_THIN = 0.5        # a strip is under this share of its length across
STRIP_TRIES = 8
TEXT_PAD = 4            # the margin s2c.sketch.text puts around a text box
MARK_AREA = 0.15        # an unnamed view under this share of the largest view's area is a mark: text, balloon, note
CROP_MARGIN = 0.04
RING_DIRECTIONS = 36    # directions around a circle's centre
RING_COVER = 0.85       # a circle inks this share of them at its radius; a centre line's four arms only a few
AUTO, SKIP = "auto", "skip"


@dataclass
class View:
    box: Box
    label_box: Box | None = None
    line_art: bool = True
    body: np.ndarray | None = field(default=None, repr=False, compare=False)  # the closed outline in `box` (sketches)


@dataclass
class Drawing:
    views: list[View]


@dataclass
class Sheet:
    drawings: list[Drawing]
    shape: tuple[int, int]
    warnings: list[str] = field(default_factory=list)
    stroke_px: float | None = None  # a hand sketch's line width: its views are drawn again straight before reading


def split_sheet(image_bgr: np.ndarray) -> Sheet:
    """The drawings and views of one image. An image with no drawing of 2 or more views is not a sheet."""
    ink = ink_mask(image_bgr)
    h, w = ink.shape
    _remove_border(ink)
    cuts = _cut_separators(ink)
    k = max(3, round(DILATE * max(h, w)))
    grown = cv2.dilate(ink, np.ones((k, k), np.uint8))
    for axis, a, b in cuts:
        if axis == "row":
            grown[a: b + 1] = 0
        else:
            grown[:, a: b + 1] = 0
    n, comp = cv2.connectedComponents(grown, connectivity=8)
    boxes, count = _ink_boxes(ink, comp, n)
    boxes = {i: b for i, b in boxes.items() if count[i] >= SPECK * h * w}
    if not boxes:
        return Sheet([], (h, w), ["No drawing found in the image."])

    groups, labels = _sort(boxes, h, k)
    views = []
    for ids in groups:
        box = _union(boxes[i] for i in ids)
        label = [labels[i] for i in ids if i in labels]
        art = _line_art(comp, ids, box, k, sum(int(count[i]) for i in ids))
        views.append(View(box, _union(b for bs in label for b in bs) if label else None, art))
    return Sheet(_drawings(views, cuts), (h, w), [])


def split_by_outlines(image_bgr: np.ndarray, stroke_px: float) -> Sheet:
    """The views of a sketch page as closed outlines (spec 3.1). Gaps up to about two line widths are mended, the
    outlines filled, and lines up to that width opened away: dimension, miter and centre lines and text enclose no
    area, so they never join two views or enlarge one. Every filled blob of at least BODY of the page is a view.
    A dimension line whose extension lines touch the outline does close a strip onto the view; it is taken off again
    where its number is written beside it (`_trim_strips`)."""
    from s2c.sketch.text import find_text_boxes

    ink = ink_mask(image_bgr)
    h, w = ink.shape
    _remove_border(ink)
    k = round(MEND * max(1.0, stroke_px)) | 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    closed = cv2.morphologyEx(ink, cv2.MORPH_CLOSE, kernel)
    texts = _words(find_text_boxes(ink, stroke_px))
    bodies, masks = [], []
    for body in _bodies(closed, kernel, BODY * h * w):
        body = cv2.dilate(_trim_strips(closed, body, texts, kernel).astype(np.uint8), kernel) > 0  # the outline's
        ys, xs = np.nonzero(body)                                         # ink, which the opening rounded off
        x, y = int(xs.min()), int(ys.min())
        bodies.append((x, y, int(np.ptp(xs)) + 1, int(np.ptp(ys)) + 1))
        masks.append(body[y: y + bodies[-1][3], x: x + bodies[-1][2]])
    if not bodies:
        return Sheet([], (h, w), ["No closed outline found in the image."])
    rest = ink.copy()
    for x, y, bw, bh in bodies:
        rest[y: y + bh, x: x + bw] = 0
    pool = dict(enumerate(bodies))
    labels: dict[int, list[Box]] = {}
    m, parts = cv2.connectedComponents(rest, connectivity=8)
    boxes, count = _ink_boxes(rest, parts, m)
    small = [i for i, b in boxes.items() if b[3] < LABEL_HEIGHT * h and count[i] >= SPECK * h * w]
    for g in _text_lines(small, boxes):
        box = _union(boxes[i] for i in g)
        if box[2] >= LABEL_ASPECT * box[3] and (owner := _label_owner(box, pool)) is not None:
            labels.setdefault(owner, []).append(box)
    views = [View(b, _union(labels[i]) if i in labels else None, True, masks[i]) for i, b in pool.items()]
    return Sheet(_drawings(views, []), (h, w), [])


def _bodies(closed: np.ndarray, kernel: np.ndarray, least: float) -> list[np.ndarray]:
    """The closed ink filled, lines opened away: masks of the blobs of at least `least` px."""
    solid = cv2.morphologyEx(ndimage.binary_fill_holes(closed > 0).astype(np.uint8), cv2.MORPH_OPEN, kernel)
    n, comp, stats, _ = cv2.connectedComponentsWithStats(solid, connectivity=8)
    return [comp == i for i in range(1, n) if stats[i][4] >= least]


def _trim_strips(closed: np.ndarray, body: np.ndarray, texts: list[Box], kernel: np.ndarray) -> np.ndarray:
    """The body without the strips its dimension lines close onto it. An edge of the body's outline with a text
    box beside it is erased; when that takes off a part of the body (under STRIP_SHARE of it) and the text is not
    in what stays, the edge was a dimension line. Text inside the part taken off ("60" written between the view and
    its dimension line) needs that part to be a strip along the edge, so a small pin with a stray mark on it stays.
    Erasing a real outline edge opens the whole view, which is too much to take off."""
    k = kernel.shape[0]
    for _ in range(STRIP_TRIES):
        ys, xs = np.nonzero(body)
        x0, y0 = max(0, int(xs.min()) - 2 * k), max(0, int(ys.min()) - 2 * k)
        x1, y1 = min(body.shape[1], int(xs.max()) + 2 * k + 1), min(body.shape[0], int(ys.max()) + 2 * k + 1)
        own = body[y0:y1, x0:x1]
        contours, _ = cv2.findContours(own.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        outer = max(contours, key=cv2.contourArea)
        poly = cv2.approxPolyDP(outer, k, True)[:, 0]
        cut = None
        for p, q in zip(poly, np.roll(poly, -1, axis=0)):
            for t in texts:  # each number beside the edge on its own: a big "word" of dashes must not open the view
                tx, ty, tw, th = t[0] - x0, t[1] - y0, t[2], t[3]
                if not _beside((tx, ty, tw, th), p, q):
                    continue
                trial = closed[y0:y1, x0:x1].copy()
                cv2.line(trial, tuple(int(v) for v in p), tuple(int(v) for v in q), 0, k + 2)
                trial[max(0, ty + TEXT_PAD): max(0, ty + th - TEXT_PAD),  # the number goes too: its mended
                      max(0, tx + TEXT_PAD): max(0, tx + tw - TEXT_PAD)] = 0  # loops would stay stuck to the view
                stay = max(_bodies(trial, kernel, 0), key=lambda m: int((m & own).sum()), default=None)
                if stay is None:
                    continue
                stay = stay & own
                lost = own & ~stay
                if not 0 < lost.sum() <= STRIP_SHARE * own.sum():
                    continue
                box = (slice(max(0, ty), max(0, ty + th)), slice(max(0, tx), max(0, tx + tw)))
                size = max(1, stay[box].size)
                if stay[box].sum() > 0.3 * size or (lost[box].sum() > 0.5 * size and not _strip(lost, p, q)):
                    continue
                cut = (trial, stay)
                break
            if cut:
                break
        if cut is None:
            return body
        closed[y0:y1, x0:x1] = cut[0]
        body = body.copy()
        body[y0:y1, x0:x1] = cut[1]
    return body


def _words(boxes: list[Box]) -> list[Box]:
    """Glyph boxes joined into words, along a row or (a number written along a vertical line) a column."""
    sets = _Sets(range(len(boxes)))
    for i, j in combinations(range(len(boxes)), 2):
        a, b = boxes[i], boxes[j]
        for p, s, o, t in ((0, 2, 1, 3), (1, 3, 0, 2)):  # neighbours along x (overlapping in y), then along y
            over = min(a[o] + a[t], b[o] + b[t]) - max(a[o], b[o])
            gap = max(a[p], b[p]) - min(a[p] + a[s], b[p] + b[s])
            if over >= 0.5 * min(a[t], b[t]) and gap <= 0.6 * max(a[t], b[t]):
                sets.join(i, j)
    return [_union(boxes[i] for i in g) for g in sets.groups()]


def _beside(t: Box, p: np.ndarray, q: np.ndarray) -> bool:
    """The text box's centre is within STRIP_REACH text heights of the edge pq, over its span."""
    tx, ty, tw, th = t
    along = q.astype(float) - p
    length = float(np.hypot(*along))
    if length < min(tw, th):
        return False
    u = along / length
    c = np.array([tx + tw / 2, ty + th / 2]) - p
    s = float(c @ u)
    return 0 <= s <= length and abs(float(c[0] * u[1] - c[1] * u[0])) <= STRIP_REACH * min(tw, th)


def _strip(lost: np.ndarray, p: np.ndarray, q: np.ndarray) -> bool:
    """The part taken off is a strip along pq: across it, under STRIP_THIN of its length along it."""
    ys, xs = np.nonzero(lost)
    u = (q.astype(float) - p) / max(1e-6, float(np.hypot(*(q.astype(float) - p))))
    along = xs * u[0] + ys * u[1]
    across = xs * u[1] - ys * u[0]
    return float(np.ptp(across)) <= STRIP_THIN * float(np.ptp(along))


def is_sheet(sheet: Sheet, align: float = ALIGN, extent: float = EXTENT) -> bool:
    """True when some drawing holds two line-drawn views that line up in a row or a column the way orthographic
    views do: centres within 10 % and the shared extent within 25 % (hand sketches: wider, `align` and `extent`).
    Photos (filled blobs), a part beside its shadow or a coin, and dimension text beside a sketch do not (Review
    Focus 1)."""
    return any(_aligned_pair(d.views, align, extent) for d in sheet.drawings)


def ink_mask(image_bgr: np.ndarray) -> np.ndarray:
    """Pixels that differ from the median border grey by more than INK_DELTA are 255."""
    if image_bgr.ndim == 2:
        gray = image_bgr
    else:
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGRA2GRAY if image_bgr.shape[2] == 4 else cv2.COLOR_BGR2GRAY)
    edge = np.concatenate([gray[0], gray[-1], gray[:, 0], gray[:, -1]])
    diff = np.abs(gray.astype(np.int16) - int(np.median(edge)))
    return np.where(diff > INK_DELTA, 255, 0).astype(np.uint8)


def _remove_border(ink: np.ndarray) -> None:
    """Erase a sheet frame: near all four sides, thin, and inked along every side of its box. Thin means under
    BORDER_FILL of its box, or nearly all ink in a narrow band along the box edges: on a small sheet the frame's
    stroke alone inks over BORDER_FILL."""
    h, w = ink.shape
    n, comp, stats, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
    for i in range(1, n):
        x, y, bw, bh, area = (int(v) for v in stats[i])
        if max(x, w - x - bw) > BORDER_NEAR * w or max(y, h - y - bh) > BORDER_NEAR * h:
            continue
        own = comp[y: y + bh, x: x + bw] == i
        edge = max(3, round(BORDER_EDGE * min(bw, bh)))
        if area >= BORDER_FILL * bw * bh and own[edge:-edge, edge:-edge].sum() > (1 - BORDER_EDGE_INK) * area:
            continue
        band = max(3, round(BORDER_BAND * min(bw, bh)))
        sides = (own[:band].any(0), own[-band:].any(0), own[:, :band].any(1), own[:, -band:].any(1))
        if min(float(s.mean()) for s in sides) >= BORDER_SIDE:
            ink[y: y + bh, x: x + bw][own] = 0


def _runs(flags: np.ndarray) -> list[tuple[int, int]]:
    edges = np.diff(np.concatenate([[0], flags.astype(np.int8), [0]]))
    return list(zip(np.nonzero(edges == 1)[0].tolist(), (np.nonzero(edges == -1)[0] - 1).tolist()))


def _cut_separators(ink: np.ndarray) -> list[tuple[str, int, int]]:
    """Erase separator bands and return them. A band at the edge of the ink separates nothing and is kept, so the
    top edge of a view that spans the image is never cut away."""
    cuts = []
    for axis in ("row", "col"):
        full = (ink > 0).mean(axis=1 if axis == "row" else 0) > SEPARATOR_INK
        for a, b in _runs(full):
            before, after = (ink[:a], ink[b + 1:]) if axis == "row" else (ink[:, :a], ink[:, b + 1:])
            if before.any() and after.any():
                cuts.append((axis, a, b))
                if axis == "row":
                    ink[a: b + 1] = 0
                else:
                    ink[:, a: b + 1] = 0
    return cuts


def _ink_boxes(ink: np.ndarray, comp: np.ndarray, n: int) -> tuple[dict[int, Box], np.ndarray]:
    """Box of the ink (not of the dilated blob) of each component, and its ink pixel count."""
    ys, xs = np.nonzero(ink)
    ids = comp[ys, xs]
    count = np.bincount(ids, minlength=n)
    lo_x, lo_y = np.full(n, ink.shape[1]), np.full(n, ink.shape[0])
    hi_x, hi_y = np.full(n, -1), np.full(n, -1)
    np.minimum.at(lo_x, ids, xs)
    np.minimum.at(lo_y, ids, ys)
    np.maximum.at(hi_x, ids, xs)
    np.maximum.at(hi_y, ids, ys)
    boxes = {i: (int(lo_x[i]), int(lo_y[i]), int(hi_x[i] - lo_x[i] + 1), int(hi_y[i] - lo_y[i] + 1))
             for i in range(1, n) if count[i] > 0}
    return boxes, count


def _union(boxes) -> Box:
    boxes = list(boxes)
    x0, y0 = min(b[0] for b in boxes), min(b[1] for b in boxes)
    x1, y1 = max(b[0] + b[2] for b in boxes), max(b[1] + b[3] for b in boxes)
    return x0, y0, x1 - x0, y1 - y0


def _inside(inner: Box, outer: Box, tol: int = 0) -> bool:
    return (inner[0] >= outer[0] - tol and inner[1] >= outer[1] - tol
            and inner[0] + inner[2] <= outer[0] + outer[2] + tol and inner[1] + inner[3] <= outer[1] + outer[3] + tol)


class _Sets:
    def __init__(self, items):
        self.parent = {i: i for i in items}

    def find(self, i):
        while self.parent[i] != i:
            self.parent[i] = self.parent[self.parent[i]]
            i = self.parent[i]
        return i

    def join(self, a, b):
        self.parent[self.find(a)] = self.find(b)

    def groups(self) -> list[list]:
        out: dict = {}
        for i in self.parent:
            out.setdefault(self.find(i), []).append(i)
        return list(out.values())


def _text_lines(small: list[int], boxes: dict[int, Box]) -> list[list[int]]:
    """Small components on one text line, close together: the letters and words of one label."""
    sets = _Sets(small)
    for i, j in combinations(small, 2):
        a, b = boxes[i], boxes[j]
        tall = max(a[3], b[3])
        if min(a[3], b[3]) < 0.5 * tall or abs((a[1] + a[3] / 2) - (b[1] + b[3] / 2)) > 0.5 * tall:
            continue
        if max(b[0] - a[0] - a[2], a[0] - b[0] - b[2]) <= TEXT_GAP * tall:
            sets.join(i, j)
    return sets.groups()


def _label_owner(box: Box, pool: dict[int, Box]) -> int | None:
    """The nearest view that `box` sits directly above or below, within LABEL_REACH of its height."""
    x, y, w, h = box
    best = None
    for vid, (vx, vy, vw, vh) in pool.items():
        if _inside(box, (vx, vy, vw, vh)):
            return None
        if x + w <= vx or vx + vw <= x:
            continue
        if y >= vy + vh:
            d = y - (vy + vh)
        elif y + h <= vy:
            d = vy - (y + h)
        else:
            continue
        if d <= LABEL_REACH * h and (best is None or d < best[0]):
            best = (d, vid)
    return None if best is None else best[1]


def _sort(boxes: dict[int, Box], image_h: int, k: int) -> tuple[list[list[int]], dict[int, list[Box]]]:
    """Components -> views (component ids, largest first) and the label boxes of each view's main component.

    Labels are short, wide text lines directly above or below a view. A text line with no view next to it stays a
    view. A second pass lets a label sit under a thin view that is itself text-shaped (the top view of a plate)."""
    small = [i for i, b in boxes.items() if b[3] < LABEL_HEIGHT * image_h]
    lines = [g for g in _text_lines(small, boxes)
             if (u := _union(boxes[i] for i in g))[3] < LABEL_HEIGHT * image_h and u[2] >= LABEL_ASPECT * u[3]]
    in_text = {i for g in lines for i in g}
    pool = {i: b for i, b in boxes.items() if i not in in_text}
    labels: dict[int, list[Box]] = {}
    pending = []
    for g in lines:
        box = _union(boxes[i] for i in g)
        owner = _label_owner(box, pool)
        if owner is None:
            pending.append((g, box))
        else:
            labels.setdefault(owner, []).append(box)
    for g, box in pending:
        pool.update({i: boxes[i] for i in g})
    for g, box in sorted(pending, key=lambda p: p[1][2]):  # the narrower of a label and a flat view is the label
        others = {i: b for i, b in pool.items() if i not in g}
        owner = _label_owner(box, others)
        if owner is not None:
            labels.setdefault(owner, []).append(box)
            for i in g:
                del pool[i]

    views: list[list[int]] = []
    for i in sorted(pool, key=lambda i: -pool[i][2] * pool[i][3]):
        home = next((v for v in views if _inside(pool[i], pool[v[0]], k // 2)), None)
        if home is None:
            views.append([i])
        else:
            home.append(i)
    owner_of = {i: v[0] for v in views for i in v}
    joined: dict[int, list[Box]] = {}
    for owner, bs in labels.items():
        if owner in owner_of:
            joined.setdefault(owner_of[owner], []).extend(bs)
    return views, joined


def _line_art(comp: np.ndarray, ids: list[int], box: Box, k: int, ink_count: int) -> bool:
    """Ink fills under LINE_ART of the view's filled outline (dilated, so small gaps in a stroke do not matter)."""
    x, y, w, h = box
    y0, x0 = max(0, y - k), max(0, x - k)
    own = np.isin(comp[y0: y + h + k, x0: x + w + k], ids).astype(np.uint8)
    contours, _ = cv2.findContours(own, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(own)
    cv2.drawContours(filled, contours, -1, 1, cv2.FILLED)
    return ink_count < LINE_ART * int(filled.sum())


def _gap(a: Box, b: Box) -> float:
    dx = max(0, b[0] - (a[0] + a[2]), a[0] - (b[0] + b[2]))
    dy = max(0, b[1] - (a[1] + a[3]), a[1] - (b[1] + b[3]))
    return math.hypot(dx, dy)


def _separated(a: Box, b: Box, cuts) -> bool:
    for axis, lo, hi in cuts:
        p, s = (1, 3) if axis == "row" else (0, 2)
        if (a[p] + a[s] <= lo and b[p] > hi) or (b[p] + b[s] <= lo and a[p] > hi):
            return True
    return False


def _drawings(views: list[View], cuts) -> list[Drawing]:
    """Views closer than DRAWING_GAP median view sizes, with no separator band between them, form one drawing."""
    limit = DRAWING_GAP * float(np.median([max(v.box[2], v.box[3]) for v in views]))
    sets = _Sets(range(len(views)))
    for i, j in combinations(range(len(views)), 2):
        a, b = views[i].box, views[j].box
        if _gap(a, b) <= limit and not _separated(a, b, cuts):
            sets.join(i, j)
    drawings = []
    for group in sets.groups():
        members = sorted((views[i] for i in group), key=lambda v: (v.box[1] + v.box[3] / 2, v.box[0]))
        drawings.append(Drawing(members))
    return sorted(drawings, key=lambda d: (min(v.box[1] for v in d.views), min(v.box[0] for v in d.views)))


def _in_line(a: Box, b: Box, row: bool, align: float = ALIGN, extent: float = EXTENT) -> bool:
    p, s = (1, 3) if row else (0, 2)
    size = max(a[s], b[s])
    return (abs((a[p] + a[s] / 2) - (b[p] + b[s] / 2)) <= align * size
            and abs(a[s] - b[s]) <= extent * size)


def _aligned_pair(views: list[View], align: float = ALIGN, extent: float = EXTENT) -> bool:
    art = [v.box for v in views if v.line_art]
    if len(art) < 2:
        return False
    biggest = max(max(b[2], b[3]) for b in art)
    art = [b for b in art if max(b[2], b[3]) >= MIN_VIEW * biggest]
    return any(_in_line(a, b, True, align, extent) or _in_line(a, b, False, align, extent)
               for a, b in combinations(art, 2))


@dataclass
class Naming:
    drawing: int                # index into sheet.drawings of the part drawing; -1 when the sheet has none
    faces: list[str]            # per view of that drawing: a face, "auto" (the user picks) or "skip" (a mark)
    projection: str             # "first" | "third"
    projection_source: str      # "symbol" | "labels" | "setting"; the symbol outranks labels
    warnings: list[str] = field(default_factory=list)


def named_count(naming: Naming) -> int:
    """Views given a face, each of which passed the scale check. The Studio treats an upload as a sheet only from 2
    on: `is_sheet` alone accepts two unrelated sketches that happen to line up."""
    return sum(f not in (AUTO, SKIP) for f in naming.faces)


def name_views(sheet: Sheet, image_bgr: np.ndarray, projection: str = "first",
               reader: Reader | None = None, scale_tol: float = SCALE) -> Naming:
    """The face each view of the part drawing shows (spec 3.2).

    A projection symbol overrides `projection`, and is not the part while any other drawing exists (a one-view
    plate beside a title-block symbol is the plate). Without a symbol, labels that all follow the other projection's
    layout switch to it. A read label names its view when the size it implies fits the sheet's shared scale; otherwise the
    layout does, under the same check. A view neither names stays "auto", never a guess; an unnamed view under
    MARK_AREA of the largest is a mark (dimension text, a balloon, a note) and is "skip". `scale_tol` is the scale
    check's tolerance (hand sketches are not drawn to scale)."""
    if projection not in ("first", "third"):
        raise ValueError(f"projection {projection!r}")
    if not sheet.drawings:
        return Naming(-1, [], projection, "setting", ["No drawing found in the image."])
    image = _bgr(image_bgr)
    ink = ink_mask(image)
    long = max(sheet.shape)
    source, warnings = "setting", []
    symbol = find_symbol(sheet, image)
    pool = list(range(len(sheet.drawings)))
    if symbol is not None:
        projection, source = symbol[1], "symbol"
        pool = [i for i in pool if i != symbol[0]] or pool
    part = max(pool, key=lambda i: (len(sheet.drawings[i].views), sum(_area(v.box) for v in sheet.drawings[i].views)))
    views = sheet.drawings[part].views
    if len(pool) > 1:
        warnings.append(f"The sheet holds {len(pool)} drawings; the views were read from the one with "
                        f"{len(views)} views.")

    texts = [_read_label(image, v.label_box, reader) for v in views]
    boxes = [v.box if v.body is not None else view_body(ink, _body(ink, v.box, long), long)[0] for v in views]
    faces, aligned, checked, notes, against = _name(boxes, texts, projection, _slack(long), scale_tol)
    if source == "setting" and against:
        other = "third" if projection == "first" else "first"
        alt = _name(boxes, texts, other, _slack(long), scale_tol)
        if set(alt[4]) < set(against):  # the labels follow the other projection, and switching breaks none
            warnings.append(f"The labels follow {other}-angle projection; the setting says {projection}-angle. "
                            "Used the labels.")
            (faces, aligned, checked, notes, against), projection, source = alt, other, "labels"
    warnings += notes
    largest = max(_area(v.box) for v in views)
    marks = [i for i, f in enumerate(faces)
             if f == AUTO and _area(views[i].box) < MARK_AREA * largest and not (aligned[i] and checked[i])]
    for i in marks:
        faces[i] = SKIP
    unnamed = faces.count(AUTO)
    if unnamed:
        warnings.append(f"{unnamed} view{'s' if unnamed > 1 else ''} could not be named; pick the face by hand.")
    if marks:
        warnings.append(f"Skipped {len(marks)} small mark{'s' if len(marks) > 1 else ''} "
                        "(dimension text, balloons or notes).")
    return Naming(part, faces, projection, source, warnings)


def crop_views(sheet: Sheet, image_bgr: np.ndarray, naming: Naming) -> list[tuple[bytes, str]]:
    """PNG crop and face of each view of the part drawing, in view order, marks left out. Each crop is the view box
    plus a CROP_MARGIN margin, on white, at the sheet's scale so the views keep their shared scale (spec 3.2)."""
    if naming.drawing < 0:
        return []
    image = _bgr(image_bgr)
    ink = ink_mask(image)
    long = max(image.shape[:2])
    out = []
    for view, face in zip(sheet.drawings[naming.drawing].views, naming.faces, strict=True):
        if face == SKIP:
            continue
        (x, y, w, h), mask = body_of(ink, view, long)  # the geometry only: no dimension line reaches a crop
        keep = cv2.dilate(mask.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
        m = max(2, round(CROP_MARGIN * max(w, h)))
        crop = np.full((h + 2 * m, w + 2 * m, 3), 255, np.uint8)
        body = image[y: y + h, x: x + w].copy()
        body[(ink[y: y + h, x: x + w] == 0) | ~keep] = 255
        crop[m: m + h, m: m + w] = body
        out.append((cv2.imencode(".png", crop)[1].tobytes(), face))
    return out


def find_symbol(sheet: Sheet, image_bgr: np.ndarray) -> tuple[int, str] | None:
    """The drawing that is the ISO 5456-2 projection symbol, and the projection it sets (spec 2, 3.2).

    Exactly 2 views: a truncated cone seen side-on (a 4-vertex trapezoid, parallel sides upright) and end-on (a
    circle with one concentric inner circle), on one centre line, the tall and short sides matching the outer and
    inner diameters within SCALE. Centre lines drawn through the symbol are allowed. Circles beside the large end
    mean first-angle, beside the small end third-angle."""
    ink = ink_mask(_bgr(image_bgr))
    long = max(sheet.shape)
    slack, k, thin = _slack(long), max(3, round(DILATE * long)), max(5, round(SPUR * long))
    for d, drawing in enumerate(sheet.drawings):
        if len(drawing.views) != 2:
            continue
        for a, b in (drawing.views, drawing.views[::-1]):
            cone, end = _trapezoid(ink, a.box, k, thin), _rings(ink, b.box, _body(ink, b.box, long), slack)
            if cone is None or end is None:
                continue
            tall, short, tall_x, short_x, axis_y = cone
            outer, inner, cx, cy = end
            if (_close(tall, outer, slack) and _close(short, inner, slack)
                    and abs(axis_y - cy) <= max(SCALE * outer, slack)):
                big_end_right, circles_right = tall_x > short_x, cx > (tall_x + short_x) / 2
                return d, "first" if big_end_right == circles_right else "third"
    return None


_KEYWORDS = {"front": "front", "rear": "back", "back": "back", "top": "top", "plan": "top", "bottom": "bottom",
             "left": "left", "right": "right", "side": "side", "end": "side", "elevation": "elevation"}


def label_face(text: str) -> str | None:
    """The face a view label names, or None. "Side" or "end" alone ("End Elevation") names a side view without
    saying which, so it is None here and the layout picks left or right."""
    hint = _label_hint(text)
    return None if hint == "side" else hint


def _label_hint(text: str) -> str | None:
    """A face, "side" (a side view, left or right unsaid) or None. Keywords match case-insensitively with one wrong
    character; two different faces in one label name nothing; "elevation" alone is the front."""
    words = re.findall(r"[a-z0-9]+", (text or "").lower().replace("view", " "))
    said = {_KEYWORDS[k] for w in words for k in _KEYWORDS if _is_word(w, k)}
    faces = said - {"side", "elevation"}
    if faces:
        return faces.pop() if len(faces) == 1 else None
    if "side" in said:
        return "side"
    return "front" if "elevation" in said else None


def _is_word(word: str, key: str) -> bool:
    """Equal, or one character wrong: substituted, missing or extra. A three-letter key forgives only a non-letter
    (a 0 read for an O), so "and" is not "end" and "tip" is not "top"."""
    if word == key:
        return True
    if len(key) < 4:
        return len(word) == len(key) and sum(a != b for a, b in zip(word, key)) == 1 and not word.isalpha()
    if len(word) == len(key):
        return sum(a != b for a, b in zip(word, key)) == 1
    if abs(len(word) - len(key)) != 1:
        return False
    short, longer = sorted((word, key), key=len)
    return any(longer[:i] + longer[i + 1:] == short for i in range(len(longer)))


def _read_label(image: np.ndarray, box: Box | None, reader: Reader | None) -> str:
    if reader is None or box is None:
        return ""
    x, y, w, h = box
    m = max(2, h // 4)
    crop = image[max(0, y - m): y + h + m, max(0, x - m): x + w + m]
    try:
        if hasattr(reader, "read"):  # s2c.reading.Reader: a batch of Crops -> ReaderResults, or None on failure
            from s2c.reading.base import Crop
            out = reader.read([Crop(crop, (x, y, w, h))])
            text = out[0].text if out else ""
        else:
            text, _ = reader(crop)
    except Exception:  # an OCR failure must not stall the sheet: the layout still names the view
        log.warning("label reader failed", exc_info=True)
        return ""
    return str(text or "").strip()


def _name(boxes: list[Box], texts: list[str], projection: str, slack: int, tol: float = SCALE):
    """Faces, whether each view lines up with the front, whether some name for it passed the scale check, the
    warnings, and the views whose label won over their place in the layout.

    Every view is tried as the front. The naming that names the most view area under the scale check wins, then the
    one agreeing with the most labels, then the layout's front (most aligned neighbours, spec 3.2): one wrong
    "FRONT VIEW" label cannot flip a naming the layout completes."""
    hints = [_label_hint(t) for t in texts]
    usual = _layout_front(boxes, slack, tol)
    best = None
    for f in range(len(boxes)):
        faces, aligned, checked, notes, agreed, against = _assign(boxes, hints, texts, f, projection, slack, tol)
        score = (sum(_area(boxes[i]) for i, x in enumerate(faces) if x != AUTO), agreed, f == usual)
        if best is None or score > best[0]:
            best = (score, faces, aligned, checked, notes, against, f)
    *result, f = best[1:]
    if f != usual and hints[f] == "front":
        placed = _layout(boxes, usual, projection, slack, tol)[0][f]
        if placed:
            result[3] = result[3] + [f"{texts[f]} is where {placed} belongs; used the label"]
    return tuple(result)


def _assign(boxes, hints, texts, f, projection, slack, tol=SCALE):
    layout, aligned = _layout(boxes, f, projection, slack, tol)
    width, height = boxes[f][2], boxes[f][3]
    depth = _depth(boxes, layout, hints, width, height, slack, tol)
    placed = [bool(n) and i != f and _fits(n, boxes[i], width, height, depth, slack, tol)
              for i, n in enumerate(layout)]
    faces, notes, agreed, against = [], {}, 0, []
    for i, box in enumerate(boxes):
        hint = hints[i]
        said = None if hint in (None, "side") else hint
        if i == f:
            face = "front"
        elif said not in (None, "front") and _fits(said, box, width, height, depth, slack, tol):
            face = said
        else:
            face = layout[i] if placed[i] else AUTO
        faces.append(face)
        if said == face or (hint == "side" and face in ("left", "right")):
            agreed += 1
            if said and i != f and layout[i] not in (None, face):
                notes[i] = f"{texts[i]} is where {layout[i]} belongs; used the label"
                against.append(i)
        elif said:
            if i == f:
                why = "is on the view read as the front"
            elif said == "front":
                why = "names the front, but another view is the front"
            else:
                why = "does not match its size"
            notes[i] = f"{texts[i]} {why}; " + (f"used as {face}" if face != AUTO else "left for you to name")
    checked = [x != AUTO for x in faces]
    for face in sorted({x for x in faces if x != AUTO}):
        same = [i for i, x in enumerate(faces) if x == face]
        if len(same) < 2:
            continue
        keep = [i for i in same if layout[i] == face]
        for i in same:
            if keep == [i]:
                continue
            what = texts[i] if hints[i] == face else f"The view placed as {face}"
            agreed -= hints[i] == face
            faces[i] = layout[i] if placed[i] and layout[i] not in faces else AUTO
            notes[i] = (f"{what} is also another view's name; "
                        + (f"used as {faces[i]}" if faces[i] != AUTO else "left for you to name"))
    against = [i for i in against if faces[i] == hints[i]]
    return faces, aligned, checked, [notes[i] for i in sorted(notes)], agreed, against


def _layout_front(boxes: list[Box], slack: int, tol: float = SCALE) -> int:
    """With two views the left (or upper) one; otherwise the view with the most neighbours in line with it and
    sharing its extent, then the largest, then the top-left."""
    if len(boxes) == 2:
        return min(range(2), key=lambda i: sum(_centre(boxes[i])))

    def neighbours(i):
        return sum(_in_line_with(boxes[i], b, row, slack) and _close(boxes[i][3 if row else 2], b[3 if row else 2],
                                                                     slack, tol)
                   for j, b in enumerate(boxes) if j != i for row in (True, False))

    return min(range(len(boxes)), key=lambda i: (-neighbours(i), -_area(boxes[i]), boxes[i][0] + boxes[i][1]))


_STEPS = {  # steps from the front along its row, and down its column -> face (spec 2)
    "first": ({1: "left", -1: "right", 2: "back", -2: "back"}, {1: "top", -1: "bottom"}),
    "third": ({1: "right", -1: "left", 2: "back", -2: "back"}, {-1: "top", 1: "bottom"}),
}


def _layout(boxes: list[Box], f: int, projection: str, slack: int,
            tol: float = SCALE) -> tuple[list[str | None], list[bool]]:
    """The face each view's place around the front f implies (spec 2), or None; and whether it is in line with f.
    Only views sharing the front's height (row) or width (column) take a place, so a dimension number between two
    views does not push the side view one step out."""
    row, col, aligned = [f], [f], [False] * len(boxes)
    for i, b in enumerate(boxes):
        if i == f:
            continue
        in_row, in_col = _in_line_with(boxes[f], b, True, slack), _in_line_with(boxes[f], b, False, slack)
        aligned[i] = in_row or in_col
        if in_row and _close(b[3], boxes[f][3], slack, tol):
            row.append(i)
        elif in_col and _close(b[2], boxes[f][2], slack, tol):
            col.append(i)
    names: list[str | None] = [None] * len(boxes)
    for line, axis, steps in ((row, 0, _STEPS[projection][0]), (col, 1, _STEPS[projection][1])):
        line.sort(key=lambda i: _centre(boxes[i])[axis])
        at = line.index(f)
        for k, i in enumerate(line):
            names[i] = steps.get(k - at)
    names[f] = "front"
    return names, aligned


def _depth(boxes, layout, hints, width, height, slack, tol=SCALE) -> float | None:
    """The part's depth in px, from the views that show it (top and bottom as height, sides as width): the value
    most of them agree on. Layout names first; labels only when no placed view shows it."""
    def values(names):
        return [boxes[i][3] if n in ("top", "bottom") else boxes[i][2] for i, n in enumerate(names)
                if n in ("top", "bottom", "left", "right") and _fits(n, boxes[i], width, height, None, slack, tol)]

    found = values(layout) or values(hints)
    if not found:
        return None
    return max(found, key=lambda v: sum(_close(v, u, slack, tol) for u in found))


def _fits(face: str, box: Box, width: int, height: int, depth: float | None, slack: int, tol: float = SCALE) -> bool:
    """The scale check: front and rear are W x H, top and bottom W x D, the sides D x H (spec 2)."""
    w, h = box[2], box[3]
    if face in ("front", "back"):
        return _close(w, width, slack, tol) and _close(h, height, slack, tol)
    if face in ("top", "bottom"):
        return _close(w, width, slack, tol) and (depth is None or _close(h, depth, slack, tol))
    return _close(h, height, slack, tol) and (depth is None or _close(w, depth, slack, tol))


def _close(a: float, b: float, slack: int, tol: float = SCALE) -> bool:
    return abs(a - b) <= max(tol * max(a, b), slack)


def _in_line_with(front: Box, b: Box, row: bool, slack: int) -> bool:
    """Same row (centres within ALIGN of the front's height) or same column (of its width)."""
    p, s = (1, 3) if row else (0, 2)
    return abs((front[p] + front[s] / 2) - (b[p] + b[s] / 2)) <= max(ALIGN * front[s], slack)


def _centre(box: Box) -> tuple[float, float]:
    return box[0] + box[2] / 2, box[1] + box[3] / 2


def _area(box: Box) -> int:
    return box[2] * box[3]


def _slack(long: int) -> int:
    return max(3, round(SLACK * long))


def _bgr(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR) if image.shape[2] == 4 else image


def _filled(ink: np.ndarray, box: Box, k: int) -> np.ndarray:
    """The box's ink closed over gaps of k px (dashes, chain lines) and filled inside its outer outlines, 0/1. The
    kernel is odd: an even one shifts the fill by a pixel."""
    x, y, w, h = box
    sub = cv2.copyMakeBorder(ink[y: y + h, x: x + w], k, k, k, k, cv2.BORDER_CONSTANT, value=0)
    closed = cv2.morphologyEx(sub, cv2.MORPH_CLOSE, np.ones((k | 1, k | 1), np.uint8))
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(closed)
    cv2.drawContours(filled, contours, -1, 1, cv2.FILLED)
    return filled[k: k + h, k: k + w]


def body_of(ink: np.ndarray, view: View, long: int) -> tuple[Box, np.ndarray]:
    """The view's geometry: the closed outline the sketch split found, or `view_body` on a drawing."""
    return (view.box, view.body) if view.body is not None else view_body(ink, view.box, long)


def view_body(ink: np.ndarray, box: Box, long: int) -> tuple[Box, np.ndarray]:
    """The view's geometry without its annotations (sheet-reading spec 2.1): the drawn region filled, lines about a
    line width thick opened away (dimension and extension lines, leaders, text, centre-line tails), and the filled
    region kept where it joins what survived. Returns the body's box on the sheet and its mask in that box. An
    outline with a small break (an edge-detected drawing) is closed first when the strict body loses most of the
    view; a view with no closed region at all keeps its box and its ink."""
    x, y, w, h = box
    sub = ink[y: y + h, x: x + w] > 0
    strict = _solid(sub, sub)
    if strict is None:
        return box, sub
    banded = _solid(sub & ~(_thin_lines(sub) > 0), sub)
    if banded is not None and _band_off(banded[0], strict[0]):
        strict = banded  # the thin-weight dimension lines had closed a band onto the view
    closed = cv2.morphologyEx(sub.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8)) > 0
    mended = _solid(closed, sub)
    (bx, by, bw, bh), body = strict
    if mended is not None:
        mx, my, mw, mh = mended[0]
        rim = max(3, 0.02 * max(mw, mh))
        around = min(bx - mx, by - my, mx + mw - bx - bw, my + mh - by - bh)  # a broken outer outline rings it;
        if around >= rim and bw * bh < 0.7 * mw * mh:                       # dimensions sit on one or two sides
            (bx, by, bw, bh), body = mended
    return (x + bx, y + by, bw, bh), body


def _band_off(inner: Box, outer: Box, tol: int = 3) -> bool:
    """`inner` is `outer` with a dimension band taken off one or two sides: it lies inside, keeps most of the area,
    and at least two of its sides stay where `outer`'s are. An outline that only looked thin (uneven lines in a
    scan) loses far more, and the view keeps its full body."""
    ix, iy, iw, ih = inner
    ox, oy, ow, oh = outer
    if (ix, iy, iw, ih) == (ox, oy, ow, oh) or iw * ih < 0.6 * ow * oh:
        return False
    inside = ix >= ox - tol and iy >= oy - tol and ix + iw <= ox + ow + tol and iy + ih <= oy + oh + tol
    same = [abs(ix - ox) <= tol, abs(iy - oy) <= tol, abs(ix + iw - ox - ow) <= tol, abs(iy + ih - oy - oh) <= tol]
    return inside and sum(same) >= 2


def _thin_lines(sub: np.ndarray) -> np.ndarray:
    """Straight runs drawn in the thin line weight: dimension and extension lines (ISO 128 draws them thin and the
    visible outline thick). Erased before the view is filled, so a dimension line whose extension lines touch the
    outline never closes a loop that would be filled into the view. A drawing in one line weight erases nothing."""
    h, w = sub.shape
    u8 = sub.astype(np.uint8)
    runs = []
    for axis, size in (("h", w), ("v", h)):
        length = max(15, round(0.15 * min(h, w)))
        kernel = (1, length) if axis == "h" else (length, 1)
        m, labels, stats, _ = cv2.connectedComponentsWithStats(
            cv2.morphologyEx(u8, cv2.MORPH_OPEN, np.ones(kernel, np.uint8)), connectivity=8)
        for i in range(1, m):
            rw, rh = int(stats[i, 2]), int(stats[i, 3])
            along, thick = (rw, rh) if axis == "h" else (rh, rw)
            runs.append((labels, i, along, thick, size))
    long_runs = [(t, along) for _, _, along, t, size in runs if along >= 0.5 * size]
    if not long_runs:
        return np.zeros_like(u8)
    # the outline's weight: what most of the long run length is drawn in (two lines merged side by side make a
    # run look thick, so never the thickest run)
    order = sorted(long_runs)
    total, acc, outline = sum(a for _, a in order), 0, order[-1][0]
    for t, a in order:
        acc += a
        if acc >= 0.75 * total:
            outline = t
            break
    out = np.zeros_like(u8)
    if outline < 3:  # a 1 or 2 px drawing has one weight: 1 px against 2 px is how a raster draws a line, not a type
        return out
    for labels, i, _, thick, _ in runs:
        if thick <= 0.5 * outline:
            out[labels == i] = 1
    return out


def _solid(drawn: np.ndarray, sub: np.ndarray) -> tuple[Box, np.ndarray] | None:
    """The filled region of `drawn` (dimension lines erased) with thin lines opened away, plus the thin parts joined
    to it that enclose area (a flange drawn a line width thick); its box and mask in the view's frame. A line stub
    (an extension line, a centre-line tail) encloses nothing and is left out. None when nothing closed survives."""
    dist = cv2.distanceTransform(sub.astype(np.uint8), cv2.DIST_L2, 3)
    stroke = 2 * max(1.0, float(np.percentile(dist[sub], 90))) if sub.any() else 2.0
    filled = ndimage.binary_fill_holes(drawn)
    k = max(5, int(2 * stroke) + 1) | 1
    core = cv2.morphologyEx(filled.astype(np.uint8), cv2.MORPH_OPEN, np.ones((k, k), np.uint8))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(core, connectivity=8)
    if n < 2:
        return None
    big = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    bx, by, bw, bh = (int(v) for v in stats[big, :4])
    keep = [i for i in range(1, n) if i == big or (
        stats[i, cv2.CC_STAT_AREA] >= 0.02 * stats[big, cv2.CC_STAT_AREA]
        and stats[i, 0] < bx + bw and stats[i, 0] + stats[i, 2] > bx
        and stats[i, 1] < by + bh and stats[i, 1] + stats[i, 3] > by)]
    core = np.isin(labels, keep)
    # only drawn pieces joined to the view: an extension line starts a small gap away from it, so its stub is not
    _, pieces = cv2.connectedComponents(filled.astype(np.uint8), connectivity=8)
    joined = np.isin(pieces, np.unique(pieces[core & (pieces > 0)]))
    near = cv2.dilate(core.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    body = filled & joined & near
    rest = filled & joined & ~near  # thin parts: kept when they hold area that is not ink (two lines and a gap)
    m, parts = cv2.connectedComponents(rest.astype(np.uint8), connectivity=8)  # or are thicker than a line
    inside = filled & ~(drawn > 0)
    band = cv2.distanceTransform(rest.astype(np.uint8), cv2.DIST_L2, 3)
    for i in range(1, m):
        part = parts == i
        if np.count_nonzero(part & inside) >= 2 or 2 * float(band[part].max()) >= 1.5 * stroke:
            body |= part
    ys, xs = np.nonzero(body)
    x0, x1, y0, y1 = int(xs.min()), int(xs.max()), int(ys.min()), int(ys.max())
    return (x0, y0, x1 - x0 + 1, y1 - y0 + 1), body[y0: y1 + 1, x0: x1 + 1]


def _body(ink: np.ndarray, box: Box, long: int) -> Box:
    """The view's box without the chain lines and leaders that run past its outline (spec 2: never part of it):
    rows and columns of the filled view thinner than SPUR of the long side are trimmed from its ends. A view that
    thin all through (a sheet-metal edge) keeps its box."""
    filled = _filled(ink, box, max(3, round(DILATE * long)))
    thin = max(5, round(SPUR * long))
    cols, rows = np.nonzero(filled.sum(0) > thin)[0], np.nonzero(filled.sum(1) > thin)[0]
    x0, x1 = (int(cols[0]), int(cols[-1])) if len(cols) else (0, box[2] - 1)
    y0, y1 = (int(rows[0]), int(rows[-1])) if len(rows) else (0, box[3] - 1)
    return box[0] + x0, box[1] + y0, x1 - x0 + 1, y1 - y0 + 1


def _trapezoid(ink: np.ndarray, box: Box, k: int, thin: int) -> tuple[float, float, float, float, float] | None:
    """A cone seen side-on: a convex 4-vertex outline with two upright parallel sides, once lines thinner than
    `thin` (its axis, drawn past both ends) are opened away. Returns the tall and short side lengths (to the outside
    of the stroke), their x (image px) and the axis height."""
    filled = cv2.morphologyEx(_filled(ink, box, k), cv2.MORPH_OPEN, np.ones((thin, thin), np.uint8))
    contours, _ = cv2.findContours(filled, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None
    c = max(contours, key=cv2.contourArea)
    poly = cv2.approxPolyDP(c, 0.03 * cv2.arcLength(c, True), True).reshape(-1, 2)
    if len(poly) != 4 or not cv2.isContourConvex(poly):
        return None
    ends = [(p, q) for p, q in zip(poly, np.roll(poly, -1, axis=0)) if abs(p[0] - q[0]) <= 0.1 * abs(p[1] - q[1])]
    if len(ends) != 2:
        return None
    (tall, tall_x, axis_y), (short, short_x, _) = sorted(
        ((abs(float(p[1] - q[1])) + 1, (p[0] + q[0]) / 2 + box[0], (p[1] + q[1]) / 2 + box[1]) for p, q in ends),
        reverse=True)
    return tall, short, tall_x, short_x, axis_y


def _rings(ink: np.ndarray, box: Box, body: Box, slack: int) -> tuple[float, float, float, float] | None:
    """A cone seen end-on: two concentric circles, and nothing else but centre lines through their centre.

    A circle inks nearly every direction around the centre at its radius; a centre line inks only a few, so it
    neither makes nor breaks a ring. The centre is that of `body`, the view without its centre-line spurs. Returns
    the outer and inner diameters (to the outside of each stroke) and the centre (image px)."""
    cx, cy = _centre(body)
    x, y, w, h = box
    ys, xs = np.nonzero(ink[y: y + h, x: x + w])
    big = max(body[2], body[3]) / 2
    if big < 8 or len(xs) == 0:
        return None
    dx, dy = xs + x + 0.5 - cx, ys + y + 0.5 - cy
    r = np.hypot(dx, dy)
    direction = ((np.arctan2(dy, dx) + math.pi) * RING_DIRECTIONS / (2 * math.pi)).astype(int) % RING_DIRECTIONS
    hit = np.zeros((int(r.max()) + 3, RING_DIRECTIONS), bool)
    hit[r.astype(int), direction] = True
    near = hit.copy()
    near[1:] |= hit[:-1]
    near[:-1] |= hit[1:]
    rings = [(a, b) for a, b in _runs(near.mean(1) >= RING_COVER) if b >= max(4, 0.15 * big)]
    if len(rings) != 2:
        return None
    near_ring = [(r >= a - 1) & (r < b + 2) for a, b in rings]  # the pixels behind each run of the 3-px window
    on_axis = (np.abs(dx) <= slack) | (np.abs(dy) <= slack)
    if (~(near_ring[0] | near_ring[1]) & ~on_axis).sum() > 0.05 * len(r):
        return None
    inner = r[near_ring[0] & ~on_axis]
    if len(inner) == 0:
        return None
    return float(max(body[2], body[3])), 2 * float(inner.max()) + 1, cx, cy


ROUND = 0.85  # a view whose filled outline covers this share of its enclosing circle is round


def round_view(ink: np.ndarray, box) -> bool:
    """The view's filled outline is a circle, once lines thinner than a tenth of the view (chain lines through a
    hole) are opened away. Round views alone never make a sheet: they are what a tight crop of one drawing leaves
    when the split takes its outline for the sheet frame, its holes."""
    x, y, w, h = box
    k = max(3, round(0.1 * min(w, h))) | 1
    sub = cv2.copyMakeBorder(ink[y: y + h, x: x + w], k, k, k, k, cv2.BORDER_CONSTANT, value=0)
    closed = cv2.morphologyEx(sub, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(closed)
    cv2.drawContours(filled, contours, -1, 255, cv2.FILLED)
    opened = cv2.morphologyEx(filled, cv2.MORPH_OPEN, np.ones((k, k), np.uint8))
    contours, _ = cv2.findContours(opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return False
    c = max(contours, key=cv2.contourArea)
    _, r = cv2.minEnclosingCircle(c)
    return cv2.contourArea(c) >= ROUND * math.pi * r * r


def sheet_crops(image_bgr: np.ndarray, projection: str = "first", reader: Reader | None = None,
                ) -> tuple[Sheet, Naming, list[tuple[int, bytes, str]]] | None:
    """(sheet, naming, [(view index, PNG crop, face)]) when the image is a drawing sheet, else None. A sheet needs
    two aligned line-drawn views (is_sheet), two of them named under the scale check (two unrelated sketches can
    line up) and one named view that is not round (a tight crop of one drawing leaves its holes as "views")."""
    sheet = split_sheet(image_bgr)
    if not is_sheet(sheet):
        return None
    naming = name_views(sheet, image_bgr, projection, reader=reader)
    if named_count(naming) < 2:
        return None
    ink = ink_mask(image_bgr)
    views = sheet.drawings[naming.drawing].views
    if all(round_view(ink, v.box) for v, f in zip(views, naming.faces) if f not in ("auto", SKIP)):
        return None
    index = [i for i, f in enumerate(naming.faces) if f != SKIP]
    crops = [(i, png, face) for (png, face), i in zip(crop_views(sheet, image_bgr, naming), index, strict=True)]
    return sheet, naming, crops
