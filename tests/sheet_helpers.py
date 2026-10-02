"""Synthetic drawing sheets for the sheet tests: views of a known part drawn as ISO line drawings (spec 2, 6)."""
from __future__ import annotations

import math
from typing import NamedTuple

import cv2
import numpy as np

from s2c.multiview.raster import MARGIN, Mesh, face_mask, solid_mesh
from s2c.multiview.spec import FACES, Envelope, face_size

FONT = cv2.FONT_HERSHEY_SIMPLEX
LABELS = {"front": "FRONT VIEW", "top": "TOP VIEW", "right": "RIGHT SIDE VIEW", "left": "LEFT SIDE VIEW",
          "back": "REAR VIEW", "bottom": "BOTTOM VIEW"}
# (column, row) of each view around the front; row +1 is below the front.
GRID = {
    "first": {"front": (0, 0), "top": (0, 1), "bottom": (0, -1), "left": (1, 0), "right": (-1, 0), "back": (2, 0)},
    "third": {"front": (0, 0), "top": (0, -1), "bottom": (0, 1), "right": (1, 0), "left": (-1, 0), "back": (2, 0)},
}

Box = tuple[int, int, int, int]


class Label(NamedTuple):
    """A label's putText arguments and its ink box: putText at (x + dx, y + dy) puts the ink's corner at (x, y)."""
    text: str
    scale: float
    line: int
    w: int
    h: int
    dx: int
    dy: int


def part_views(solid_or_mesh, env: Envelope, long_side: int = 240) -> dict[str, np.ndarray]:
    """Silhouette of each of the six faces from raster.face_mask, all at one scale (the longest envelope edge is
    `long_side` px) and cropped to the part, so they line up on a sheet the way orthographic views do."""
    mesh = solid_or_mesh if isinstance(solid_or_mesh, Mesh) else solid_mesh(solid_or_mesh)
    s = long_side / max(env.x_mm, env.y_mm, env.z_mm)
    views = {}
    for face in FACES:
        px = round(s * max(face_size(face, env))) + 2 * MARGIN  # face_mask then draws at s px/mm
        views[face] = _crop(face_mask(mesh, face, env, px=px)[0])
    return views


def draw_sheet(views: dict[str, np.ndarray], layout: str = "first", labels: bool = True, gap: int = 60,
               line: int = 2, symbol: str | None = None, border: bool = False, centre_lines: bool = False,
               hidden: dict[str, list] | None = None) -> tuple[np.ndarray, dict[str, tuple]]:
    """A BGR sheet of line drawings, and the ground-truth ink box (x, y, w, h) of everything drawn.

    Each mask in `views` (face -> binary silhouette, one shared scale) is drawn as its contours, `line` px black on
    white; round openings are drawn as circles. Views are placed around the front, `gap` px apart, rows and columns
    centred on each other:
    - "first" (ISO 5456-2 first-angle): top below the front, bottom above it, the view from the left ("left") on the
      right of the front, the view from the right ("right") on its left, the rear ("back") at the far right, beside
      the left view.
    - "third" (ASME Y14.3 third-angle): top above, bottom below, "right" on the right, "left" on the left, the rear
      at the far right, beside the right view.

    `labels` writes LABELS[face] with cv2.putText, centred under each view, one label height below it.
    `symbol` ("first" | "third") draws the ISO projection symbol as its own drawing, far enough right of the views
    that split_sheet's 3 x median gap rule separates it. `border` draws a sheet frame. `centre_lines` draws chain-line
    crosses on every drawn circle, and the symbol's axis and cross. `hidden` maps a face to dashed segments ("h" | "v", pos, start, end) in the format
    of outline.find_hidden_lines: fractions of the view's box from its left and top edges, as in the image; "h" is
    the line y = pos from x = start to x = end.

    Truth keys: the face for each view (its centre and hidden lines included), "label:<face>" for each label, and
    "symbol:side" (the trapezoid) and "symbol:end" (the two circles) for the symbol.
    """
    if layout not in GRID or symbol not in (None, "first", "third"):
        raise ValueError(f"layout {layout!r} / symbol {symbol!r}")
    hidden = hidden or {}
    drawn = {f: _view_layer(_crop(m), line, centre_lines, hidden.get(f, ())) for f, m in views.items()}
    longs = [max(layer.shape) for layer in drawn.values()]
    med = float(np.median(longs)) if longs else 240.0
    sym = max(40, round(0.25 * med)) if symbol else 0
    label_h = max(12, round(0.07 * med)) if labels else 0
    for _ in range(4):  # a label must stay farther from its view than the split's dilation reaches
        texts = {f: _text(LABELS[f], label_h, line) for f in drawn} if labels else {}
        spots, sym_at, size = _place(drawn, texts, GRID[layout], gap, sym, longs)
        short = max(3, round(0.008 * max(size))) + 3 - min((t.h for t in texts.values()), default=10 ** 6)
        if short <= 0:
            break
        label_h += short

    h, w = size
    ink = np.zeros((h, w), np.uint8)
    truth: dict[str, tuple] = {}
    for face, layer in drawn.items():
        x, y = spots[face]
        lh, lw = layer.shape
        ink[y: y + lh, x: x + lw] |= layer
        truth[face] = (x, y, lw, lh)
        if face in texts:
            truth[f"label:{face}"] = _put(ink, texts[face], truth[face])
    if symbol:
        truth["symbol:side"], truth["symbol:end"] = draw_symbol(ink, symbol, *sym_at, sym, line, centre_lines)
    if border:
        b = max(4, gap // 4)
        cv2.rectangle(ink, (b, b), (w - 1 - b, h - 1 - b), 255, line + 1)
    return cv2.cvtColor(255 - ink, cv2.COLOR_GRAY2BGR), truth


def relabel(img: np.ndarray, truth: dict[str, tuple], face: str, text: str, line: int = 2) -> Box:
    """Replace the label under `face` with `text` (a mislabelled view), at the same height; returns its new box."""
    x, y, w, h = truth[f"label:{face}"]
    img[y: y + h, x: x + w] = 255
    ink = np.zeros(img.shape[:2], np.uint8)
    box = _put(ink, _text(text, h, line), truth[face])
    img[ink > 0] = 0
    truth[f"label:{face}"] = box
    return box


def _crop(mask: np.ndarray) -> np.ndarray:
    on = mask > 127
    ys, xs = np.nonzero(on)
    if len(xs) == 0:
        raise ValueError("empty view mask")
    return np.where(on[ys.min(): ys.max() + 1, xs.min(): xs.max() + 1], 255, 0).astype(np.uint8)


def _circle(contour) -> tuple[int, int, int] | None:
    """Centre and radius when a contour is round: a hole, or the outline of a turned part seen end-on."""
    (cx, cy), r = cv2.minEnclosingCircle(contour)
    if r < 3 or cv2.contourArea(contour) < 0.8 * math.pi * r * r:
        return None
    return round(cx), round(cy), round(r)


def _dashed(ink: np.ndarray, p0, p1, pattern, line: int) -> None:
    """p0 -> p1 as repeating (on, off) runs in px: hidden lines are dashed, centre lines are chain lines."""
    (x0, y0), (x1, y1) = p0, p1
    length = math.hypot(x1 - x0, y1 - y0)
    if length == 0:
        return
    ux, uy = (x1 - x0) / length, (y1 - y0) / length
    t, i = 0.0, 0
    while t < length:
        on, off = pattern[i % len(pattern)]
        e = min(t + on, length)
        cv2.line(ink, (round(x0 + ux * t), round(y0 + uy * t)), (round(x0 + ux * e), round(y0 + uy * e)), 255, line)
        t, i = e + off, i + 1


def _view_layer(mask: np.ndarray, line: int, centre_lines: bool, hidden) -> np.ndarray:
    """One view drawn on its own black canvas (ink 255), cropped to its ink."""
    over = max(8, 4 * line)
    pad = over + 2 * line
    mh, mw = mask.shape
    ink = np.zeros((mh + 2 * pad, mw + 2 * pad), np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    chain = [(8 * line, 1.5 * line), (1.5 * line, 1.5 * line)]
    for c in contours:
        round_ = _circle(c)
        if round_ is None:
            cv2.drawContours(ink, [c], -1, 255, line, offset=(pad, pad))
            continue
        cx, cy, r = round_[0] + pad, round_[1] + pad, round_[2]
        cv2.circle(ink, (cx, cy), r, 255, line)
        if centre_lines:
            _dashed(ink, (cx - r - over, cy), (cx + r + over, cy), chain, line)
            _dashed(ink, (cx, cy - r - over), (cx, cy + r + over), chain, line)
    for kind, pos, start, end in hidden:
        if kind == "h":
            yy = pad + pos * (mh - 1)
            p0, p1 = (pad + start * (mw - 1), yy), (pad + end * (mw - 1), yy)
        else:
            xx = pad + pos * (mw - 1)
            p0, p1 = (xx, pad + start * (mh - 1)), (xx, pad + end * (mh - 1))
        _dashed(ink, p0, p1, [(5 * line, 2.5 * line)], line)
    return _ink_crop(ink)


def _ink_crop(ink: np.ndarray) -> np.ndarray:
    ys, xs = np.nonzero(ink)
    return ink[ys.min(): ys.max() + 1, xs.min(): xs.max() + 1]


def _text(text: str, height: int, line: int) -> Label:
    """The putText scale that makes `text` about `height` px tall, measured on a scratch canvas."""
    def measure(scale):
        (tw, th), base = cv2.getTextSize(text, FONT, scale, line)
        canvas = np.zeros((th + base + 4 * line + 4, tw + 4 * line + 4), np.uint8)
        org = (2 * line + 2, th + 2 * line + 2)
        cv2.putText(canvas, text, org, FONT, scale, 255, line, cv2.LINE_8)
        ys, xs = np.nonzero(canvas >= 128)
        return Label(text, scale, line, int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1),
                     int(org[0] - xs.min()), int(org[1] - ys.min()))

    return measure(height / measure(1.0).h)


def _put(ink: np.ndarray, label: Label, view: Box) -> Box:
    """cv2.putText the label centred under `view`, one label height below it; returns its ink box."""
    x, y = view[0] + (view[2] - label.w) // 2, view[1] + view[3] + label.h
    layer = np.zeros_like(ink)
    cv2.putText(layer, label.text, (x + label.dx, y + label.dy), FONT, label.scale, 255, label.line, cv2.LINE_8)
    layer = np.where(layer >= 128, 255, 0).astype(np.uint8)  # OpenCV 5 anti-aliases text; keep the ink crisp
    ink |= layer
    return cv2.boundingRect(layer)


def _place(drawn, texts, grid, gap: int, sym: int, longs):
    """Top-left corner of each view, of the symbol, and the sheet size (h, w)."""
    label = {f: (t.w, 2 * t.h) for f, t in texts.items()}  # width, and height with its gap above
    cols = sorted({grid[f][0] for f in drawn})
    rows = sorted({grid[f][1] for f in drawn})
    col_w = {c: max(max(drawn[f].shape[1], label.get(f, (0, 0))[0]) for f in drawn if grid[f][0] == c) for c in cols}
    row_h = {r: max(drawn[f].shape[0] for f in drawn if grid[f][1] == r) for r in rows}
    row_label = {r: max((label[f][1] for f in drawn if grid[f][1] == r and f in label), default=0) for r in rows}
    col_x, x = {}, gap
    for c in cols:
        col_x[c], x = x, x + col_w[c] + gap
    row_y, y = {}, gap
    for r in rows:
        row_y[r], y = y, y + row_h[r] + row_label[r] + gap
    spots = {}
    for f, layer in drawn.items():
        c, r = grid[f]
        spots[f] = (col_x[c] + (col_w[c] - layer.shape[1]) // 2, row_y[r] + (row_h[r] - layer.shape[0]) // 2)
    right, bottom = (x, y) if drawn else (gap, gap)
    sym_at = None
    if sym:
        sw, sh = _symbol_size(sym)
        med_all = float(np.median(list(longs) + [sym, sym]))  # the split also counts the symbol's two views
        sep = math.ceil(3.2 * med_all) + 1
        sym_at = (right - gap + sep, max(gap, bottom // 2 - sh // 2)) if drawn else (gap, gap)
        right = sym_at[0] + sw + gap
        bottom = max(bottom, sym_at[1] + sh + gap)
    return spots, sym_at, (bottom, right)


def _symbol_size(size: int) -> tuple[int, int]:
    return size + size // 2 + size, size  # cone length, space, end view; height is the large diameter


def draw_symbol(ink: np.ndarray, projection: str, x: int, y: int, size: int, line: int,
                 centre_lines: bool = False) -> tuple[Box, Box]:
    """ISO 5456-2 projection symbol: a truncated cone (narrow end left) as a trapezoid, and its end view as two
    concentric circles. First-angle puts the circles beside the cone's large end, third-angle beside its small end.
    `centre_lines` adds the cone's axis and the end view's cross as chain lines, overhanging each view by a twelfth
    of the symbol size, short enough that the two views stay apart."""
    big, small, length, space = size, size // 2, size, size // 2
    if projection == "first":
        cone_x, end_x = x, x + length + space
    else:
        end_x, cone_x = x, x + big + space
    cy = y + big // 2
    side = np.zeros_like(ink)
    pts = np.array([(cone_x, cy - small // 2), (cone_x + length, cy - big // 2), (cone_x + length, cy + big // 2),
                    (cone_x, cy + small // 2)], np.int32)
    cv2.polylines(side, [pts], True, 255, line)
    end = np.zeros_like(ink)
    cv2.circle(end, (end_x + big // 2, cy), big // 2, 255, line)
    cv2.circle(end, (end_x + big // 2, cy), small // 2, 255, line)
    if centre_lines:
        over, chain, ex = max(3, size // 12), [(8 * line, 1.5 * line), (1.5 * line, 1.5 * line)], end_x + big // 2
        _dashed(side, (cone_x - over, cy), (cone_x + length + over, cy), chain, line)
        _dashed(end, (ex - big // 2 - over, cy), (ex + big // 2 + over, cy), chain, line)
        _dashed(end, (ex, cy - big // 2 - over), (ex, cy + big // 2 + over), chain, line)
    ink |= side | end
    return cv2.boundingRect(side), cv2.boundingRect(end)
