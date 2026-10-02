"""Exact orthographic line drawings of a CadQuery part for the complex-parts tests: OCC hidden-line removal
projects each face's view, visible edges are drawn solid and hidden ones dashed, in our face frames (a right,
b up, the part's minimum corner at the origin)."""
from __future__ import annotations

import cadquery as cq
import cv2
import numpy as np
from OCP.BRepAdaptor import BRepAdaptor_Curve
from OCP.GCPnts import GCPnts_QuasiUniformDeflection
from OCP.gp import gp_Ax2, gp_Dir, gp_Pnt
from OCP.HLRAlgo import HLRAlgo_Projector
from OCP.HLRBRep import HLRBRep_Algo, HLRBRep_HLRToShape
from OCP.TopAbs import TopAbs_EDGE
from OCP.TopExp import TopExp_Explorer
from OCP.TopoDS import TopoDS

from s2c.multiview.benchmark import compose_sheet

# face -> (normal toward the viewer, +a direction); +b = normal x +a, as in build._face_plane
FRAMES = {
    "front": ((0, 0, 1), (1, 0, 0)), "back": ((0, 0, -1), (-1, 0, 0)),
    "top": ((0, 1, 0), (1, 0, 0)), "bottom": ((0, -1, 0), (1, 0, 0)),
    "right": ((1, 0, 0), (0, 0, -1)), "left": ((-1, 0, 0), (0, 0, 1)),
}


def _polylines(compound) -> list[np.ndarray]:
    out = []
    if compound is None or compound.IsNull():
        return out
    explorer = TopExp_Explorer(compound, TopAbs_EDGE)
    while explorer.More():
        curve = BRepAdaptor_Curve(TopoDS.Edge_s(explorer.Current()))
        pts = GCPnts_QuasiUniformDeflection(curve, 1e-3, curve.FirstParameter(), curve.LastParameter())
        if pts.IsDone() and pts.NbPoints() >= 2:
            out.append(np.array([(pts.Value(i).X(), pts.Value(i).Y()) for i in range(1, pts.NbPoints() + 1)]))
        explorer.Next()
    return out


def hlr(part: cq.Workplane, face: str) -> tuple[list[np.ndarray], list[np.ndarray], tuple[float, float]]:
    """Visible and hidden polylines of `face`'s view in face-frame mm, and the view's (width, height)."""
    shape = part.val().wrapped
    normal, x_dir = FRAMES[face]
    hl = HLRBRep_Algo()
    hl.Add(shape)
    hl.Projector(HLRAlgo_Projector(gp_Ax2(gp_Pnt(0, 0, 0), gp_Dir(*normal), gp_Dir(*x_dir))))
    hl.Update()
    hl.Hide()
    shapes = HLRBRep_HLRToShape(hl)
    visible = _polylines(shapes.VCompound()) + _polylines(shapes.OutLineVCompound()) \
        + _polylines(shapes.Rg1LineVCompound())
    hidden = _polylines(shapes.HCompound()) + _polylines(shapes.OutLineHCompound())
    bb = part.val().BoundingBox()
    corners = np.array([(x, y, z) for x in (bb.xmin, bb.xmax) for y in (bb.ymin, bb.ymax) for z in (bb.zmin, bb.zmax)])
    b_dir = np.cross(normal, x_dir)
    a0, b0 = (corners @ np.array(x_dir)).min(), (corners @ b_dir).min()
    size = (float((corners @ np.array(x_dir)).max() - a0), float((corners @ b_dir).max() - b0))
    shift = np.array([a0, b0])
    return [p - shift for p in visible], [p - shift for p in hidden], size


def _dashed(ink: np.ndarray, pts: np.ndarray, line: int, dash: int, gap: int) -> None:
    """Draw a polyline as even dashes along its length."""
    seg = np.diff(pts, axis=0)
    lengths = np.hypot(seg[:, 0], seg[:, 1])
    total = float(lengths.sum())
    if total < 1:
        return
    cum = np.concatenate([[0], np.cumsum(lengths)])
    t = 0.0
    while t < total:
        s, e = t, min(t + dash, total)
        a, b = np.interp(s, cum, pts[:, 0]), np.interp(s, cum, pts[:, 1])
        c, d = np.interp(e, cum, pts[:, 0]), np.interp(e, cum, pts[:, 1])
        cv2.line(ink, (round(a), round(b)), (round(c), round(d)), 255, line)
        t += dash + gap


def draw_view(part: cq.Workplane, face: str, px_per_mm: float, line: int = 2, hidden: bool = True) -> np.ndarray:
    """The view as an ink layer (255 on 0), cropped to the part's projected box plus the line width."""
    visible, hid, (w, h) = hlr(part, face)
    pad = line + 1
    ink = np.zeros((round(h * px_per_mm) + 2 * pad, round(w * px_per_mm) + 2 * pad), np.uint8)

    def px(p: np.ndarray) -> np.ndarray:
        return np.stack([p[:, 0] * px_per_mm + pad, (h - p[:, 1]) * px_per_mm + pad], 1)

    if hidden:
        dash = max(6, 4 * line)
        for p in hid:
            _dashed(ink, px(p), line, dash, max(4, 2 * line))
    for p in visible:
        cv2.polylines(ink, [np.round(px(p)).astype(np.int32)], False, 255, line)
    return ink


def line_sheet(part: cq.Workplane, faces=("front", "top", "right"), long_px: int = 400, line: int = 2,
               hidden: bool = True, labels: bool = False):
    """A first-angle sheet of the part's views (benchmark.compose_sheet), and the ink box of each view."""
    bb = part.val().BoundingBox()
    s = long_px / max(bb.xlen, bb.ylen, bb.zlen)
    return compose_sheet({f: draw_view(part, f, s, line, hidden) for f in faces}, labels)


# ---- dimensioned drawing sheets -------------------------------------------------------------------------------------

def iso_view(part: cq.Workplane, px_per_mm: float, line: int = 2) -> np.ndarray:
    """The part seen from its front-top-right corner (isometric), visible edges only: the picture a sheet adds."""
    normal = np.array([1.0, 1.0, 1.0]) / np.sqrt(3)
    x_dir = np.array([1.0, 0.0, -1.0]) / np.sqrt(2)
    hl = HLRBRep_Algo()
    hl.Add(part.val().wrapped)
    hl.Projector(HLRAlgo_Projector(gp_Ax2(gp_Pnt(0, 0, 0), gp_Dir(*normal), gp_Dir(*x_dir))))
    hl.Update()
    hl.Hide()
    shapes = HLRBRep_HLRToShape(hl)
    polys = _polylines(shapes.VCompound()) + _polylines(shapes.OutLineVCompound())
    pts = np.concatenate(polys)
    lo, hi = pts.min(0), pts.max(0)
    pad = line + 1
    w, h = hi - lo
    ink = np.zeros((round(h * px_per_mm) + 2 * pad, round(w * px_per_mm) + 2 * pad), np.uint8)
    for p in polys:
        q = np.stack([(p[:, 0] - lo[0]) * px_per_mm + pad, (hi[1] - p[:, 1]) * px_per_mm + pad], 1)
        cv2.polylines(ink, [np.round(q).astype(np.int32)], False, 255, line)
    return ink


def _text(ink: np.ndarray, text: str, x: int, y: int, height: int, vertical: bool,
          font: int = cv2.FONT_HERSHEY_SIMPLEX) -> tuple[int, int, int, int]:
    """Write `text` with its ink box's top-left corner at (x, y), read left to right or bottom to top."""
    scale, thick = height / 22.0, max(1, round(height / 10))
    (tw, th), base = cv2.getTextSize(text, font, scale, thick)
    tile = np.zeros((th + base + 4, tw + 4), np.uint8)
    cv2.putText(tile, text, (2, th + 2), font, scale, 255, thick, cv2.LINE_AA)
    ys, xs = np.nonzero(tile > 60)
    tile = tile[ys.min(): ys.max() + 1, xs.min(): xs.max() + 1]
    if vertical:
        tile = cv2.rotate(tile, cv2.ROTATE_90_COUNTERCLOCKWISE)
    th, tw = tile.shape
    ink[y: y + th, x: x + tw] = np.maximum(ink[y: y + th, x: x + tw], tile)
    return x, y, tw, th


def _arrow(ink: np.ndarray, tip: tuple[int, int], toward: tuple[int, int], size: int) -> None:
    d = np.array(toward, float) - np.array(tip, float)
    d /= np.linalg.norm(d)
    n = np.array([-d[1], d[0]])
    a = np.array(tip) + d * size * 2.2 + n * size * 0.7
    b = np.array(tip) + d * size * 2.2 - n * size * 0.7
    cv2.fillPoly(ink, [np.round([tip, a, b]).astype(np.int32)], 255)


def dimension(ink: np.ndarray, p0: tuple[int, int], p1: tuple[int, int], offset: int, text: str, line: int = 1,
              height: int = 22, gap: int = 4, font: int = cv2.FONT_HERSHEY_SIMPLEX) -> tuple[int, int, int, int]:
    """An ISO linear dimension between two points of a view edge: extension lines leaving a small gap from the
    edge, a dimension line `offset` px away with arrowheads, and the value over its middle. Horizontal when the
    points share a row (offset > 0: below), vertical when they share a column (offset < 0: left). Returns the
    text's ink box."""
    (x0, y0), (x1, y1) = p0, p1
    over, size = 6, max(3, height // 5)
    scale, thick = height / 22.0, max(1, round(height / 10))
    (tw, _), _ = cv2.getTextSize(text, font, scale, thick)
    s = 1 if offset > 0 else -1
    if y0 == y1:
        yd = y0 + offset
        for x in (x0, x1):
            cv2.line(ink, (x, y0 + s * gap), (x, yd + s * over), 255, line)
        cv2.line(ink, (x0, yd), (x1, yd), 255, line)
        _arrow(ink, (x0, yd), (x1, yd), size)
        _arrow(ink, (x1, yd), (x0, yd), size)
        return _text(ink, text, (x0 + x1) // 2 - tw // 2, yd - height - 6, height, False, font)
    xd = x0 + offset
    for y in (y0, y1):
        cv2.line(ink, (x0 + s * gap, y), (xd + s * over, y), 255, line)
    cv2.line(ink, (xd, y0), (xd, y1), 255, line)
    _arrow(ink, (xd, y0), (xd, y1), size)
    _arrow(ink, (xd, y1), (xd, y0), size)
    return _text(ink, text, xd - height - 6, (y0 + y1) // 2 - tw // 2, height, True, font)


GRID3 = {  # (column, row) around the front; row +1 is below it
    "first": {"front": (0, 0), "top": (0, 1), "bottom": (0, -1), "left": (1, 0), "right": (-1, 0)},
    "third": {"front": (0, 0), "top": (0, -1), "bottom": (0, 1), "right": (1, 0), "left": (-1, 0)},
}
_AB = {"front": "xy", "back": "xy", "top": "xz", "bottom": "xz", "right": "zy", "left": "zy"}


def drawing_sheet(part: cq.Workplane, faces=("front", "top", "left"), layout: str = "first", px_per_mm: float = 4.0,
                  dims: bool = True, iso: bool = False, line: int = 2, gap: int = 110, ext_gap: int = 4,
                  font: int = cv2.FONT_HERSHEY_SIMPLEX):
    """A dimensioned engineering sheet of the part: the views placed by `layout`, each with its overall width (below)
    and height (left) dimensioned in true millimetres, and optionally an isometric picture in a free corner.
    Returns the BGR sheet, the ink box of each view ("iso" for the picture), and the words written: [(box, text)]."""
    drawn = {f: draw_view(part, f, px_per_mm, line) for f in faces}
    grid = GRID3[layout]
    cols = sorted({grid[f][0] for f in faces})
    rows = sorted({grid[f][1] for f in faces})
    col_w = {c: max(drawn[f].shape[1] for f in faces if grid[f][0] == c) for c in cols}
    row_h = {r: max(drawn[f].shape[0] for f in faces if grid[f][1] == r) for r in rows}
    col_x, x = {}, gap
    for c in cols:
        col_x[c], x = x, x + col_w[c] + gap
    row_y, y = {}, gap
    for r in rows:
        row_y[r], y = y, y + row_h[r] + gap
    pic = iso_view(part, px_per_mm * 0.8, line) if iso else None
    width = x + (pic.shape[1] + gap if pic is not None else 0)
    height = max(y, pic.shape[0] + 2 * gap if pic is not None else 0)
    ink = np.zeros((height, width), np.uint8)
    boxes, words = {}, []
    bb = part.val().BoundingBox()
    size = {"x": bb.xlen, "y": bb.ylen, "z": bb.zlen}
    for f in faces:
        v = drawn[f]
        c, r = grid[f]
        vx = col_x[c] + (col_w[c] - v.shape[1]) // 2
        vy = row_y[r] + (row_h[r] - v.shape[0]) // 2
        ink[vy: vy + v.shape[0], vx: vx + v.shape[1]] |= v
        boxes[f] = (vx, vy, v.shape[1], v.shape[0])
        if dims:
            pad = line + 1
            x0, y0, x1, y1 = vx + pad, vy + pad, vx + v.shape[1] - pad - 1, vy + v.shape[0] - pad - 1
            a_axis, b_axis = _AB[f]
            for p0, p1, off, axis in (((x0, y1), (x1, y1), 40, a_axis), ((x0, y0), (x0, y1), -40, b_axis)):
                text = f"{size[axis]:g}"
                words.append((dimension(ink, p0, p1, off, text, gap=ext_gap, font=font), text))
    if pic is not None:
        ink[gap: gap + pic.shape[0], x: x + pic.shape[1]] |= pic
        boxes["iso"] = (x, gap, pic.shape[1], pic.shape[0])
    return cv2.cvtColor(255 - ink, cv2.COLOR_GRAY2BGR), boxes, words


class WordReader:
    """A stand-in for the reading service in tests: each crop reads as the true text of the word it covers."""
    name = "words"
    calibrated = True

    def __init__(self, words):
        self.words = words

    def read(self, crops):
        from s2c.reading.base import ReaderResult
        out = []
        for c in crops:
            x, y, w, h = c.box
            best, text = 0.0, ""
            for (wx, wy, ww, wh), t in self.words:
                ix = max(0, min(x + w, wx + ww) - max(x, wx))
                iy = max(0, min(y + h, wy + wh) - max(y, wy))
                share = ix * iy / max(1, ww * wh)
                if share > best:
                    best, text = share, t
            ok = best >= 0.5
            out.append(ReaderResult(text=text if ok else "", confidence=0.95 if ok else 0.1))
        return out
