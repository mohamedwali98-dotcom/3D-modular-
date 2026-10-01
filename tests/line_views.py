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
