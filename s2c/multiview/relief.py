"""Notches, steps and pockets read from the inner lines of line drawings (complex-parts spec 2026-10-01, section 3).

The visual hull keeps only what the outer outlines show. A drawing also shows where the faces of the part are: the
lines inside each view. Here the part is cut into a grid of cells along every drawn line, the cells start as the
hull, and regions between drawn lines are carved from the viewer's side until each view's predicted drawing agrees
with the drawn one. What was carved comes back as FacePocket features. Proportions only: every millimetre is scaled
from the typed envelope (rule 2)."""
from __future__ import annotations

import logging
from dataclasses import dataclass

import cv2
import numpy as np

from s2c.multiview import spec as S

log = logging.getLogger(__name__)

TOL = 0.015          # grid lines closer than this share of their axis are one line
COVER = 0.5          # a drawn line covers a grid edge when it runs along at least this share of it
HIDDEN_COST = 0.5    # a drawn hidden line the model has no edge for costs this share of a visible mismatch
MIN_GAIN = 0.05      # carving must remove at least this share of the hull's mismatch to be kept
MAX_GRID = 48        # more grid lines than this on an axis: too busy to read, the hull stays
MAX_STEPS = 40
SQUARE = 0.85        # share of each outline's perimeter on axis-parallel edges; rounder parts keep the hull
MAX_CARVE = 0.5      # a reading that carves away more than this share of the hull is not trusted
POCKET_PROV = "scaled"

Seg = tuple[str, float, float, float]  # ("h" | "v", pos, start, end) in face-frame mm: "h" is constant b

# face -> (global axis of a, sign of a, global axis of b, sign of b, viewing axis, side the viewer stands on)
_FRAME = {
    "front": ("x", 1, "y", 1, "z", 1), "back": ("x", -1, "y", 1, "z", -1),
    "top": ("x", 1, "z", -1, "y", 1), "bottom": ("x", 1, "z", 1, "y", -1),
    "right": ("z", -1, "y", 1, "x", 1), "left": ("z", 1, "y", 1, "x", -1),
}
_AXES = ("x", "y", "z")


@dataclass
class _View:
    face: str
    visible: list[Seg]
    hidden: list[Seg]


def _to_mm(lines, face: str, env: S.Envelope) -> list[Seg]:
    """Bbox fractions in image orientation (outline.find_*_lines) -> face-frame millimetres, b up."""
    a_len, b_len = S.face_size(face, env)
    out = []
    for axis, pos, s, e in lines:
        if axis == "h":
            out.append(("h", (1 - pos) * b_len, s * a_len, e * a_len))
        else:
            out.append(("v", pos * a_len, (1 - e) * b_len, (1 - s) * b_len))
    return out


def _global(face: str, a: float, b: float, env: S.Envelope) -> dict[str, float]:
    return S.to_global(face, a, b, env)


def _cluster(values: list[float], length: float) -> np.ndarray:
    """Sorted grid coordinates: values within TOL of the axis merged to their mean, the ends kept exact."""
    tol = TOL * length
    pts = sorted([0.0, length, *[min(max(v, 0.0), length) for v in values]])
    groups: list[list[float]] = [[pts[0]]]
    for v in pts[1:]:
        if v - groups[-1][-1] <= tol:
            groups[-1].append(v)
        else:
            groups.append([v])
    out = []
    for g in groups:
        out.append(0.0 if g[0] == 0.0 else length if g[-1] == length else float(np.mean(g)))
    return np.array(out)


def _grid(views: list[_View], outlines: dict[str, S.Outline], env: S.Envelope) -> dict[str, np.ndarray]:
    values: dict[str, list[float]] = {a: [] for a in _AXES}
    for face, ol in outlines.items():
        for a, b in ol.outer:
            for axis, v in _global(face, a, b, env).items():
                values[axis].append(v)
    for view in views:
        for axis, pos, s, e in view.visible + view.hidden:
            pts = [(pos, s), (pos, e)] if axis == "v" else [(s, pos), (e, pos)]
            for a, b in pts:
                for ax, v in _global(view.face, a, b, env).items():
                    values[ax].append(v)
    return {axis: _cluster(values[axis], env.length(axis)) for axis in _AXES}


def _inside(outline: S.Outline, pts: np.ndarray) -> np.ndarray:
    """Points (n, 2) in the outline's face frame that lie inside its outer loop and outside its inner loops."""
    def test(loop) -> np.ndarray:
        c = np.asarray(loop, np.float32).reshape(-1, 1, 2)
        return np.array([cv2.pointPolygonTest(c, (float(x), float(y)), False) >= 0 for x, y in pts])
    keep = test(outline.outer)
    for loop in outline.inner:
        keep &= ~test(loop)
    return keep


def hull_cells(grid: dict[str, np.ndarray], outlines: dict[str, S.Outline], env: S.Envelope) -> np.ndarray:
    """occ[i, j, k] over x, y, z cells: the cell centre projects inside every canonical outline."""
    cx, cy, cz = ((g[:-1] + g[1:]) / 2 for g in (grid["x"], grid["y"], grid["z"]))
    Z = env.z_mm
    f = _inside(outlines["front"], np.array([(x, y) for x in cx for y in cy])).reshape(len(cx), len(cy))
    t = _inside(outlines["top"], np.array([(x, Z - z) for x in cx for z in cz])).reshape(len(cx), len(cz))
    r = _inside(outlines["right"], np.array([(Z - z, y) for z in cz for y in cy])).reshape(len(cz), len(cy))
    return f[:, :, None] & t[:, None, :] & r.T[None, :, :]


# ---- one view's drawing, predicted and observed -----------------------------------------------------------------

def _view_axes(face: str) -> tuple[tuple[int, int, int], tuple[bool, bool, bool]]:
    """Transpose order (a, b, d) of the x, y, z cell array and the flips that make each index grow with a, b and
    toward the viewer."""
    a_axis, a_sign, b_axis, b_sign, d_axis, d_side = _FRAME[face]
    order = tuple(_AXES.index(ax) for ax in (a_axis, b_axis, d_axis))
    return order, (a_sign < 0, b_sign < 0, d_side < 0)


def _in_view(occ: np.ndarray, face: str) -> np.ndarray:
    order, flips = _view_axes(face)
    v = np.transpose(occ, order)
    for axis, flip in enumerate(flips):
        if flip:
            v = np.flip(v, axis)
    return v


def _from_view(v: np.ndarray, face: str) -> np.ndarray:
    order, flips = _view_axes(face)
    for axis, flip in enumerate(flips):
        if flip:
            v = np.flip(v, axis)
    return np.transpose(v, np.argsort(order))


def predicted(occ: np.ndarray, face: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """(visible, hidden) lines on the vertical grid edges (na + 1, nb) and on the horizontal ones (na, nb + 1)."""
    v = _in_view(occ, face)
    nd = v.shape[2]
    idx = np.arange(nd)[None, None, :]
    depth = np.where(v.any(2), np.max(np.where(v, idx, -1), 2), -1)
    pad = np.pad(depth, 1, constant_values=-1)
    vis_v = pad[:-1, 1:-1] != pad[1:, 1:-1]
    vis_h = pad[1:-1, :-1] != pad[1:-1, 1:]
    p = np.pad(v, 1, constant_values=False)

    def edges(c00, c10, c01, c11) -> np.ndarray:
        n = c00.astype(int) + c10 + c01 + c11
        diag = (c00 & c11 & ~c10 & ~c01) | (c10 & c01 & ~c00 & ~c11)
        return ((n % 2) == 1) | diag

    ev = edges(p[:-1, 1:-1, :-1], p[1:, 1:-1, :-1], p[:-1, 1:-1, 1:], p[1:, 1:-1, 1:]).any(2)
    eh = edges(p[1:-1, :-1, :-1], p[1:-1, 1:, :-1], p[1:-1, :-1, 1:], p[1:-1, 1:, 1:]).any(2)
    return vis_v, ev & ~vis_v, vis_h, eh & ~vis_h


def _covered(segs: list[Seg], axis: str, at: np.ndarray, span: np.ndarray, tol: float) -> np.ndarray:
    """(len(at), len(span) - 1): drawn lines of `axis` at each grid line `at` cover each span cell."""
    out = np.zeros((len(at), len(span) - 1), bool)
    lo, hi = span[:-1], span[1:]
    for ax, pos, s, e in segs:
        if ax != axis:
            continue
        rows = np.nonzero(np.abs(at - pos) <= tol)[0]
        if not len(rows):
            continue
        overlap = np.clip(np.minimum(hi, max(s, e)) - np.maximum(lo, min(s, e)), 0, None)
        out[rows] |= overlap >= COVER * (hi - lo)
    return out


@dataclass
class _Evidence:
    face: str
    obs_v: np.ndarray    # drawn visible on vertical grid edges (na + 1, nb)
    obs_h: np.ndarray    # (na, nb + 1)
    hid_v: np.ndarray
    hid_h: np.ndarray
    w_v: np.ndarray      # each edge's length as a share of the view
    w_h: np.ndarray


def _evidence(view: _View, grid: dict[str, np.ndarray], env: S.Envelope) -> _Evidence:
    a_axis, a_sign, b_axis, b_sign, _, _ = _FRAME[view.face]
    a_len, b_len = env.length(a_axis), env.length(b_axis)
    # grid lines in this face's frame, growing with a and b
    A = grid[a_axis] if a_sign > 0 else (a_len - grid[a_axis])[::-1]
    B = grid[b_axis] if b_sign > 0 else (b_len - grid[b_axis])[::-1]
    ta, tb = TOL * a_len, TOL * b_len
    obs_v = _covered(view.visible, "v", A, B, ta)
    obs_h = _covered(view.visible, "h", B, A, tb).T
    hid_v = _covered(view.hidden, "v", A, B, ta)
    hid_h = _covered(view.hidden, "h", B, A, tb).T
    w_v = np.broadcast_to(np.diff(B)[None, :] / b_len, obs_v.shape)
    w_h = np.broadcast_to(np.diff(A)[:, None] / a_len, obs_h.shape)
    return _Evidence(view.face, obs_v, obs_h, hid_v, hid_h, w_v, w_h)


def _cost(occ: np.ndarray, ev: _Evidence) -> float:
    vis_v, hid_v, vis_h, hid_h = predicted(occ, ev.face)
    miss_v = (vis_v != ev.obs_v) | (ev.hid_v & ~vis_v & ~hid_v & ~ev.obs_v)
    miss_h = (vis_h != ev.obs_h) | (ev.hid_h & ~vis_h & ~hid_h & ~ev.obs_h)
    cost_v = np.where(vis_v != ev.obs_v, 1.0, HIDDEN_COST) * miss_v * ev.w_v
    cost_h = np.where(vis_h != ev.obs_h, 1.0, HIDDEN_COST) * miss_h * ev.w_h
    return float(cost_v.sum() + cost_h.sum())


def _regions(ev: _Evidence) -> list[np.ndarray]:
    """Cells of the view (na, nb) split by its drawn visible lines: each region is a candidate face to carve."""
    na, nb = ev.obs_h.shape[0], ev.obs_v.shape[1]
    label = -np.ones((na, nb), int)
    out = []
    for sa in range(na):
        for sb in range(nb):
            if label[sa, sb] >= 0:
                continue
            k = len(out)
            stack, cells = [(sa, sb)], []
            label[sa, sb] = k
            while stack:
                a, b = stack.pop()
                cells.append((a, b))
                for da, db in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    a2, b2 = a + da, b + db
                    if not (0 <= a2 < na and 0 <= b2 < nb) or label[a2, b2] >= 0:
                        continue
                    wall = ev.obs_v[max(a, a2), b] if da else ev.obs_h[a, max(b, b2)]
                    if not wall:
                        label[a2, b2] = k
                        stack.append((a2, b2))
            mask = np.zeros((na, nb), bool)
            mask[tuple(np.array(cells).T)] = True
            out.append(mask)
    return out


def carve(occ: np.ndarray, evidence: list[_Evidence]) -> tuple[np.ndarray, float, float]:
    """Greedy carving from the viewer's side, region by region; returns the cells and the start and end cost."""
    def total(o: np.ndarray) -> float:
        return sum(_cost(o, ev) for ev in evidence)

    start = best_cost = total(occ)
    moves = [(ev.face, region) for ev in evidence for region in _regions(ev)]
    for _ in range(MAX_STEPS):
        best = None
        for face, region in moves:
            v = _in_view(occ, face)
            column = v[region]                        # (cells, nd)
            if not column.any():
                continue
            nd = v.shape[2]
            for keep in range(-1, nd - 1):            # keep cells at depth <= keep, empty the rest of the region
                if not column[:, keep + 1:].any():
                    continue
                cut = v.copy()
                sub = cut[region]
                sub[:, keep + 1:] = False
                cut[region] = sub
                trial = _from_view(cut, face)
                c = total(trial)
                if c < best_cost - 1e-9 and (best is None or c < best[0]):
                    best = (c, trial)
        if best is None:
            break
        best_cost, occ = best[0], best[1]
    return occ, start, best_cost


# ---- cells -> pockets ---------------------------------------------------------------------------------------------

def _boxes(removed: np.ndarray) -> list[tuple[slice, slice, slice]]:
    left = removed.copy()
    out = []
    nx, ny, nz = removed.shape
    for i, j, k in zip(*np.nonzero(removed), strict=True):
        if not left[i, j, k]:
            continue
        i2 = i + 1
        while i2 < nx and left[i2, j, k]:
            i2 += 1
        j2 = j + 1
        while j2 < ny and left[i:i2, j2, k].all():
            j2 += 1
        k2 = k + 1
        while k2 < nz and left[i:i2, j:j2, k2].all():
            k2 += 1
        left[i:i2, j:j2, k:k2] = False
        out.append((slice(i, i2), slice(j, j2), slice(k, k2)))
    return out


def _open_face(box, occ: np.ndarray) -> str | None:
    """The face this box opens to: nothing solid between it and that envelope face. The nearest such face wins."""
    sx, sy, sz = box
    best = None
    for face, (_, _, _, _, d_axis, side) in _FRAME.items():
        ax = _AXES.index(d_axis)
        sl = [sx, sy, sz]
        n = occ.shape[ax]
        lo, hi = sl[ax].start, sl[ax].stop
        sl[ax] = slice(hi, n) if side > 0 else slice(0, lo)
        if occ[tuple(sl)].any():
            continue
        gap = (n - hi) if side > 0 else lo
        if best is None or gap < best[0]:
            best = (gap, face)
    return None if best is None else best[1]


def _pocket(box, face: str, grid: dict[str, np.ndarray], env: S.Envelope) -> dict:
    a_axis, a_sign, b_axis, b_sign, d_axis, side = _FRAME[face]
    ext = {ax: (grid[ax][s.start], grid[ax][s.stop]) for ax, s in zip(_AXES, box, strict=True)}

    def frame(axis: str, sign: int) -> tuple[float, float]:
        lo, hi = ext[axis]
        return (lo, hi) if sign > 0 else (env.length(axis) - hi, env.length(axis) - lo)

    a0, a1 = frame(a_axis, a_sign)
    b0, b1 = frame(b_axis, b_sign)
    lo, hi = ext[d_axis]
    length = env.length(d_axis)
    depth = (length - lo) if side > 0 else hi
    through = depth >= length - TOL * length
    return {"type": "pocket", "face": face, "a_mm": float(a0 + a1) / 2, "b_mm": float(b0 + b1) / 2,
            "width_mm": float(a1 - a0), "height_mm": float(b1 - b0), "depth_mm": None if through else float(depth)}


def _square(outline: S.Outline) -> bool:
    """Most of the outline runs along the axes: the cells can follow it exactly. A round or tapered outline
    (a turned part, a cone) would leave staircase edges no drawn line explains."""
    pts = np.asarray(outline.outer, float)
    seg = np.roll(pts, -1, 0) - pts
    length = np.hypot(seg[:, 0], seg[:, 1])
    straight = (np.abs(seg[:, 0]) <= 0.035 * length) | (np.abs(seg[:, 1]) <= 0.035 * length)
    return float(length[straight].sum()) >= SQUARE * float(length.sum())


def relief(observations, outlines: dict[str, S.Outline], env: S.Envelope) -> tuple[list[dict], list[str]]:
    """FacePocket dicts read from the line-art views' inner lines, and Review warnings. Empty when the drawings hold
    no evidence beyond the hull, or when the reading does not explain them clearly better than the hull."""
    views = [_View(o.face, _to_mm(o.outline.lines, o.face, env), _to_mm(o.outline.hidden, o.face, env))
             for o in observations if o.line_art and o.outline.lines]
    if len(views) < 2 or not all(f in outlines for f in S.CANONICAL_FACES):
        return [], []
    if not all(_square(outlines[f]) for f in S.CANONICAL_FACES):
        return [], []
    grid = _grid(views, outlines, env)
    if max(len(g) for g in grid.values()) > MAX_GRID:
        return [], ["The drawing has too many lines to read its notches and pockets; only the outlines are used."]
    occ = hull_cells(grid, outlines, env)
    if not occ.any():
        return [], []
    evidence = [_evidence(v, grid, env) for v in views]
    carved, start, end = carve(occ, evidence)
    if start <= 0 or (start - end) < MIN_GAIN * start:
        return [], []
    if _cell_volume(occ & ~carved, grid) > MAX_CARVE * _cell_volume(occ, grid):
        return [], ["The inner lines of the drawing could not be read with confidence; only the outlines are used."]
    pockets = []
    for box in _boxes(occ & ~carved):
        face = _open_face(box, carved)
        if face is not None:
            pockets.append(_pocket(box, face, grid, env))
    if not pockets:
        return [], []
    n = len(pockets)
    note = (f"{n} notch{'es' if n > 1 else ''} or pocket{'s' if n > 1 else ''} read from the inner lines of the "
            "drawing; check them in the model.")
    return pockets, [note]


def _cell_volume(cells: np.ndarray, grid: dict[str, np.ndarray]) -> float:
    dx, dy, dz = (np.diff(grid[a]) for a in _AXES)
    return float((cells * dx[:, None, None] * dy[None, :, None] * dz[None, None, :]).sum())


def pocket_provenance(pockets: list[dict], start: int) -> dict[str, str]:
    prov = {}
    for k, p in enumerate(pockets, start=start):
        for name in ("a_mm", "b_mm", "width_mm", "height_mm", "depth_mm"):
            if p.get(name) is not None:
                prov[f"features[{k}].{name}"] = POCKET_PROV
    return prov
