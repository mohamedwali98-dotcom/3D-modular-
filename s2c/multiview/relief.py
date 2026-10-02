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
from scipy import ndimage

from s2c.multiview import spec as S
from s2c.multiview.fuse import chamfer_ring

log = logging.getLogger(__name__)

TOL = 0.015          # grid lines closer than this share of their axis are one line
COVER = 0.5          # a drawn line covers a grid edge when it runs along at least this share of it
HIDDEN_COST = 0.5    # a drawn hidden line the model has no edge for costs this share of a visible mismatch
MIN_GAIN = 0.05      # carving must remove at least this share of the hull's mismatch to be kept...
MIN_GAIN_ABS = 0.3   # ...and this much of it outright (in view lengths): a line width's sliver is no evidence
DASH_END = 0.05     # a dashed line may stop a dash and a gap short of the outline: its ends this close snap to it
DEFAULT_DEPTH = 0.25  # share of its axis a pocket gets when no other view shows how deep it goes (a check)
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
    circles: list[tuple[float, float, float]] = ()  # (a, b, d) mm: a drawn circle explains the grid edges it covers
    stroke: tuple[float, float] = (0.0, 0.0)          # the line width in mm along a and along b


def _stroke_px(o) -> float:
    """The drawing's line width in the outline's pixels: measured by the pipeline, else a typical share."""
    _, _, w, h = o.outline.bbox
    return float(o.stroke) if getattr(o, "stroke", 0) else 0.006 * max(w, h)


def _view(o, env: S.Envelope) -> _View:
    """A line-art observation in face-frame millimetres. A drawn edge sits half a line inside the outline's bbox:
    lines within one line width of the bbox are the outline itself and snap onto it, so a thin part's two outline
    edges never read as a step."""
    _, _, w, h = o.outline.bbox
    px = _stroke_px(o)
    snap = (px / max(w, 1), px / max(h, 1))
    a_len, b_len = S.face_size(o.face, env)
    stroke = (px * a_len / max(w - 1, 1), px * b_len / max(h - 1, 1))
    dashed = (max(snap[0], DASH_END), max(snap[1], DASH_END))
    return _View(o.face, _to_mm(o.outline.lines, o.face, env, snap),
                 _to_mm(o.outline.hidden, o.face, env, snap, ends=dashed),
                 [_circle_mm(o, c, env) for c in o.outline.circles], stroke)


def _circle_mm(o, c, env: S.Envelope) -> tuple[float, float, float]:
    """A drawn circle's centre (a, b) and diameter in the face frame, scaled from the outline's bbox."""
    a_len, b_len = S.face_size(o.face, env)
    x, y, w, h = o.outline.bbox
    sa, sb = a_len / max(w - 1, 1), b_len / max(h - 1, 1)
    return float((c.cx - x) * sa), float((y + h - 1 - c.cy) * sb), float(c.d * (sa + sb) / 2)


def _to_mm(lines, face: str, env: S.Envelope, snap: tuple[float, float] = (0.0, 0.0),
           ends: tuple[float, float] | None = None) -> list[Seg]:
    """Bbox fractions in image orientation (outline.find_*_lines) -> face-frame millimetres, b up. A line within
    `snap` (bbox share across it: a for "v", b for "h") of the bbox's edge is put on the edge; its ends within
    `ends` (along it; `snap` by default) of the edges are put on them."""
    a_len, b_len = S.face_size(face, env)
    ends = ends or snap
    out = []
    for axis, pos, s, e in lines:
        near, along = (snap[1], ends[0]) if axis == "h" else (snap[0], ends[1])
        pos = 0.0 if pos <= near else 1.0 if pos >= 1 - near else pos
        s = 0.0 if s <= along else s
        e = 1.0 if e >= 1 - along else e
        if axis == "h":
            out.append(("h", (1 - pos) * b_len, s * a_len, e * a_len))
        else:
            out.append(("v", pos * a_len, (1 - e) * b_len, (1 - s) * b_len))
    return out


def _global(face: str, a: float, b: float, env: S.Envelope) -> dict[str, float]:
    return S.to_global(face, a, b, env)


def _cluster(values: list[float], length: float, stroke: float = 0.0) -> np.ndarray:
    """Sorted grid coordinates: values within TOL of the axis (a line width at least) merged to their mean, the
    ends kept exact."""
    tol = max(TOL * length, stroke)
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
    stroke = {a: 0.0 for a in _AXES}
    for view in views:
        a_axis, _, b_axis, _, _, _ = _FRAME[view.face]
        stroke[a_axis] = max(stroke[a_axis], view.stroke[0])
        stroke[b_axis] = max(stroke[b_axis], view.stroke[1])
    return {axis: _cluster(values[axis], env.length(axis), stroke[axis]) for axis in _AXES}


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
class ReliefEvidence:
    face: str
    obs_v: np.ndarray    # drawn visible on vertical grid edges (na + 1, nb)
    obs_h: np.ndarray    # (na, nb + 1)
    hid_v: np.ndarray
    hid_h: np.ndarray
    w_v: np.ndarray      # each edge's length as a share of the view
    w_h: np.ndarray


def _evidence(view: _View, grid: dict[str, np.ndarray], env: S.Envelope) -> ReliefEvidence:
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
    w_v = np.broadcast_to(np.diff(B)[None, :] / b_len, obs_v.shape).copy()
    w_h = np.broadcast_to(np.diff(A)[:, None] / a_len, obs_h.shape).copy()
    for a, b, d in view.circles:  # a hole or a pin: the grid edges in its square are the circle's business
        in_a = (A >= a - d / 2 - ta) & (A <= a + d / 2 + ta)
        in_b = (B >= b - d / 2 - tb) & (B <= b + d / 2 + tb)
        w_v[np.ix_(in_a, in_b[:-1] & in_b[1:])] = 0
        w_h[np.ix_(in_a[:-1] & in_a[1:], in_b)] = 0
    return ReliefEvidence(view.face, obs_v, obs_h, hid_v, hid_h, w_v, w_h)


def _cost(occ: np.ndarray, ev: ReliefEvidence) -> float:
    vis_v, hid_v, vis_h, hid_h = predicted(occ, ev.face)
    miss_v = (vis_v != ev.obs_v) | (ev.hid_v & ~vis_v & ~hid_v & ~ev.obs_v)
    miss_h = (vis_h != ev.obs_h) | (ev.hid_h & ~vis_h & ~hid_h & ~ev.obs_h)
    cost_v = np.where(vis_v != ev.obs_v, 1.0, HIDDEN_COST) * miss_v * ev.w_v
    cost_h = np.where(vis_h != ev.obs_h, 1.0, HIDDEN_COST) * miss_h * ev.w_h
    return float(cost_v.sum() + cost_h.sum())


def _regions(ev: ReliefEvidence) -> list[np.ndarray]:
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


def _one_piece(occ: np.ndarray) -> bool:
    """The cells form one solid, joined face to face."""
    _, n = ndimage.label(occ)
    return n == 1


def carve(occ: np.ndarray, evidence: list[ReliefEvidence]) -> tuple[np.ndarray, float, float, np.ndarray]:
    """Greedy carving from the viewer's side, region by region. Returns the cells, the start and end cost, and the
    cells removed by a cut that only its own view asked for: how deep such a cut goes, no other view says."""
    def costs(o: np.ndarray) -> list[float]:
        return [_cost(o, ev) for ev in evidence]

    now = costs(occ)
    start = best_cost = sum(now)
    unsupported = np.zeros_like(occ)
    moves = [(k, ev.face, region) for k, ev in enumerate(evidence) for region in _regions(ev)]
    for _ in range(MAX_STEPS):
        best = None
        for k, face, region in moves:
            v = _in_view(occ, face)
            column = v[region]                        # (cells, nd)
            if not column.any():
                continue
            nd = v.shape[2]
            for keep in range(nd - 2, -2, -1):        # shallowest first: on a tie the least material goes
                if not column[:, keep + 1:].any():
                    continue
                cut = v.copy()
                sub = cut[region]
                sub[:, keep + 1:] = False
                cut[region] = sub
                trial = _from_view(cut, face)
                if not _one_piece(trial):
                    continue  # a part never falls apart: that reading of the lines is wrong
                per = costs(trial)
                c = sum(per)
                if c < best_cost - 1e-9 and (best is None or c < best[0] - 1e-9):
                    others = any(per[j] < now[j] - 1e-9 for j in range(len(per)) if j != k)
                    best = (c, trial, per, others)
        if best is None:
            break
        best_cost, trial, now, others = best
        if not others:
            unsupported |= occ & ~trial
        occ = trial
    return occ, start, best_cost, unsupported


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


def _extents(box, carved: np.ndarray, grid: dict[str, np.ndarray], env: S.Envelope) -> dict[str, tuple[float, float]]:
    """The box in millimetres, each side that opens onto empty space pushed past it, so rounding the pocket's sizes
    to 0.5 mm never leaves a skin of the part over the opening."""
    ext = {}
    for n, (ax, sl) in enumerate(zip(_AXES, box, strict=True)):
        lo, hi = float(grid[ax][sl.start]), float(grid[ax][sl.stop])
        margin = max(1.0, 0.02 * env.length(ax))
        for end in ("lo", "hi"):
            beyond = list(box)
            idx = sl.start - 1 if end == "lo" else sl.stop
            if 0 <= idx < carved.shape[n]:
                beyond[n] = slice(idx, idx + 1)
                if carved[tuple(beyond)].any():
                    continue  # a wall of the pocket: it stays where the drawing puts it
            if end == "lo":
                lo -= margin
            else:
                hi += margin
        ext[ax] = (lo, hi)
    return ext


def _pocket(box, face: str, carved: np.ndarray, grid: dict[str, np.ndarray], env: S.Envelope) -> dict:
    a_axis, a_sign, b_axis, b_sign, d_axis, side = _FRAME[face]
    ext = _extents(box, carved, grid, env)

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
    setup = _setup(observations, outlines, env)
    if isinstance(setup, str):
        return [], [setup]
    if setup is None:
        return [], []
    grid, occ, evidence, stroke = setup
    carved, start, end, unsupported = carve(occ, evidence)
    if start <= 0 or (start - end) < max(MIN_GAIN * start, MIN_GAIN_ABS):
        return [], []
    if _cell_volume(occ & ~carved, grid) > MAX_CARVE * _cell_volume(occ, grid):
        return [], ["The inner lines of the drawing could not be read with confidence; only the outlines are used."]
    pockets, guessed = [], 0
    smallest = max(1.0, 2 * stroke)
    for box in _boxes(occ & ~carved):
        face = _open_face(box, carved)
        if face is None:
            continue
        p = _pocket(box, face, carved, grid, env)
        if min(p["width_mm"], p["height_mm"], p["depth_mm"] or smallest) < smallest:
            continue  # thinner than two lines: the drawing's own line width, not a feature
        if unsupported[box].any():  # no other view says how deep: a blind default the user sets
            length = env.length(_FRAME[face][4])
            p["depth_mm"] = round(max(1.0, DEFAULT_DEPTH * length) * 2) / 2
            p["_depth_default"] = True
            guessed += 1
        pockets.append(p)
    if not pockets:
        return [], []
    n = len(pockets)
    notes = [(f"{n} notch{'es' if n > 1 else ''} or pocket{'s' if n > 1 else ''} read from the inner lines of the "
              "drawing; check them in the model.")]
    if guessed:
        notes.append(f"{guessed} pocket depth{'s are' if guessed > 1 else ' is'} not drawn (no hidden lines in "
                     "another view); set to a quarter of the part, check it.")
    return pockets, notes


def _setup(observations, outlines: dict[str, S.Outline], env: S.Envelope):
    """(grid, hull cells, evidence per view), a note when the drawing is too busy, or None when relief does not
    apply: fewer than two line-art views with inner lines, a missing canonical outline, round outlines."""
    views = [_view(o, env) for o in observations if o.line_art and o.outline.lines]
    if len(views) < 2 or not all(f in outlines for f in S.CANONICAL_FACES):
        return None
    if not all(_square(outlines[f]) for f in S.CANONICAL_FACES):
        return None
    grid = _grid(views, outlines, env)
    if max(len(g) for g in grid.values()) > MAX_GRID:
        return "The drawing has too many lines to read its notches and pockets; only the outlines are used."
    occ = hull_cells(grid, outlines, env)
    if not occ.any():
        return None
    return grid, occ, [_evidence(v, grid, env) for v in views], max(max(v.stroke) for v in views)


def mismatch(observations, outlines: dict[str, S.Outline], env: S.Envelope) -> float | None:
    """How much of the drawn lines the best carving of these views still fails to explain: lower is a more
    consistent reading. None when relief does not apply. Used to tell first- from third-angle."""
    setup = _setup(observations, outlines, env)
    if setup is None or isinstance(setup, str):
        return None
    _, occ, evidence, _ = setup
    return carve(occ, evidence)[2]


def _cell_volume(cells: np.ndarray, grid: dict[str, np.ndarray]) -> float:
    dx, dy, dz = (np.diff(grid[a]) for a in _AXES)
    return float((cells * dx[:, None, None] * dy[None, :, None] * dz[None, None, :]).sum())


def _profile(face: str, outline: S.Outline, env: S.Envelope, along: str, at: tuple[float, float], d_axis: str,
             side: int) -> float | None:
    """How far the outline of canonical `face` reaches along global `d_axis` toward `side`, over the columns whose
    global `along` coordinate lies in `at`. None when no column there holds material."""
    a_axis, a_sign, b_axis, b_sign, _, _ = _FRAME[face]
    a_len, b_len = env.length(a_axis), env.length(b_axis)
    s = 400.0 / max(a_len, b_len)
    mask = np.zeros((round(b_len * s) + 1, round(a_len * s) + 1), np.uint8)
    cv2.fillPoly(mask, [np.round([(a * s, (b_len - b) * s) for a, b in outline.outer]).astype(np.int32)], 1)
    for loop in outline.inner:
        cv2.fillPoly(mask, [np.round([(a * s, (b_len - b) * s) for a, b in loop]).astype(np.int32)], 0)
    sign_along = a_sign if along == a_axis else b_sign
    lo, hi = sorted(v if sign_along > 0 else env.length(along) - v for v in at)
    if along == a_axis:  # columns of the mask; material rows give b
        filled = np.nonzero(mask[:, max(0, round(lo * s)): round(hi * s) + 1].any(1))[0]
        coords, sign = b_len - filled / s, b_sign
    else:                # rows of the mask; material columns give a
        filled = np.nonzero(mask[max(0, round((b_len - hi) * s)): round((b_len - lo) * s) + 1, :].any(0))[0]
        coords, sign = filled / s, a_sign
    if not len(filled):
        return None
    glob = coords if sign > 0 else env.length(d_axis) - coords
    return float(glob.max() if side > 0 else glob.min())


def bosses(observations, edges: dict[int, set[int]], outlines: dict[str, S.Outline],
           env: S.Envelope) -> tuple[list[dict], list[str]]:
    """Round pins: a drawn circle read as an edge whose neighbour views show a bump of its width standing out to the
    envelope face on that side. Its square hull is cut back to the cylinder (FaceBoss)."""
    out = []
    for k, o in enumerate(observations):
        if not o.line_art or o.outline.circular:  # a round view is a turned part's end: the revolve handles it
            continue
        _, _, _, _, d_axis, side = _FRAME[o.face]
        length = env.length(d_axis)
        hubs = [i for i, c in enumerate(o.outline.circles)
                if i not in edges.get(k, ()) and c.ring and not chamfer_ring(c, o.stroke)]
        for i in sorted(edges.get(k, ())) + hubs:
            c = o.outline.circles[i]
            a, b, d = _circle_mm(o, c, env)
            if c.ring and (i in hubs or chamfer_ring(c, o.stroke)):
                d *= c.ring / c.d  # a chamfered pin's tip, or a hub around a bore: the outer circle is the boss
            centre = S.to_global(o.face, a, b, env)
            heights = []
            for g in S.CANONICAL_FACES:
                g_axes = S.FACE_AXES[g][:2]
                along = next((ax for ax in g_axes if ax != d_axis), None)
                if d_axis not in g_axes or along not in centre:
                    continue
                # beside the pin: past the drawn line around it (the bump reads a stroke wider than the circle)
                u, gap, ring = centre[along], max(1.0, 0.15 * d), max(1.0, 0.2 * d)
                def reach_at(lo: float, hi: float, g=g, along=along, d_axis=d_axis, side=side) -> float | None:
                    return _profile(g, outlines[g], env, along, (lo, hi), d_axis, side)

                r0, r1, r2 = d / 2 + gap, d / 2 + gap + ring, d / 2 + gap + 2 * ring
                inside = reach_at(u - 0.3 * d, u + 0.3 * d)
                near = [reach_at(u - r1, u - r0), reach_at(u + r0, u + r1)]
                far = [reach_at(u - r2, u - r1), reach_at(u + r1, u + r2)]
                if inside is None or None in near or None in far:
                    continue
                if max(abs(n - f) for n, f in zip(near, far, strict=True)) > TOL * length:
                    continue  # the surface beside it slopes: a taper or a fillet, not a shoulder a pin stands on
                around = max(near) if side > 0 else min(near)
                reach = inside if side > 0 else length - inside
                h = (inside - around) * side
                if reach >= length - TOL * length and h > TOL * length:
                    heights.append(h)
            if heights:
                out.append({"type": "boss", "face": o.face, "a_mm": a, "b_mm": b, "diameter_mm": d,
                            "height_mm": float(min(heights))})
    if not out:
        return [], []
    n = len(out)
    note = f"{n} round pin{'s' if n > 1 else ''} read from the circles and the bumps in the next view; check the height."
    return out, [note]


def pocket_provenance(pockets: list[dict], start: int, inferred: bool = False) -> dict[str, str]:
    """Provenance of the pockets' and bosses' numbers: scaled from the typed envelope (rule 2), "inferred" when an
    outline they were read against was drawn by a model or assumed, and "default" for a depth no view showed. The
    private marks are taken off the dicts here."""
    prov = {}
    for k, p in enumerate(pockets, start=start):
        guessed = p.pop("_depth_default", False)
        for name in ("a_mm", "b_mm", "width_mm", "height_mm", "depth_mm", "diameter_mm"):
            if p.get(name) is not None:
                prov[f"features[{k}].{name}"] = ("default" if guessed and name == "depth_mm"
                                                 else "inferred" if inferred else POCKET_PROV)
    return prov
