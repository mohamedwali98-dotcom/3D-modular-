"""Fuse per-image observations into a MultiViewSpec. Spec sections 4 and 5.
Envelope trust: typed > written > measured. Nothing else passes the gate."""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np

from s2c.multiview import spec as S
from s2c.multiview.edits import left_out
from s2c.multiview.ocr import Linked, Reading
from s2c.multiview.outline import PixelOutline, to_face_mm
from s2c.multiview.raster import iou, outline_mask

CLEARANCE_CLASSES = {  # ISO 273 clearance holes for M2, M2.5, M3, M4, M5, M6, M8, M10
    "fine": (2.2, 2.7, 3.2, 4.3, 5.3, 6.4, 8.4, 10.5),
    "medium": (2.4, 2.9, 3.4, 4.5, 5.5, 6.6, 9.0, 11.0),
    "coarse": (2.6, 3.1, 3.6, 4.8, 5.8, 7.0, 10.0, 12.0),
}
THICKNESS_MM = (1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0)
SNAPPABLE = frozenset({"scaled", "inferred", "estimated"})
DISAGREE = 0.05
SCALE_TRAP = 3.0  # sketches are not to scale, but a written size 3x off the rest is a misread until confirmed
_FEATURE_PATH = re.compile(r"features\[(\d+)\]\.(\w+)")


@dataclass
class Observation:
    face: str
    kind: str                                   # sketch | photo | drawing
    outline: PixelOutline
    values: list[Linked] = field(default_factory=list)
    mm_per_px: float | None = None              # reference-object scale, photos only
    blind: dict[int, bool] = field(default_factory=dict)             # circle index -> blind, Solaria only (rule 2)
    depth_estimates: dict[int, float] = field(default_factory=dict)  # unused; kept for Solaria's cleanup pop
    depth_ratio: dict[int, float] = field(default_factory=dict)      # circle index -> blind depth / axis length, Solaria
    depth_from_image: set[int] = field(default_factory=set)          # circles whose blind flag came from Solaria
    confidence: float = 0.9
    line_art: bool = False                      # a drawing in lines (drawing-sheet spec 3.3)
    hidden: list[tuple[str, float, float, float]] = field(default_factory=list)  # as PixelOutline.hidden
    stroke: float = 0.0                         # line width of a line drawing in px, 0 when not measured
    scale_confirmed: bool = True                # False: mm_per_px gives suggestions, not measured values

    def __post_init__(self):
        """Whoever builds the observation, a line-art outline makes it line art and brings its hidden lines."""
        self.line_art = self.line_art or self.outline.line_art
        self.hidden = self.hidden or list(self.outline.hidden)


def attach_label(obs: Observation, label) -> None:
    """No-op: the vision model's hole flags and depth guesses never reach geometry (rule 2). Only Solaria
    (depth.py apply_depth) may mark a hole blind or give it a depth."""


def _value(r: Reading) -> float:
    return r.value_mm * 2 if r.kind == "radius" else r.value_mm


def _differs(a: float, b: float) -> bool:
    return abs(a - b) / max(a, b) > DISAGREE


# ---- envelope ---------------------------------------------------------------

@dataclass
class _Candidate:
    value: float
    prov: str          # user_written | unconfirmed | measured; "unconfirmed" never leaves this module
    confidence: float
    face: str
    px: float = 0.0    # the span the value measures, in image pixels
    axis: str = ""     # envelope axis this candidate feeds (x, y or z)
    source: object = None  # the Reading a written candidate came from; a round part's "ab" reading feeds
                            # both its axes from the same Reading, so it must never count as two other sizes


def _envelope_candidates(observations: list[Observation]) -> dict[str, list[_Candidate]]:
    cands: dict[str, list[_Candidate]] = {"x": [], "y": [], "z": []}
    for o in observations:
        a_axis, b_axis, _ = S.FACE_AXES[o.face]
        _, _, w, h = o.outline.bbox
        for which, axis, px in (("a", a_axis, w - 1), ("b", b_axis, h - 1)):
            readings = [lv.reading for lv in o.values
                        if (lv.axis == which and lv.reading.kind == "linear") or lv.axis == "ab"]
            if readings:
                r = max(readings, key=_value)
                prov = "user_written" if r.confirmed else "unconfirmed"
                cands[axis].append(_Candidate(_value(r), prov, r.confidence, o.face, float(px), axis, r))
            if o.mm_per_px:  # a drawn outline is measured to its line's outside; a dimension, line middle to middle
                span = px - o.stroke if o.line_art else px
                prov = "measured" if o.scale_confirmed else "unconfirmed"
                cands[axis].append(_Candidate(span * o.mm_per_px, prov, o.confidence, o.face, float(px), axis))
    return cands


def _scale_trap(cands: dict[str, list[_Candidate]], user_values: dict) -> list[str]:
    """A written size whose mm per pixel is more than SCALE_TRAP times off the median of at least two other
    written sizes is demoted to unconfirmed. A note is dropped for an axis the user already confirmed."""
    written = [c for axis in "xyz" for c in cands[axis] if c.prov in ("user_written", "unconfirmed") and c.px > 0]
    notes = []
    for c in written:
        others, seen = [], set()
        for o in written:
            if o is c or (c.source is not None and o.source is c.source):
                continue
            key = id(o.source) if o.source is not None else id(o)
            if key in seen:
                continue
            seen.add(key)
            others.append(o.value / o.px)
        if len(others) < 2:
            continue
        median, scale = float(np.median(others)), c.value / c.px
        if c.prov == "user_written" and max(scale / median, median / scale) > SCALE_TRAP:
            c.prov = "unconfirmed"
            if f"envelope.{c.axis}_mm" not in user_values:
                notes.append(f"{c.face}: {c.value:g} mm does not fit the drawing's scale; check it")
    return notes


def fuse_envelope(observations: list[Observation], user_values: dict | None = None):
    user_values = user_values or {}
    cands = _envelope_candidates(observations)
    warnings: list[str] = _scale_trap(cands, user_values)
    values: dict[str, float] = {}
    prov: dict[str, str] = {}
    pending: dict[str, float] = {}
    for axis in "xyz":
        key, name = f"envelope.{axis}_mm", S.AXIS_NAMES[axis]
        if key in user_values:
            values[axis], prov[key] = float(user_values[key]), "user_edited"
            continue
        written = [c for c in cands[axis] if c.prov == "user_written"]
        measured = [c for c in cands[axis] if c.prov == "measured"]
        unconfirmed = [c for c in cands[axis] if c.prov == "unconfirmed"]
        if written:
            best = max(written, key=lambda c: c.confidence)
            for c in written:
                if c is not best and _differs(c.value, best.value):
                    warnings.append(f"{name}: {c.face} says {c.value:g} mm, {best.face} says {best.value:g} mm; "
                                    f"using {best.value:g}")
            for c in measured:
                if _differs(c.value, best.value):
                    warnings.append(f"{name}: written {best.value:g} mm, measured {c.value:.1f} mm; "
                                    "using the written value")
            values[axis], prov[key] = best.value, "user_written"
        elif measured:
            best = max(measured, key=lambda c: c.confidence)
            values[axis], prov[key] = round(best.value, 2), "measured"
        elif unconfirmed:
            pending[axis] = max(unconfirmed, key=lambda c: c.confidence).value
    missing = [a for a in "xyz" if a not in values]
    if missing:
        first = missing[0]
        name = S.AXIS_NAMES[first]
        remedy = (f"Check the {name}: the sketch reads {pending[first]:g} mm. Confirm or correct it."
                  if first in pending else f"Enter the {name} in mm.")
        suggested = {**_suggest(observations, values, missing),
                     **{f"envelope.{a}_mm": v for a, v in pending.items()}}
        return S.MvAbstain(
            stage="dimensions", reason=f"missing_{first}", remedy=remedy,
            partial={"known": {f"envelope.{a}_mm": v for a, v in values.items()},
                     "missing": [f"envelope.{a}_mm" for a in missing],
                     "suggested": suggested,
                     "provenance": {f"envelope.{a}_mm": prov[f"envelope.{a}_mm"] for a in values}})
    return S.Envelope(x_mm=values["x"], y_mm=values["y"], z_mm=values["z"]), prov, warnings


def _suggest(observations, values, missing) -> dict[str, float]:
    """A scaled pre-fill for each missing axis, from a view that shows it next to a known axis."""
    out = {}
    for axis in missing:
        for o in observations:
            a_axis, b_axis, _ = S.FACE_AXES[o.face]
            _, _, w, h = o.outline.bbox
            if a_axis == axis and b_axis in values:
                out[f"envelope.{axis}_mm"] = round(values[b_axis] * (w - 1) / (h - 1) * 2) / 2
                break
            if b_axis == axis and a_axis in values:
                out[f"envelope.{axis}_mm"] = round(values[a_axis] * (h - 1) / (w - 1) * 2) / 2
                break
    return out


# ---- outlines -----------------------------------------------------------------

def _scales(o: Observation, env: S.Envelope) -> tuple[float, float]:
    a_len, b_len = S.face_size(o.face, env)
    _, _, w, h = o.outline.bbox
    return a_len / max(w - 1, 1), b_len / max(h - 1, 1)


def _clamp(pts, a_len, b_len):
    return [(min(max(a, 0.0), a_len), min(max(b, 0.0), b_len)) for a, b in pts]


def _outline_prov(o: Observation) -> str:
    return "measured" if o.mm_per_px and o.scale_confirmed else "scaled"


def observed_outline(o: Observation, env: S.Envelope) -> S.Outline:
    sa, sb = _scales(o, env)
    a_len, b_len = S.face_size(o.face, env)
    bbox = o.outline.bbox

    def mm(points):
        return S.to_canonical(o.face, _clamp(to_face_mm(points, bbox, sa, sb), a_len, b_len), env)

    source = "observed" if o.face in S.CANONICAL_FACES else "mirrored"
    return S.Outline(outer=mm(o.outline.outer), inner=[mm(loop) for loop in o.outline.inner], source=source,
                     confidence=o.confidence)


def canonical_outlines(observations: list[Observation], env: S.Envelope):
    """Best observed or mirrored outline per canonical face, with its provenance, and warnings."""
    out, warnings = {}, []
    for o in observations:
        if o.kind == "photo":
            a_len, b_len = S.face_size(o.face, env)
            _, _, w, h = o.outline.bbox
            if _differs((w - 1) / (h - 1), a_len / b_len):
                warnings.append(f"{o.face}: photo is not square-on, retake it facing the part")
    for face in S.CANONICAL_FACES:
        group = [(o, observed_outline(o, env)) for o in observations if S.CANONICAL_OF[o.face] == face]
        if not group:
            continue
        best_o, best = max(group, key=lambda t: (t[0].confidence, t[1].source == "observed"))
        a_len, b_len = S.face_size(face, env)
        ref = outline_mask(best.outer, best.inner, a_len, b_len)
        for o, ol in group:
            if o is not best_o and iou(outline_mask(ol.outer, ol.inner, a_len, b_len), ref) < 0.9:
                warnings.append(f"{o.face} and {best_o.face} outlines disagree; using {best_o.face}")
        out[face] = (best, _outline_prov(best_o))
    return out, warnings


# ---- features -------------------------------------------------------------------

def _duplicate_through(feats: list[dict], face: str, a: float, b: float, env: S.Envelope) -> bool:
    here = S.to_global(face, a, b, env)
    for f in feats:
        if f["depth_mm"] is None and S.CANONICAL_OF[f["face"]] == S.CANONICAL_OF[face]:
            there = S.to_global(f["face"], f["a_mm"], f["b_mm"], env)
            if all(abs(here[k] - there[k]) <= 1.0 for k in here):
                return True
    return False


def _diameter(o: Observation, i: int, sa: float, sb: float) -> tuple[float, str]:
    """Circle i's diameter in mm and its provenance: written next to it (inferred when unconfirmed or off scale), else
    measured on a photo, else scaled."""
    c = o.outline.circles[i]
    drawn = c.d * o.mm_per_px if o.mm_per_px else c.d * (sa + sb) / 2
    written = [(_value(lv.reading), lv.reading.confirmed) for lv in o.values if lv.hole_index == i]
    if written:
        d, confirmed = written[-1]
        off = drawn > 0 and max(d / drawn, drawn / d) > SCALE_TRAP
        return float(d), "user_written" if confirmed and not off else "inferred"
    return float(drawn), "measured" if o.mm_per_px and o.scale_confirmed else "scaled"


def features_from(observations: list[Observation], env: S.Envelope, edges: dict[int, set[int]] | None = None):
    """Every circle becomes a hole on its own face, except those `edges` (observation index -> circle indices, from
    classify_drawn_circles) reads as edges; a through hole seen from both sides is kept once."""
    feats: list[dict] = []
    prov: dict[str, str] = {}
    edges = edges or {}
    for k_obs, o in enumerate(observations):
        sa, sb = _scales(o, env)
        axis_len = env.length(S.FACE_AXES[o.face][2])
        for i, c in enumerate(o.outline.circles):
            if i in edges.get(k_obs, ()):
                continue
            (a, b), = to_face_mm(np.array([[c.cx, c.cy]]), o.outline.bbox, sa, sb)
            d, d_prov = _diameter(o, i, sa, sb)
            depth, depth_prov = None, None
            if o.blind.get(i):
                if i in o.depth_ratio:
                    depth, depth_prov = o.depth_ratio[i] * axis_len, "estimated"
                else:
                    depth, depth_prov = axis_len / 2, "default"
            elif _duplicate_through(feats, o.face, a, b, env):
                continue
            k = len(feats)
            feats.append({"type": "hole", "face": o.face, "a_mm": a, "b_mm": b, "diameter_mm": d,
                          "depth_mm": depth})
            pos = _outline_prov(o)
            prov.update({S.feature_path(k, "a_mm"): pos, S.feature_path(k, "b_mm"): pos,
                         S.feature_path(k, "diameter_mm"): d_prov})
            if depth is not None:
                prov[S.feature_path(k, "depth_mm")] = depth_prov
    return feats, prov


# ---- drawn circles: holes or edges (drawing-sheet spec 3.4) ---------------------

HIDDEN_TOL = 0.03  # a hidden line lies this close to a circle's side, as a share of the view
EDGE_TOL = 0.04    # a silhouette as wide as a circle, within this share of its diameter, explains it
STATION = 1e-3     # widths are read this share of the length either side of each outline corner
CORNER_DEG = 30.0  # an outline vertex turning less than this lies on a curve, where a width is no step or end


def _hidden_along(g: Observation, s: str, look: str, env: S.Envelope) -> list[float]:
    """Where g's hidden lines that run along `look` lie on axis s, in mm. Their positions are fractions of g's
    bbox from its left and top edges, as drawn."""
    a_axis, b_axis, _ = S.FACE_AXES[g.face]
    a_len, b_len = S.face_size(g.face, env)
    out = []
    for kind, pos, _, _ in g.hidden:
        if kind == "h":  # runs along a; pos runs down from the bbox top, b runs up
            run, across, a, b = a_axis, b_axis, 0.0, (1 - pos) * b_len
        else:
            run, across, a, b = b_axis, a_axis, pos * a_len, 0.0
        if run == look and across == s:
            out.append(S.to_global(g.face, a, b, env)[s])
    return out


def _corners(outer) -> np.ndarray:
    """Which vertices of a pixel outline are corners. The outline follows a curve in short chords that each turn
    a little, so a vertex on a curve is no corner."""
    p = np.asarray(outer, np.float64).reshape(-1, 2)
    before, after = p - np.roll(p, 1, axis=0), np.roll(p, -1, axis=0) - p
    turn = np.arctan2(before[:, 0] * after[:, 1] - before[:, 1] * after[:, 0], (before * after).sum(axis=1))
    return np.degrees(np.abs(turn)) > CORNER_DEG


def _widths(g: Observation, s: str, look: str, env: S.Envelope) -> list[tuple[float, float]]:
    """The extent on axis s of g's outer silhouette just either side of each outline corner along `look`: a step,
    a shoulder or a taper end, never the middle of a taper or a curve. The outline runs along the outside of its
    line and a circle along the middle of its line, so the extent loses half a line width on each side."""
    sa, sb = _scales(g, env)
    pts = [S.to_global(g.face, a, b, env) for a, b in to_face_mm(g.outline.outer, g.outline.bbox, sa, sb)]
    poly = np.array([(p[look], p[s]) for p in pts])
    length = env.length(look)
    at = poly[_corners(g.outline.outer), 0]
    hs = np.unique(np.concatenate([at - STATION * length, at + STATION * length]))
    half = g.stroke * (sa if s == S.FACE_AXES[g.face][0] else sb) / 2
    return [(lo + half, hi - half) for h in hs[(hs > 0) & (hs < length)] for lo, hi in _intervals(poly, h)]


def _intervals(poly: np.ndarray, h: float) -> list[tuple[float, float]]:
    """The pieces of solid where the line `look` = h crosses the closed outline: two pins side by side are two
    pieces, each as wide as its pin, not one span from the first to the last. poly rows are (look, s)."""
    h1, t1 = poly[:, 0], poly[:, 1]
    h2, t2 = np.roll(h1, -1), np.roll(t1, -1)
    cross = (np.minimum(h1, h2) < h) & (h <= np.maximum(h1, h2))
    ts = np.sort(t1[cross] + (t2[cross] - t1[cross]) * (h - h1[cross]) / (h2[cross] - h1[cross]))
    return [(float(ts[k]), float(ts[k + 1])) for k in range(0, len(ts) - 1, 2)]


RING_CHAMFER = 0.3  # a ring this close around a circle (share of its diameter, or 4 line widths) is a chamfer


def chamfer_ring(c, stroke_px: float) -> bool:
    """The circle's concentric ring is a chamfer's edge (a pin's tip), not a hub around a bore: it lies within
    RING_CHAMFER of the diameter, or within four line widths."""
    return bool(c.ring) and c.ring - c.d <= max(RING_CHAMFER * c.d, 4 * stroke_px)


def _verdict(o: Observation, i: int, others: list[Observation], env: S.Envelope) -> tuple[str, str | None]:
    """("hole" | "edge", the view that decided it, or None when no view explains the circle)."""
    c = o.outline.circles[i]
    sa, sb = _scales(o, env)
    a_axis, b_axis, look = S.FACE_AXES[o.face]
    (a, b), = to_face_mm(np.array([[c.cx, c.cy]]), o.outline.bbox, sa, sb)
    at = S.to_global(o.face, a, b, env)
    pairs = []  # (view, the axis it shares with o, the circle's diameter along that axis, and its chamfer ring's)
    for g in others:
        s = next(ax for ax in (a_axis, b_axis) if ax in S.FACE_AXES[g.face][:2])
        scale = sa if s == a_axis else sb
        pairs.append((g, s, c.d * scale, c.ring * scale if chamfer_ring(c, o.stroke) else 0.0))
    for g, s, d, _ in pairs:
        tol = HIDDEN_TOL * env.length(s)
        qs = _hidden_along(g, s, look, env)
        low = {k for k, q in enumerate(qs) if abs(q - (at[s] - d / 2)) <= tol}
        high = {k for k, q in enumerate(qs) if abs(q - (at[s] + d / 2)) <= tol}
        if low and high and len(low | high) >= 2:
            return "hole", g.face
    for g, s, d, ring in pairs:
        if g.outline.circular:
            continue  # every chord of a round silhouette is some width, centred on its axis: none is a step
        for size in (d, ring) if ring else (d,):  # a pin with a chamfered tip: its outer circle is the pin
            if any(abs(hi - lo - size) <= EDGE_TOL * size and abs((hi + lo) / 2 - at[s]) <= EDGE_TOL * size
                   for lo, hi in _widths(g, s, look, env)):
                return "edge", g.face
    return "hole", None


def classify_drawn_circles(observations: list[Observation], env: S.Envelope) -> tuple[dict[int, set[int]], list[str]]:
    """Drawing-sheet spec 3.4: which circles of a line drawing are edges, as observation index -> circle indices,
    and the warnings. A circle is a hole when another line-art view draws its sides hidden, an edge when another
    view's silhouette is that wide there, and otherwise a hole: a visible circle must be some edge. Filled renders
    and sketches keep every circle. Holes no view explains share one warning per face, so a plate stays readable."""
    edges: dict[int, set[int]] = {}
    warnings = []
    for k, o in enumerate(observations):
        if not o.line_art:
            continue
        others = [g for g in observations if g.line_art and S.CANONICAL_OF[g.face] != S.CANONICAL_OF[o.face]]
        sa, sb = _scales(o, env)
        lone = []
        for i in range(len(o.outline.circles)):
            kind, by = _verdict(o, i, others, env)
            d = round(_diameter(o, i, sa, sb)[0], 1)
            if kind == "edge":
                edges.setdefault(k, set()).add(i)
                warnings.append(f"Circle Ø{d:g} on {o.face}: read as an edge (step in {by})")
            elif by is None:
                lone.append(d)
        if len(lone) == 1:
            warnings.append(f"Circle Ø{lone[0]:g} on {o.face}: read as a hole, no other view explains it")
        elif lone:
            sizes = ", ".join(f"Ø{d:g}" for d in sorted(lone))
            warnings.append(f"{len(lone)} circles on {o.face} ({sizes}): read as holes, no other view explains them")
    return edges, warnings


# ---- snapping -----------------------------------------------------------------

def _grid(v: float) -> float:
    return round(v * 2) / 2


def snap_diameter(d: float, clearance: str = "medium") -> float:
    best = min(CLEARANCE_CLASSES[clearance], key=lambda c: abs(c - d))
    return best if abs(best - d) <= 0.4 else _grid(d)


def snap_coord(v: float, length: float) -> float:
    """Envelope edges exactly, thin walls to standard thicknesses, everything else to 0.5 mm."""
    if v <= 0.5:
        return 0.0
    if length - v <= 0.5:
        return float(length)
    wall = min(v, length - v)
    if wall < 12:
        t = min(THICKNESS_MM, key=lambda t: abs(t - wall))
        if abs(t - wall) <= 0.3:
            return t if v < length / 2 else float(length - t)
    return min(max(_grid(v), 0.0), float(length))


EDGE_SLOPE = 0.035  # tan(2 deg): how far off-axis a "straight" outline edge may drift
EDGE_MIN_MM = 1.0    # shorter edges are curve segments, not sketched straight lines
MOVE_LIMIT = 0.02    # a level does not move more than this fraction of its axis


def _level_targets(levels: set, axis_len: float, all_coords) -> dict:
    """snap_coord per level, kept only if it moves, the move is small, and it lands neither on another
    level's snapped or original value nor on any other vertex's coordinate on this axis (spec 4.5)."""
    candidates = {v: snap_coord(v, axis_len) for v in levels}
    targets = {}
    for v, new in candidates.items():
        if abs(new - v) <= 1e-9 or abs(new - v) > MOVE_LIMIT * axis_len:
            targets[v] = v
            continue
        collides = any(w != v and (abs(new - w) <= 1e-6 or abs(new - candidates[w]) <= 1e-6) for w in levels)
        if not collides:
            collides = any(abs(c - v) > 1e-6 and abs(new - c) <= 1e-6 for c in all_coords)
        targets[v] = v if collides else new
    return targets


def _snap_outline(points: list, a_len: float, b_len: float) -> list:
    """Snap only straight axis-parallel edges (spec 4.5); curves and sloped or thin edges pass through."""
    n = len(points)
    if n < 3:
        return list(points)
    on_horiz, on_vert = [False] * n, [False] * n
    horiz_edges, vert_edges = [], []
    for i in range(n):
        a1, b1 = points[i]
        a2, b2 = points[(i + 1) % n]
        da, db = a2 - a1, b2 - b1
        length = (da * da + db * db) ** 0.5
        if length < EDGE_MIN_MM:
            continue
        if abs(db) <= EDGE_SLOPE * abs(da):
            on_horiz[i] = on_horiz[(i + 1) % n] = True
            horiz_edges.append((i, (i + 1) % n))
        elif abs(da) <= EDGE_SLOPE * abs(db):
            on_vert[i] = on_vert[(i + 1) % n] = True
            vert_edges.append((i, (i + 1) % n))
    all_a = [p[0] for p in points]
    all_b = [p[1] for p in points]
    a_targets = _level_targets({round(points[i][0], 6) for i in range(n) if on_vert[i]}, a_len, all_a)
    b_targets = _level_targets({round(points[i][1], 6) for i in range(n) if on_horiz[i]}, b_len, all_b)
    na = [a_targets.get(round(a, 6), a) if on_vert[i] else a for i, (a, _) in enumerate(points)]
    nb = [b_targets.get(round(b, 6), b) if on_horiz[i] else b for i, (_, b) in enumerate(points)]
    _keep_straight(points, na, vert_edges, axis=0)
    _keep_straight(points, nb, horiz_edges, axis=1)
    return list(zip(na, nb))


def _keep_straight(points: list, new: list, edges: list, axis: int) -> None:
    """A straight edge never ends up more slanted than it was: when its two ends would snap to different
    values, both keep their own. Repeats because an end may be shared by a chain of edges."""
    changed = True
    while changed:
        changed = False
        for i, j in edges:
            if abs(new[i] - new[j]) > abs(points[i][axis] - points[j][axis]) + 1e-9:
                new[i], new[j] = points[i][axis], points[j][axis]
                changed = True


def _crosses(points: list, moved: set) -> bool:
    """Whether an edge touching a moved vertex properly crosses any non-adjacent edge."""
    n = len(points)

    def ccw(p, q, r):
        return (r[1] - p[1]) * (q[0] - p[0]) - (q[1] - p[1]) * (r[0] - p[0])

    for i in {k for m in moved for k in ((m - 1) % n, m)}:
        p, q = points[i], points[(i + 1) % n]
        for j in range(n):
            if j in (i, (i - 1) % n, (i + 1) % n):
                continue
            r, s = points[j], points[(j + 1) % n]
            if ccw(p, q, r) * ccw(p, q, s) < 0 and ccw(r, s, p) * ccw(r, s, q) < 0:
                return True
    return False


def outline_kinds(observations: list[Observation]) -> dict[str, str]:
    """The input kind (sketch, photo, drawing) behind each canonical outline: the observation
    `canonical_outlines` picks, by the same rule."""
    kinds = {}
    for face in S.CANONICAL_FACES:
        group = [o for o in observations if S.CANONICAL_OF[o.face] == face]
        if group:
            kinds[face] = max(group, key=lambda o: (o.confidence, o.face in S.CANONICAL_FACES)).kind
    return kinds


def _snap_sketch(points: list, a_len: float, b_len: float) -> list:
    """A hand sketch: every vertex to the envelope edges, standard walls and the 0.5 mm grid, which squares
    wobbly strokes and closes cut corners. Points that land on each other merge."""
    out: list = []
    for a, b in points:
        p = (snap_coord(a, a_len), snap_coord(b, b_len))
        if not out or p != out[-1]:
            out.append(p)
    if len(out) > 1 and out[0] == out[-1]:
        out.pop()
    return out


def snap(data: dict, clearance: str = "medium", kinds: dict | None = None) -> None:
    """Snap scaled, inferred and estimated values of a spec dict in place (spec 4.5). Sketched outlines are
    squared vertex by vertex; drawn and photographed ones keep their curves and thin features (level snap)."""
    prov, snapped = data["provenance"], data.setdefault("snapped", [])
    for k, f in enumerate(data["features"]):
        for name in ("a_mm", "b_mm", "diameter_mm", "depth_mm", "width_mm", "length_mm", "height_mm"):
            path = S.feature_path(k, name)
            if f.get(name) is None or prov.get(path) not in SNAPPABLE:
                continue
            hole = name == "diameter_mm" and f.get("type", "hole") == "hole"  # clearance sizes are for holes
            new = snap_diameter(f[name], clearance) if hole else _grid(f[name])
            if new > 0 and abs(new - f[name]) > 1e-9:
                f[name] = new
                snapped.append(path)
    lengths = {"x": data["envelope"]["x_mm"], "y": data["envelope"]["y_mm"], "z": data["envelope"]["z_mm"]}
    for face in S.CANONICAL_FACES:
        path = f"views.{face}.outer"
        if prov.get(path) not in SNAPPABLE:
            continue
        a_axis, b_axis, _ = S.FACE_AXES[face]
        outline = data["views"][face]
        orig = [tuple(p) for p in outline["outer"]]
        if (kinds or {}).get(face) == "sketch":
            new = _snap_sketch(orig, lengths[a_axis], lengths[b_axis])
            ok = len(set(new)) >= 3 and new != orig and not _crosses(new, set(range(len(new))))
        else:
            new = _snap_outline(outline["outer"], lengths[a_axis], lengths[b_axis])
            moved = {i for i, (p, q) in enumerate(zip(orig, new)) if p != q}
            ok = bool(moved) and len(set(new)) >= len(set(orig)) and not _crosses(new, moved)
        if ok:
            outline["outer"] = new
            snapped.append(path)


def _drop_features(data: dict, removed: set[int]) -> None:
    """Take features out of a spec dict and renumber their provenance paths."""
    keep = [k for k in range(len(data["features"])) if k not in removed]
    new_index = {old: new for new, old in enumerate(keep)}
    data["features"] = [data["features"][k] for k in keep]
    prov = {}
    for path, p in data["provenance"].items():
        m = _FEATURE_PATH.fullmatch(path)
        if not m:
            prov[path] = p
        elif int(m.group(1)) in new_index:
            prov[S.feature_path(new_index[int(m.group(1))], m.group(2))] = p
    data["provenance"] = prov


def assemble(env: S.Envelope, env_prov: dict, outlines: dict, feats: list[dict], feat_prov: dict,
             warnings: list[str], user_values: dict | None = None, accepted=(), snap_values: bool = True,
             clearance: str = "medium", kinds: dict | None = None) -> S.MultiViewSpec:
    """outlines: canonical face -> (Outline, provenance). Applies the user's edits, then snapping."""
    data = {
        "envelope": env.model_dump(),
        "views": {face: ol.model_dump() for face, (ol, _) in outlines.items()},
        "features": [dict(f) for f in feats],
        "finishes": [],
        "provenance": {**env_prov, **{f"views.{face}.outer": p for face, (_, p) in outlines.items()}, **feat_prov},
        "warnings": list(dict.fromkeys(warnings)),
        "confidence": round(min(ol.confidence for ol, _ in outlines.values()), 3),
    }
    for face in accepted:
        if f"views.{face}.outer" in data["provenance"]:
            data["provenance"][f"views.{face}.outer"] = "user_edited"
    removed = set()
    for path, value in (user_values or {}).items():
        m = _FEATURE_PATH.fullmatch(path)
        if not m:
            continue
        k, name = int(m.group(1)), m.group(2)
        feature = data["features"][k] if k < len(data["features"]) else None
        if feature is None or (name != "keep" and (name not in feature or name in ("type", "face"))):
            # an edit past the list or onto a feature without that field is said, never applied (the web app also
            # drops typed feature values when the rejected faces change, since that can renumber the features)
            warning = left_out(name, feature is not None)
            if warning not in data["warnings"]:
                data["warnings"].append(warning)
            continue
        if name == "keep":  # the user took a misread feature out: "features[k].keep" = 0
            if not value:
                removed.add(k)
            continue
        feature[name] = float(value)
        data["provenance"][path] = "user_edited"
    if removed:
        _drop_features(data, removed)
    if snap_values:
        snap(data, clearance, kinds)
    return S.MultiViewSpec.model_validate(data)
