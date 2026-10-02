"""MultiViewSpec -> CadQuery solid: intersection of three extruded outlines, then face features and finishes.
Spec section 6.4. Deterministic; no model output is ever executed here."""
from __future__ import annotations

import logging
from pathlib import Path

import cadquery as cq

from s2c.multiview import exporters
from s2c.multiview.spec import (
    FACE_AXES,
    Envelope,
    FaceBoss,
    FaceHole,
    FacePocket,
    Fillet,
    MultiViewSpec,
    Outline,
    face_size,
)


class BuildError(Exception):
    def __init__(self, reason: str, remedy: str):
        super().__init__(f"{reason}: {remedy}")
        self.reason, self.remedy = reason, remedy


INVALID = ("invalid_solid", "Simplify the outline or retake the photo.")
EMPTY = ("intersection_empty", "The views do not describe one part. Check which face each photo shows.")
log = logging.getLogger(__name__)


FUZZY_MM = 1e-4  # a retry tolerance far below any drawn or measured detail


def _intersect(a: cq.Workplane, b: cq.Workplane) -> cq.Workplane:
    """Exact boolean first. OCC can return nothing when two faces almost coincide (a hole edge 0.02 mm from a
    foot edge), so an empty or invalid result is retried as a fuzzy boolean; a truly empty one stays empty."""
    out = a.intersect(b)
    solids = out.solids().vals()
    if solids and all(s.isValid() for s in solids):
        return out
    fuzzy = a.intersect(b, tol=FUZZY_MM)
    return fuzzy if fuzzy.solids().vals() else out


def volume(solid: cq.Workplane) -> float:
    return float(sum(s.Volume() for s in solid.solids().vals()))


# ---- the three prisms -------------------------------------------------------

def _canonical_plane(face: str, env: Envelope) -> tuple[cq.Plane, float]:
    """Build plane of a canonical face and the extrusion length that fills the envelope.
    Local (u, v): front (X, Y), top (X, Z), right (Z, Y). CadQuery sets yDir = normal x xDir."""
    if face == "front":
        return cq.Plane(origin=(0, 0, 0), xDir=(1, 0, 0), normal=(0, 0, 1)), env.z_mm
    if face == "top":
        return cq.Plane(origin=(0, env.y_mm, 0), xDir=(1, 0, 0), normal=(0, -1, 0)), env.y_mm
    return cq.Plane(origin=(env.x_mm, 0, 0), xDir=(0, 0, 1), normal=(-1, 0, 0)), env.x_mm


def _plane_uv(face: str, pts, env: Envelope) -> list[tuple[float, float]]:
    """Canonical face-frame points -> local (u, v) of the build plane (spec section 3)."""
    if face == "front":
        return [(a, b) for a, b in pts]
    if face == "top":
        return [(a, env.z_mm - b) for a, b in pts]
    return [(env.z_mm - a, b) for a, b in pts]


def _clean(pts) -> list[tuple[float, float]]:
    out: list[tuple[float, float]] = []
    for a, b in pts:
        p = (round(float(a), 6), round(float(b), 6))
        if not out or p != out[-1]:
            out.append(p)
    if len(out) > 1 and out[0] == out[-1]:
        out.pop()
    return out


def _prism(face: str, outline: Outline, env: Envelope) -> cq.Workplane:
    plane, length = _canonical_plane(face, env)
    outer = _clean(_plane_uv(face, outline.outer, env))
    if len(outer) < 3:
        raise BuildError(*INVALID)
    solid = cq.Workplane(plane).polyline(outer).close().extrude(length)
    for loop in outline.inner:
        pts = _clean(_plane_uv(face, loop, env))
        if len(pts) >= 3:
            solid = solid.cut(cq.Workplane(plane).polyline(pts).close().extrude(length))
    return solid


def _check(solid: cq.Workplane) -> None:
    solids = solid.solids().vals()
    if not solids or volume(solid) < 1e-6:
        raise BuildError(*EMPTY)
    if len(solids) > 1 or not solids[0].isValid():
        raise BuildError(*INVALID)


# ---- features ----------------------------------------------------------------

def _face_plane(face: str, env: Envelope, offset: float = 0.0) -> cq.Plane:
    """Plane on an envelope face: origin at (a, b) = (0, 0), xDir along +a, normal pointing out of the part.
    For every face, normal x (+a) = +b, so local (u, v) = (a, b)."""
    x, y, z = env.x_mm, env.y_mm, env.z_mm
    origin, x_dir, normal = {
        "front": ((0, 0, z), (1, 0, 0), (0, 0, 1)),
        "back": ((x, 0, 0), (-1, 0, 0), (0, 0, -1)),
        "top": ((0, y, z), (1, 0, 0), (0, 1, 0)),
        "bottom": ((0, 0, 0), (1, 0, 0), (0, -1, 0)),
        "right": ((x, 0, z), (0, 0, -1), (1, 0, 0)),
        "left": ((0, 0, 0), (0, 0, 1), (-1, 0, 0)),
    }[face]
    moved = tuple(o + n * offset for o, n in zip(origin, normal))
    return cq.Plane(origin=moved, xDir=x_dir, normal=normal)


BOSS_CLEAR = 0.15  # the cut around a pin reaches this share of its diameter (1 mm at least) past it on each side


def _cut_feature(solid: cq.Workplane, f, env: Envelope, typed: bool = True) -> cq.Workplane:
    a_len, b_len = face_size(f.face, env)
    if not (0 <= f.a_mm <= a_len and 0 <= f.b_mm <= b_len):
        raise BuildError("feature_outside_part", "A hole or slot lies outside the part. Check its position.")
    if isinstance(f, (FacePocket, FaceBoss)):
        out = solid.cut(_open_cut(f, env))
        pieces = out.solids().vals()
        if len(pieces) > len(solid.solids().vals()) or not all(p.isValid() for p in pieces):
            log.warning("a %s on %s would split the part; left out", f.type, f.face)
            return solid  # read from lines, not typed: a cut that splits the part misread them
        return out
    if f.depth_mm is None:  # through: start 1 mm outside, end 1 mm past the far side
        wp, dist = cq.Workplane(_face_plane(f.face, env, 1.0)), env.length(FACE_AXES[f.face][2]) + 2.0
    else:  # blind: depth measured from the envelope face inward
        wp, dist = cq.Workplane(_face_plane(f.face, env)), f.depth_mm
    wp = wp.center(f.a_mm, f.b_mm)
    shape = wp.circle(f.diameter_mm / 2) if isinstance(f, FaceHole) else wp.slot2D(f.length_mm, f.width_mm, f.angle_deg)
    out = solid.cut(shape.extrude(-dist))
    pieces = out.solids().vals()
    if not typed and (not pieces or len(pieces) > len(solid.solids().vals()) or not all(p.isValid() for p in pieces)):
        log.warning("a %s read on %s would split the part or leave no valid solid; left out", f.type, f.face)
        return solid  # read from a drawn circle, not typed: as for pockets, the reading is wrong, not the part
    return out


def _open_cut(f: FacePocket | FaceBoss, env: Envelope) -> cq.Workplane:
    """The material a pocket or a boss removes. Both open on their face, so the cut starts 1 mm outside it."""
    far = env.length(FACE_AXES[f.face][2])
    depth = f.height_mm if isinstance(f, FaceBoss) else (far + 1.0 if f.depth_mm is None else f.depth_mm)
    plane = _face_plane(f.face, env, 1.0)
    if isinstance(f, FacePocket):
        return cq.Workplane(plane).center(f.a_mm, f.b_mm).rect(f.width_mm, f.height_mm).extrude(-(depth + 1.0))
    side = f.diameter_mm + 2 * max(1.0, BOSS_CLEAR * f.diameter_mm)  # the hull's pin is a drawn line wider than the pin
    square = cq.Workplane(plane).center(f.a_mm, f.b_mm).rect(side, side).extrude(-(depth + 1.0))
    return square.cut(cq.Workplane(plane).center(f.a_mm, f.b_mm).circle(f.diameter_mm / 2).extrude(-(depth + 1.0)))


_EDGE_SELECTORS = {"all": None, "all_vertical": "|Z", "top": ">Z", "bottom": "<Z"}


def _apply_finish(solid: cq.Workplane, finish) -> cq.Workplane:
    kind = "fillet" if isinstance(finish, Fillet) else "chamfer"
    failed = BuildError(f"{kind}_failed", "Reduce the fillet radius." if kind == "fillet" else "Reduce the chamfer size.")
    selector = _EDGE_SELECTORS[finish.edges]
    edges = solid.edges() if selector is None else solid.edges(selector)
    try:
        out = edges.fillet(finish.radius_mm) if kind == "fillet" else edges.chamfer(finish.radius_mm)
    except Exception as e:  # OCC raises StdFail_NotDone and friends
        raise failed from e
    if not out.solids().vals() or not out.val().isValid():
        raise failed
    return out


def _turn(hull: cq.Workplane, spec: MultiViewSpec) -> cq.Workplane:
    """A round part cut down to its solid of revolution, so a hub comes out round instead of square.
    The turn only refines a hull that already built: if OCC fails on it, the hull is kept."""
    from s2c.multiview import turned  # deferred: turned reuses _clean from this module

    axis = None
    try:
        axis = turned.turned_axis(spec)
        if axis is None:
            return hull
        solid = _intersect(hull, turned.revolve(spec, axis))
        _check(solid)
    except Exception as e:  # noqa: BLE001 - OCC raises anything; the hull is a valid answer
        log.warning("turned build around %s failed, keeping the hull: %s", axis, e)
        return hull
    return solid


def build(spec: MultiViewSpec, notes: list[str] | None = None) -> cq.Workplane:
    """The part. A feature read from a drawing that would break it is left out, and `notes` gets one line saying so."""
    env = spec.envelope
    try:
        solid = _prism("front", spec.views.front, env)
        for face in ("top", "right"):
            solid = _intersect(solid, _prism(face, getattr(spec.views, face), env))
            if not solid.solids().vals():  # an empty result would make CadQuery fall back to an earlier solid
                raise BuildError(*EMPTY)
    except BuildError:
        raise
    except Exception as e:
        raise BuildError(*INVALID) from e
    _check(solid)
    turned = _turn(solid, spec)
    if turned is solid:
        return _finish_part(solid, spec, notes)
    try:
        return _finish_part(turned, spec, notes)
    except BuildError as e:  # a revolve's faceted rim can refuse a fillet the hull takes
        log.warning("finishing the turned part failed (%s), finishing the hull instead", e.reason)
        return _finish_part(solid, spec, notes)


def _finish_part(solid: cq.Workplane, spec: MultiViewSpec, notes: list[str] | None = None) -> cq.Workplane:
    """Holes, slots, then fillets and chamfers, on a solid that already passed _check."""
    left_out = []
    for i, f in enumerate(spec.features):
        prov = spec.provenance.get(f"features[{i}].diameter_mm", spec.provenance.get(f"features[{i}].width_mm"))
        cut = _cut_feature(solid, f, spec.envelope, typed=prov in ("user_written", "user_edited"))
        if cut is solid:
            typed_too = " Type its size to keep it." if not isinstance(f, (FacePocket, FaceBoss)) else ""
            left_out.append(f"A {f.type} read on the {f.face} view would break the part, so it was left out.{typed_too}")
        solid = cut
    for finish in spec.finishes:
        solid = _apply_finish(solid, finish)
    _check(solid)
    if notes is not None:  # only the attempt that built says what it left out
        notes.extend(left_out)
    return solid


def export(solid: cq.Workplane, out_dir: Path) -> tuple[Path, Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    step, stl = out_dir / "part.step", out_dir / "part.stl"
    with exporters._OCCT_LOCK:
        cq.exporters.export(solid, str(step))
        cq.exporters.export(solid, str(stl), tolerance=0.01, angularTolerance=0.1)
    return step, stl
