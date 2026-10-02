"""Studio geometry settings applied to a MultiViewSpec. Spec 2026-09-23-studio section 4.
spec.py is unchanged: the finish uses the existing Fillet and Chamfer models, and the size the user chose is
user_edited. "all_vertical" means edges along the depth axis, as the multi-view spec section 5 defines it."""
from __future__ import annotations

import math

from s2c.multiview import build as _build_mod
from s2c.multiview.build import BuildError
from s2c.multiview.settings import GeometrySettings
from s2c.multiview.spec import MultiViewSpec, finish_path

FINISH_LIMIT = 0.45  # of the smallest envelope side


def max_finish_mm(spec: MultiViewSpec) -> float:
    env = spec.envelope
    return round(FINISH_LIMIT * min(env.x_mm, env.y_mm, env.z_mm), 2)


def apply_geometry(spec: MultiViewSpec, geometry: GeometrySettings) -> tuple[MultiViewSpec, list[str]]:
    if geometry.finish == "none":
        return spec, []
    size = min(geometry.finish_mm, max_finish_mm(spec))
    if size == geometry.finish_mm:
        warnings = []
    else:
        msg = f"{geometry.finish.capitalize()} reduced to {size:g} mm to fit the part"
        warnings = [msg]
    data = spec.model_dump()
    k = len(data["finishes"])
    data["finishes"].append({"type": geometry.finish, "edges": geometry.finish_edges, "radius_mm": size})
    data["provenance"][finish_path(k, "radius_mm")] = "user_edited"
    data["warnings"] = [*data["warnings"], *warnings]
    return MultiViewSpec.model_validate(data), warnings


def _floor_mm(value: float) -> float:
    return math.floor(value * 10 + 1e-9) / 10


def largest_finish(spec: MultiViewSpec, geometry: GeometrySettings, lo: float = 0.2) -> float | None:
    """The largest finish size (0.1 mm steps) that actually builds, for the failure message. Bisects between
    lo and the size that was tried, at most 7 calls to build.build; never goes through build_part, so nothing
    is written to disk or the part cache."""
    def fits(size: float) -> bool:
        trial, _ = apply_geometry(spec, geometry.model_copy(update={"finish_mm": size}))
        try:
            _build_mod.build(trial)
            return True
        except BuildError:
            return False

    hi = _floor_mm(min(geometry.finish_mm, max_finish_mm(spec)))
    if hi <= lo:
        return round(lo, 1) if fits(lo) else None
    if not fits(lo):
        return None
    if fits(hi):
        return hi
    best, low, high = lo, lo, hi
    for _ in range(5):
        mid = _floor_mm((low + high) / 2)
        if mid <= best:
            break
        if fits(mid):
            best, low = mid, mid
        else:
            high = mid
    return round(best, 1)
