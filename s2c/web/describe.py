"""Deterministic part builder for the Describe chat. The chat model only names a part type and which of the
user's own numbers go where; this module checks them and draws the MultiViewSpec. No model output is ever
executed. Plan: docs/superpowers/plans/2026-09-27-describe-chat-plan.md ("Contract")."""
from __future__ import annotations

import math
from typing import Literal

from pydantic import BaseModel, Field

from s2c.multiview.spec import MultiViewSpec, feature_path

PartType = Literal["plate", "l_bracket", "spacer", "flange"]
REQUIRED: dict[str, tuple[str, ...]] = {
    "plate": ("width_mm", "height_mm", "thickness_mm"),
    "l_bracket": ("leg_a_mm", "leg_b_mm", "width_mm", "thickness_mm"),
    "spacer": ("outer_diameter_mm", "inner_diameter_mm", "length_mm"),
    "flange": ("outer_diameter_mm", "inner_diameter_mm", "thickness_mm", "bolt_circle_diameter_mm",
               "bolt_hole_diameter_mm", "bolt_count"),
}
CIRCLE_POINTS = 96


class Hole(BaseModel):
    a_mm: float
    b_mm: float
    diameter_mm: float


class PartRequest(BaseModel):
    type: PartType
    values: dict[str, float] = {}
    holes: list[Hole] = Field(default_factory=list)


def check(req: PartRequest) -> list[str]:
    """Keys that are missing or not buildable, in table order. Empty means the part can be built."""
    v = req.values
    missing = [k for k in REQUIRED[req.type] if k not in v or not v[k] > 0]
    bad: list[str] = []
    if (req.type in ("spacer", "flange") and {"outer_diameter_mm", "inner_diameter_mm"} <= v.keys()
            and v["inner_diameter_mm"] >= v["outer_diameter_mm"]):
        bad.append("inner_diameter_mm")
    if req.type == "flange" and not missing and not bad:
        od, idia, bc, hd = (v[k] for k in ("outer_diameter_mm", "inner_diameter_mm", "bolt_circle_diameter_mm",
                                           "bolt_hole_diameter_mm"))
        if not idia < bc < od:
            bad.append("bolt_circle_diameter_mm")
        elif bc - hd <= idia or bc + hd >= od:
            bad.append("bolt_hole_diameter_mm")
        if v["bolt_count"] != int(v["bolt_count"]) or v["bolt_count"] > 64:
            bad.append("bolt_count")
    if (req.type == "l_bracket" and {"leg_a_mm", "leg_b_mm", "thickness_mm"} <= v.keys()
            and not v["thickness_mm"] < min(v["leg_a_mm"], v["leg_b_mm"])):
        bad.append("thickness_mm")
    if req.type == "plate" and {"width_mm", "height_mm"} <= v.keys():
        for i, h in enumerate(req.holes):
            r = h.diameter_mm / 2
            if not (h.diameter_mm > 0 and r <= h.a_mm <= v["width_mm"] - r and r <= h.b_mm <= v["height_mm"] - r):
                bad.append(f"holes[{i}]")
    return list(dict.fromkeys(missing + [k for k in bad if k not in missing]))


def _rect(w: float, h: float) -> list[tuple[float, float]]:
    return [(0.0, 0.0), (w, 0.0), (w, h), (0.0, h)]


def _circle(c: float, r: float) -> list[tuple[float, float]]:
    pts = []
    for i in range(CIRCLE_POINTS):
        t = 2 * math.pi * i / CIRCLE_POINTS
        pts.append((min(max(c + r * math.cos(t), 0.0), 2 * c), min(max(c + r * math.sin(t), 0.0), 2 * c)))
    return pts


def _outline(outer, inner=()) -> dict:
    return {"outer": outer, "inner": [list(loop) for loop in inner], "source": "observed", "confidence": 1.0}


def spec_from_request(req: PartRequest) -> MultiViewSpec:
    """Build the spec. Call only when check(req) is empty."""
    v, t = req.values, req.type
    prov = {"envelope.x_mm": "user_written", "envelope.y_mm": "user_written", "envelope.z_mm": "user_written",
            "views.front.outer": "user_written", "views.top.outer": "user_written",
            "views.right.outer": "user_written"}
    features: list[dict] = []
    if t == "plate":
        w, h, th = v["width_mm"], v["height_mm"], v["thickness_mm"]
        env, front, top, right = (w, h, th), _outline(_rect(w, h)), _outline(_rect(w, th)), _outline(_rect(th, h))
        for i, hole in enumerate(req.holes):
            features.append({"type": "hole", "face": "front", "a_mm": hole.a_mm, "b_mm": hole.b_mm,
                             "diameter_mm": hole.diameter_mm})
            prov.update({feature_path(i, k): "user_written" for k in ("a_mm", "b_mm", "diameter_mm")})
    elif t == "l_bracket":
        a, b, w, th = v["leg_a_mm"], v["leg_b_mm"], v["width_mm"], v["thickness_mm"]
        env = (a, b, w)
        front = _outline([(0.0, 0.0), (a, 0.0), (a, th), (th, th), (th, b), (0.0, b)])
        top, right = _outline(_rect(a, w)), _outline(_rect(w, b))
    else:
        d, di = v["outer_diameter_mm"], v["inner_diameter_mm"]
        length = v["length_mm"] if t == "spacer" else v["thickness_mm"]
        c = d / 2
        env = (d, d, length)
        front = _outline(_circle(c, c), [_circle(c, di / 2)])
        top, right = _outline(_rect(d, length)), _outline(_rect(length, d))
        prov["views.front.outer"] = "scaled"  # the centre is derived, not written
        if t == "flange":
            n, rb = int(v["bolt_count"]), v["bolt_circle_diameter_mm"] / 2
            for i in range(n):
                ang = math.pi / 2 + 2 * math.pi * i / n
                features.append({"type": "hole", "face": "front", "a_mm": round(c + rb * math.cos(ang), 4),
                                 "b_mm": round(c + rb * math.sin(ang), 4),
                                 "diameter_mm": v["bolt_hole_diameter_mm"]})
                prov.update({feature_path(i, "a_mm"): "scaled", feature_path(i, "b_mm"): "scaled",
                             feature_path(i, "diameter_mm"): "user_written"})
    x, y, z = env
    return MultiViewSpec.model_validate({
        "envelope": {"x_mm": x, "y_mm": y, "z_mm": z},
        "views": {"front": front, "top": top, "right": right},
        "features": features, "provenance": prov, "confidence": 1.0,
    })
