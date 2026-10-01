"""Shared builders for the multi-view tests."""
import math

import numpy as np

from s2c.multiview.spec import Chamfer, FaceBoss, FaceHole, FacePocket, FaceSlot, Fillet, MultiViewSpec, numeric_names


def rect(w, h, x0=0.0, y0=0.0):
    return [(x0, y0), (x0 + w, y0), (x0 + w, y0 + h), (x0, y0 + h)]


def circle(cx, cy, r, n=180):
    return [(cx + r * math.cos(2 * math.pi * i / n), cy + r * math.sin(2 * math.pi * i / n)) for i in range(n)]


def outline(pts, inner=(), source="observed", confidence=0.9):
    return {"outer": list(pts), "inner": [list(p) for p in inner], "source": source, "confidence": confidence}


_FEATURES = {"hole": FaceHole, "slot": FaceSlot, "pocket": FacePocket, "boss": FaceBoss}
_FINISHES = {"fillet": Fillet, "chamfer": Chamfer}


def make_spec(env, front=None, top=None, right=None, features=(), finishes=(), prov="user_written"):
    x, y, z = env
    provenance = {"envelope.x_mm": prov, "envelope.y_mm": prov, "envelope.z_mm": prov,
                  "views.front.outer": prov, "views.top.outer": prov, "views.right.outer": prov}
    for i, f in enumerate(features):
        provenance.update({f"features[{i}].{n}": prov for n in numeric_names(_FEATURES[f["type"]](**f))})
    for i, f in enumerate(finishes):
        provenance.update({f"finishes[{i}].{n}": prov for n in numeric_names(_FINISHES[f["type"]](**f))})
    return MultiViewSpec.model_validate({
        "envelope": {"x_mm": x, "y_mm": y, "z_mm": z},
        "views": {"front": front or outline(rect(x, y)), "top": top or outline(rect(x, z)),
                  "right": right or outline(rect(z, y))},
        "features": list(features), "finishes": list(finishes), "provenance": provenance, "confidence": 0.9,
    })


def box_mesh(x, y, z):
    from s2c.multiview.raster import Mesh
    v = np.array([[0, 0, 0], [x, 0, 0], [x, y, 0], [0, y, 0], [0, 0, z], [x, 0, z], [x, y, z], [0, y, z]], float)
    f = np.array([[0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7], [0, 1, 5], [0, 5, 4],
                  [1, 2, 6], [1, 6, 5], [2, 3, 7], [2, 7, 6], [3, 0, 4], [3, 4, 7]])
    return Mesh(v, f)
