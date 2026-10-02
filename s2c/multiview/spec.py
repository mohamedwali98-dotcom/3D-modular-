"""Multi-view contract, face frames and the mirror rule. Spec 2026-09-22, sections 3 and 5.
Owned by the geometry owner. Does not touch s2c/partspec/."""
from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

MAX_MM = 10_000.0   # 10 m: past any part this app builds, and it keeps an infinite or absurd size out of CadQuery
MAX_POINTS = 5000   # outline points per loop
MAX_LOOPS = 200     # openings per outline
MAX_FEATURES = 200
MAX_FINISHES = 10
Mm = Annotated[float, Field(gt=0, le=MAX_MM, description="millimetres")]
Face = Literal["front", "back", "left", "right", "top", "bottom"]
MvProvenance = Literal["user_written", "measured", "user_edited", "scaled", "inferred", "estimated", "default"]
ViewSource = Literal["observed", "mirrored", "inferred", "assumed"]
EdgeSelector = Literal["all", "all_vertical", "top", "bottom"]
Point = tuple[float, float]

FACES = ("front", "back", "left", "right", "top", "bottom")
CANONICAL_FACES = ("front", "top", "right")
TRUSTED = frozenset({"user_written", "measured", "user_edited"})
CANONICAL_OF = {"front": "front", "back": "front", "top": "top", "bottom": "top", "right": "right", "left": "right"}
# face -> (global axis of a, global axis of b, axis the face looks along)
FACE_AXES = {
    "front": ("x", "y", "z"), "back": ("x", "y", "z"),
    "top": ("x", "z", "y"), "bottom": ("x", "z", "y"),
    "right": ("z", "y", "x"), "left": ("z", "y", "x"),
}
AXIS_NAMES = {"x": "width", "y": "height", "z": "depth"}
POINT_TOLERANCE_MM = 0.5


class _Strict(BaseModel):
    """extra="forbid" rejects fields a model invents, such as an estimated width."""
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class Envelope(_Strict):
    x_mm: Mm
    y_mm: Mm
    z_mm: Mm

    def length(self, axis: str) -> float:
        return {"x": self.x_mm, "y": self.y_mm, "z": self.z_mm}[axis]


class Outline(_Strict):
    outer: list[Point] = Field(min_length=3, max_length=MAX_POINTS)
    inner: list[Annotated[list[Point], Field(max_length=MAX_POINTS)]] = Field(default=[], max_length=MAX_LOOPS)
    source: ViewSource
    confidence: float = Field(ge=0, le=1)


class FaceHole(_Strict):
    type: Literal["hole"] = "hole"
    face: Face
    a_mm: float
    b_mm: float
    diameter_mm: Mm
    depth_mm: Mm | None = None  # None means through along the face axis


class FaceSlot(_Strict):
    type: Literal["slot"] = "slot"
    face: Face
    a_mm: float
    b_mm: float
    width_mm: Mm
    length_mm: Mm  # end to end
    angle_deg: float = 0.0
    depth_mm: Mm | None = None


class FacePocket(_Strict):
    """An axis-aligned rectangular cut from the envelope face inward, centred at (a, b): a pocket, a step or, running
    off the outline, a corner notch. Read from the inner lines of a drawing (complex-parts spec 2026-10-01)."""
    type: Literal["pocket"] = "pocket"
    face: Face
    a_mm: float
    b_mm: float
    width_mm: Mm   # along a
    height_mm: Mm  # along b
    depth_mm: Mm | None = None  # None means through


class FaceBoss(_Strict):
    """A round pin standing on the face: within height_mm of the envelope face only its cylinder is kept."""
    type: Literal["boss"] = "boss"
    face: Face
    a_mm: float
    b_mm: float
    diameter_mm: Mm
    height_mm: Mm


class Fillet(_Strict):
    type: Literal["fillet"] = "fillet"
    edges: EdgeSelector = "all_vertical"
    radius_mm: Mm


class Chamfer(_Strict):
    type: Literal["chamfer"] = "chamfer"
    edges: EdgeSelector = "all_vertical"
    radius_mm: Mm


FaceFeature = Annotated[FaceHole | FaceSlot | FacePocket | FaceBoss, Field(discriminator="type")]
Finish = Annotated[Fillet | Chamfer, Field(discriminator="type")]


class Views(_Strict):
    front: Outline
    top: Outline
    right: Outline


class MvAbstain(_Strict):
    stage: Literal["label", "outline", "dimensions", "complete", "build", "slice", "verify"]
    reason: str
    remedy: str
    partial: dict | None = None


class MultiViewSpec(_Strict):
    version: Literal["mv1"] = "mv1"
    envelope: Envelope
    views: Views
    features: list[FaceFeature] = Field(default=[], max_length=MAX_FEATURES)
    finishes: list[Finish] = Field(default=[], max_length=MAX_FINISHES)
    provenance: dict[str, MvProvenance] = Field(max_length=10 * (MAX_FEATURES + MAX_FINISHES) + 10)
    snapped: list[str] = Field(default=[], max_length=10 * (MAX_FEATURES + MAX_FINISHES) + 10)
    warnings: list[str] = Field(default=[], max_length=500)
    confidence: float = Field(ge=0, le=1)

    @model_validator(mode="after")
    def _check(self) -> MultiViewSpec:
        missing = [p for p in numeric_field_paths(self) if p not in self.provenance]
        if missing:
            raise ValueError(f"missing provenance for {missing}")
        for axis in ("x", "y", "z"):
            if self.provenance[f"envelope.{axis}_mm"] not in TRUSTED:
                raise ValueError(f"envelope.{axis}_mm needs a trusted source: written, measured or typed")
        tol = POINT_TOLERANCE_MM
        for face in CANONICAL_FACES:
            a_len, b_len = face_size(face, self.envelope)
            outline = getattr(self.views, face)
            for a, b in outline.outer + [p for loop in outline.inner for p in loop]:
                if not (-tol <= a <= a_len + tol and -tol <= b <= b_len + tol):
                    raise ValueError(f"views.{face} point ({a:g}, {b:g}) lies outside the envelope")
        return self


def numeric_names(model: BaseModel) -> list[str]:
    """Numeric fields of a feature or finish that need provenance. None (through) needs none."""
    return [k for k, v in model.model_dump().items()
            if k != "type" and not isinstance(v, bool) and isinstance(v, (int, float))]


def feature_path(i: int, name: str) -> str:
    """The provenance (and edit) key of a feature's value: the one spelling every module uses."""
    return f"features[{i}].{name}"


def finish_path(i: int, name: str) -> str:
    return f"finishes[{i}].{name}"


def numeric_field_paths(spec: MultiViewSpec) -> list[str]:
    paths = ["envelope.x_mm", "envelope.y_mm", "envelope.z_mm"]
    paths += [f"views.{face}.outer" for face in CANONICAL_FACES]
    for i, f in enumerate(spec.features):
        paths += [feature_path(i, n) for n in numeric_names(f)]
    for i, f in enumerate(spec.finishes):
        paths += [finish_path(i, n) for n in numeric_names(f)]
    return paths


def face_size(face: str, env: Envelope) -> tuple[float, float]:
    """Width and height of the envelope rectangle seen from `face`, in that face's (a, b) frame."""
    a_axis, b_axis, _ = FACE_AXES[face]
    return env.length(a_axis), env.length(b_axis)


def to_global(face: str, a: float, b: float, env: Envelope) -> dict[str, float]:
    """The two global coordinates a face-frame point fixes (spec section 3 table)."""
    x, z = env.x_mm, env.z_mm
    return {
        "front": {"x": a, "y": b}, "back": {"x": x - a, "y": b},
        "top": {"x": a, "z": z - b}, "bottom": {"x": a, "z": b},
        "right": {"z": z - a, "y": b}, "left": {"z": a, "y": b},
    }[face]


def to_canonical(face: str, points, env: Envelope) -> list[Point]:
    """Mirror rule: points in `face`'s frame -> the frame of its canonical face (front, top or right)."""
    a_len, b_len = face_size(face, env)
    if face in ("back", "left"):
        return [(float(a_len - a), float(b)) for a, b in points]
    if face == "bottom":
        return [(float(a), float(b_len - b)) for a, b in points]
    return [(float(a), float(b)) for a, b in points]
