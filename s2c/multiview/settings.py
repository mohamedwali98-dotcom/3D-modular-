"""Studio settings shared by the UI, the API and the command line. Spec 2026-09-23-studio section 4.
Every model has defaults and pydantic ranges, so StudioSettings() is always valid."""
from __future__ import annotations

import hashlib
import json
import math
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

Quality = Literal["draft", "normal", "fine"]
Clearance = Literal["fine", "medium", "coarse"]
Material = Literal["PLA", "PETG", "ABS", "ASA", "TPU"]
Supports = Literal["off", "buildplate", "everywhere"]
InfillPattern = Literal["grid", "gyroid", "rectilinear", "honeycomb", "cubic", "lightning"]
FinishKind = Literal["none", "fillet", "chamfer"]
EdgeChoice = Literal["all_vertical", "top", "bottom", "all"]

MESH_TOLERANCES: dict[str, tuple[float, float]] = {"draft": (0.1, 0.5), "normal": (0.02, 0.2), "fine": (0.005, 0.1)}
# material -> (nozzle, first-layer nozzle, bed, first-layer bed), degrees C
MATERIALS: dict[str, tuple[int, int, int, int]] = {
    "PLA": (210, 215, 60, 60), "PETG": (240, 240, 80, 80), "ABS": (250, 255, 100, 100),
    "ASA": (255, 260, 100, 100), "TPU": (225, 225, 50, 50),
}
DENSITIES: dict[str, float] = {"PLA": 1.24, "PETG": 1.27, "ABS": 1.04, "ASA": 1.07, "TPU": 1.21}  # g/cm3
FILAMENT_MM = 1.75
NOZZLES = (0.2, 0.4, 0.6, 0.8)
EDGE_LABELS = {"all_vertical": "Outline corners", "top": "Front-face edges", "bottom": "Back-face edges",
               "all": "All edges"}
FORMATS: dict[str, str] = {
    "stl": "STL mesh", "step": "STEP (CAD)", "3mf": "3MF (slicers)", "obj": "OBJ mesh", "glb": "GLB (web, AR)",
    "ply": "PLY mesh", "brep": "BREP (OpenCascade)", "blend": "Blender", "dxf": "DXF drawing",
    "svg": "SVG drawing", "pdf": "PDF drawing", "gcode": "G-code",
}


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


class AiSettings(_Model):
    """The AI helpers. Qwen-Image (and its sketch rescue), TripoSR and Solaria send the user's images to hosted
    services, so they are off until the user turns them on. The reader uses the vision model the operator set."""
    use_reader: bool = True
    use_qwen_image: bool = False
    use_rescue: bool = False
    use_triposr: bool = False
    use_solaria: bool = False
    seed: int = Field(7, ge=0, le=2**31 - 1)
    randomize_seed: bool = False
    attempts: int = Field(2, ge=1, le=4)


class GeometrySettings(_Model):
    snap: bool = True
    clearance: Clearance = "medium"
    finish: FinishKind = "none"
    finish_mm: float = Field(1.0, ge=0.2, le=10.0)
    finish_edges: EdgeChoice = "all_vertical"


class MeshSettings(_Model):
    quality: Quality = "normal"

    @property
    def tolerances(self) -> tuple[float, float]:
        return MESH_TOLERANCES[self.quality]


class PrintSettings(_Model):
    material: Material = "PLA"
    nozzle_mm: float = 0.4
    layer_mm: float = Field(0.2, ge=0.05, le=0.32)
    infill_pct: int = Field(20, ge=0, le=100)
    infill_pattern: InfillPattern = "grid"
    perimeters: int = Field(3, ge=1, le=8)
    supports: Supports = "buildplate"
    brim_mm: float = Field(0.0, ge=0.0, le=10.0)
    scale_pct: float = Field(100.0, ge=50.0, le=200.0)

    @field_validator("nozzle_mm")
    @classmethod
    def _known_nozzle(cls, v: float) -> float:
        if not any(abs(v - n) < 1e-9 for n in NOZZLES):
            raise ValueError(f"nozzle must be one of {NOZZLES} mm")
        return v

    @model_validator(mode="after")
    def _layer_fits_nozzle(self) -> PrintSettings:
        if self.layer_mm > 0.75 * self.nozzle_mm + 1e-9:
            raise ValueError(f"layer height {self.layer_mm:g} mm is more than 0.75 x the {self.nozzle_mm:g} mm nozzle")
        return self


class ExportSettings(_Model):
    formats: list[str] = Field(default_factory=lambda: ["stl", "step", "3mf", "gcode"])

    @field_validator("formats")
    @classmethod
    def _known_formats(cls, v: list[str]) -> list[str]:
        unknown = sorted(set(v) - set(FORMATS))
        if unknown:
            raise ValueError(f"unknown formats {unknown}; choose from {list(FORMATS)}")
        return [f for f in FORMATS if f in v]


class StudioSettings(_Model):
    ai: AiSettings = Field(default_factory=AiSettings)
    geometry: GeometrySettings = Field(default_factory=GeometrySettings)
    mesh: MeshSettings = Field(default_factory=MeshSettings)
    printing: PrintSettings = Field(default_factory=PrintSettings)
    export: ExportSettings = Field(default_factory=ExportSettings)


def filament_metres(grams: float, material: str) -> float:
    """Length of 1.75 mm filament for a mass: 12.4 g of PLA is 10 cm3, about 4.16 m."""
    return grams / DENSITIES[material] / (math.pi * (FILAMENT_MM / 2) ** 2)  # cm3 / mm2 = 1000 mm3 / mm2 = m


def settings_hash(model: BaseModel) -> str:
    body = json.dumps(model.model_dump(mode="json"), sort_keys=True)
    return hashlib.sha256(body.encode()).hexdigest()[:16]
