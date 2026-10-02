"""Wires the stages together. Real modules when present, fakes otherwise."""
from __future__ import annotations

import importlib
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from s2c.logdir import log_file
from s2c.merge import merge
from s2c.partspec.models import Abstain, Annotations, Measurements, PartSpec, SourceInput, Topology
from s2c.silhouette import input_silhouette, iou, normalize_mask
from s2c.vision.client import VLMClient
from s2c.vision.topology import topology_from_image

log = logging.getLogger(__name__)
IOU_GREEN = 0.85


@dataclass
class AnalyzeResult:
    input_mask: np.ndarray
    partspec: PartSpec | None = None
    abstain: Abstain | None = None
    topology: Topology | None = None
    annotations: Annotations | None = None
    measurements: Measurements | None = None


@dataclass
class BuildResult:
    step_path: Path
    stl_path: Path
    iou: float
    views: dict[str, np.ndarray]
    warnings: list[str] = field(default_factory=list)


@dataclass
class Pipeline:
    topology: Callable[[bytes, SourceInput], Topology | Abstain]
    annotations: Callable[[np.ndarray, Topology], Annotations]
    measure: Callable[[np.ndarray], Measurements | Abstain]
    build: Callable[[PartSpec], object]
    export: Callable[[object, Path], tuple[Path, Path]]
    silhouettes: Callable[..., dict[str, np.ndarray]]

    def analyze(self, image_bytes: bytes, input_kind: SourceInput,
                user_values: dict[str, float] | None = None) -> AnalyzeResult:
        image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            return AnalyzeResult(input_mask=np.zeros((512, 512), np.uint8),
                                 abstain=Abstain(stage="vision", reason="unreadable_image",
                                                 remedy="The image could not be decoded. Use JPEG or PNG."))
        try:
            mask = input_silhouette(image)
        except ValueError:
            return AnalyzeResult(input_mask=np.zeros((512, 512), np.uint8),
                                 abstain=Abstain(stage="vision", reason="no_outline",
                                                 remedy="No part outline found. Use a plain background and good light."))
        res = AnalyzeResult(input_mask=mask)
        topo = self.topology(image_bytes, input_kind)
        if isinstance(topo, Abstain):
            res.abstain = topo
            return res
        res.topology = topo
        if input_kind == "photo":
            m = self.measure(image)
            if isinstance(m, Abstain):
                res.abstain = m
                return res
            res.measurements = m
        else:
            res.annotations = self.annotations(image, topo)
        out = merge(topo, source_input=input_kind, annotations=res.annotations,
                    measurements=res.measurements, user_values=user_values)
        if isinstance(out, Abstain):
            res.abstain = out
        else:
            res.partspec = out
        return res

    def remerge(self, topology: Topology, *, source_input: SourceInput, annotations=None,
                measurements=None, user_values=None) -> PartSpec | Abstain:
        return merge(topology, source_input=source_input, annotations=annotations,
                     measurements=measurements, user_values=user_values)

    def build_and_verify(self, spec: PartSpec, input_mask: np.ndarray, out_dir: Path) -> BuildResult | Abstain:
        try:
            solid = self.build(spec)
        except Exception as e:  # BuildError from the geometry owner carries reason and remedy
            # Logged with the traceback for our own diagnostics; the Abstain handed back to the
            # user never carries the exception's message, only the fixed, user-safe strings below.
            log.exception("build stage failed")
            reason = getattr(e, "reason", "build_failed")
            remedy = getattr(e, "remedy", "The part could not be built. Check the dimensions.")
            return Abstain(stage="build", reason=reason, remedy=remedy)
        step, stl = self.export(solid, Path(out_dir))
        views = {k: normalize_mask(v) for k, v in self.silhouettes(solid, px=512).items()}
        score = iou(input_mask, views["front"])
        warnings = list(spec.warnings)
        if score < IOU_GREEN:
            warnings.append(f"Low confidence (IoU {score:.2f}), check the dimensions.")
        return BuildResult(step_path=step, stl_path=stl, iou=score, views=views, warnings=warnings)


def _load(module: str, attr: str, fallback):
    try:
        return getattr(importlib.import_module(module), attr)
    except (ImportError, AttributeError):
        log.warning("using fake for %s.%s", module, attr)
        return fallback


def fake_pipeline() -> Pipeline:
    from s2c.fakes import builder, metrology, ocr, views, vision
    client = VLMClient(chat=vision.chat, model="fake", log_path=log_file("vlm_fake.jsonl"))
    return Pipeline(
        topology=lambda b, k: topology_from_image(b, client, k),
        annotations=ocr.read_annotations, measure=metrology.measure,
        build=builder.build, export=builder.export, silhouettes=views.silhouettes,
    )


def default_pipeline(client: VLMClient | None = None) -> Pipeline:
    from s2c.fakes import builder, metrology, ocr, views, vision
    if client is None:
        import os
        client = VLMClient.from_env() if os.environ.get("VLM_API_KEY") else VLMClient(
            chat=vision.chat, model="fake", log_path=log_file("vlm_fake.jsonl"))
    return Pipeline(
        topology=lambda b, k: topology_from_image(b, client, k),
        annotations=_load("s2c.ocr", "read_annotations", ocr.read_annotations),
        measure=_load("s2c.metrology", "measure", metrology.measure),
        build=_load("s2c.builder", "build", builder.build),
        export=_load("s2c.builder", "export", builder.export),
        silhouettes=_load("s2c.views", "silhouettes", views.silhouettes),
    )
