"""The multi-view path end to end. Model calls happen in observe(); fuse() calls TripoSR at most once per
request and caches the mesh; build() never calls a model. Spec sections 6 and 7."""
from __future__ import annotations

import copy
import logging
import os
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from pydantic import ValidationError

from s2c.multiview import qwen_reader
from s2c.multiview import spec as S
from s2c.multiview.build import BuildError, export
from s2c.multiview.build import build as build_solid
from s2c.multiview.complete import MeshProvider, complete
from s2c.multiview.depth import DepthProvider, apply_depth, solaria_depth
from s2c.multiview.fuse import (
    Observation,
    assemble,
    attach_label,
    canonical_outlines,
    classify_drawn_circles,
    features_from,
    fuse_envelope,
    outline_kinds,
)
from s2c.multiview.label import Chat, MvLabel, env_chat, hint_label, label_image
from s2c.multiview.merge_views import merge_same_face
from s2c.multiview.ocr import BatchReader, Reader, link, read_values
from s2c.multiview.outline import PixelOutline, _stroke, extract, ink_mask, resize_long_side
from s2c.multiview.qwen_faces import RESCUE_PENALTY, SEED, TRIES, rescue_sketch
from s2c.multiview.qwen_image import MAX_REFS, ImageGen, default_gen
from s2c.multiview.qwen_reader import qwen_batch_reader
from s2c.multiview.raster import Mesh, face_mask, iou, normalize_mask, polygon_mask, solid_mesh
from s2c.multiview.reference import find_reference
from s2c.multiview.relief import bosses, pocket_provenance, relief
from s2c.multiview.settings import AiSettings, GeometrySettings
from s2c.multiview.slice import slice_solid
from s2c.multiview.turned import WARNING as TURNED_WARNING
from s2c.multiview.turned import complete_turned, turned_axis
from s2c.reading import ReadingService, as_reader, read_timeout_s

log = logging.getLogger(__name__)
IOU_GREEN = 0.85

Progress = Callable[[str, dict], None]


def _emit(progress: Progress | None, **data) -> None:
    """Report one pipeline stage event; a no-op when no callback was given. Never wraps the callback in
    try/except -- an exception it raises (e.g. JobCancelled) must propagate out of observe()/fuse()."""
    progress and progress("stage", data)


@dataclass
class ImageInput:
    data: bytes
    face: str | None = None
    kind: str | None = None
    mm_per_px: float | None = None  # a known scale in the image's own pixels: a drawing sheet's dimensions
    scale_confirmed: bool = True    # False: sizes from that scale are suggestions the user confirms
    numbers: bool = True            # False: its numbers were read already (a view cropped from a sheet)
    line_art: bool = False          # True: known to be a line drawing (a sketch view drawn again from its strokes)


@dataclass
class Observed:
    """Everything that needed a model call, cached per request so merge never repeats it."""
    observations: list[Observation]
    images: list[np.ndarray]
    masks: dict[str, np.ndarray]
    labels: list[MvLabel]
    warnings: list[str] = field(default_factory=list)
    mesh: Mesh | None = None
    qwen_cache: dict = field(default_factory=dict)            # (face, seed) -> drawn image, or None after a failure
    filled_by: dict[str, str] = field(default_factory=dict)   # canonical face -> who filled it


@dataclass
class BuildResult:
    step: Path
    stl: Path
    print_stl: Path | None
    gcode: Path | None
    print_time_s: float | None
    filament_g: float | None
    views: dict[str, np.ndarray]
    iou: dict[str, float]
    warnings: list[str]


def forget_images(observed: Observed, res) -> None:
    """Raw images are kept only until a spec exists; after that, merge needs only the silhouettes and the
    cached mesh. This keeps the rule that images live for the request, plus silhouettes for one hour."""
    if not isinstance(res, S.MvAbstain):
        observed.images.clear()


def input_mask(outline: PixelOutline, edges=()) -> np.ndarray:
    """The input silhouette of one image: outer outline filled, openings and circles cut out, normalised.
    Circles read as edges (their indices in `edges`) are not holes, so they stay filled."""
    holes = list(outline.inner)
    holes += [cv2.ellipse2Poly((round(c.cx), round(c.cy)), (round(c.d / 2), round(c.d / 2)), 0, 0, 360, 5)
              for i, c in enumerate(outline.circles) if i not in edges]
    return normalize_mask(polygon_mask(outline.outer, holes, outline.shape))


def line_width(bgr: np.ndarray, outline: PixelOutline, mask_out=()) -> float:
    """A line drawing's line width in px, measured as the outline stage does: the ink inside the outline, its area
    over half its edge length."""
    filled = np.zeros(outline.shape, np.uint8)
    cv2.fillPoly(filled, [np.asarray(outline.outer, np.int32).reshape(-1, 1, 2)], 255)
    return _stroke(cv2.bitwise_and(ink_mask(bgr, mask_out), filled))


class MvPipeline:
    def __init__(self, chat: Chat | None = None, reader: Reader | None = None,
                 mesh_provider: MeshProvider | None = None, slicer: Path | None = None, profile: Path | None = None,
                 batch_reader: BatchReader | None = None, image_gen: ImageGen | None = None,
                 depth: DepthProvider | None = None):
        self.chat, self.reader, self.mesh_provider = chat, reader, mesh_provider
        self.slicer, self.profile = slicer, profile
        self.batch_reader = batch_reader
        self.image_gen = image_gen
        self.depth = depth
        self.seed, self.attempts = SEED, TRIES
        self.draw_faces = self.rescue_enabled = True

    def reading(self) -> ReadingService | None:
        """The readers in trust order: the batch (Qwen-VL) reader first, it keeps the ⌀ and R signs; then TrOCR."""
        base, model = os.environ.get("VLM_BASE_URL"), os.environ.get("VLM_MODEL")
        qwen_key = f"qwen:{base}:{model}" if base and model and self.batch_reader is not None else None
        readers = [r for r in (
            as_reader(self.batch_reader, "qwen", calibrated=False, batch=True, timeout_s=read_timeout_s(),
                      cache_key=qwen_key),
            as_reader(self.reader, "trocr", calibrated=True, batch=False)) if r is not None]
        return ReadingService(readers) if readers else None

    def configured(self, ai: AiSettings) -> MvPipeline:
        """A copy for one request with the user's AI switches, seed and attempts; the shared pipeline never changes."""
        pipe = copy.copy(self)
        if not ai.use_reader:
            pipe.batch_reader = None
        if not ai.use_triposr:
            pipe.mesh_provider = None
        if not ai.use_solaria:
            pipe.depth = None
        pipe.draw_faces, pipe.rescue_enabled = ai.use_qwen_image, ai.use_rescue
        pipe.seed, pipe.attempts = ai.seed, ai.attempts
        return pipe

    def _label(self, item: ImageInput) -> MvLabel | S.MvAbstain:
        if self.chat is not None:
            return label_image(item.data, self.chat, item.face, item.kind)
        if item.face:
            return hint_label(item.face, item.kind or "sketch")
        return S.MvAbstain(stage="label", reason="face_unknown", remedy="Tell us which face this photo shows.")

    def observe(self, images: list[ImageInput], reference: str | None = None,
                progress: Progress | None = None) -> Observed | S.MvAbstain:
        observed = Observed([], [], {}, [])
        reads = self.reader is not None or self.batch_reader is not None
        if not reads:
            observed.warnings.append("OCR unavailable: enter the dimensions by hand")
        service = self.reading() if reads else None
        excluded: list[tuple] = []
        for i, item in enumerate(images):
            try:
                bgr = cv2.imdecode(np.frombuffer(item.data, np.uint8), cv2.IMREAD_COLOR)
            except cv2.error:  # past OPENCV_IO_MAX_IMAGE_PIXELS (s2c/__init__.py): refused, never decoded
                bgr = None
            if bgr is None:
                return S.MvAbstain(stage="outline", reason="bad_image",
                                   remedy="The file is not an image. Upload a JPEG or PNG.")
            given = item.mm_per_px * max(bgr.shape[:2]) if item.mm_per_px else None  # mm across the long side
            bgr = resize_long_side(bgr)
            _emit(progress, key="label", state="running", index=i)
            label = self._label(item)
            if isinstance(label, S.MvAbstain):
                return label
            height, width = bgr.shape[:2]
            _emit(progress, key="label", state="done", index=i, face=label.face, kind=label.input_kind,
                  confidence=label.confidence, width=width, height=height)
            mm_per_px = given / max(bgr.shape[:2]) if given else None
            mask_out = ()
            if reference and label.input_kind == "photo":
                ref = find_reference(bgr, reference)
                if isinstance(ref, S.MvAbstain):
                    return ref
                bgr, mm_per_px = ref.image, ref.mm_per_px
                mask_out = (ref.bbox,) if ref.bbox else ()
            _emit(progress, key="outline", state="running", index=i)
            outline, rescued = self._outline(bgr, mask_out, label.input_kind, item.line_art)
            if isinstance(outline, S.MvAbstain):
                return outline
            _emit(progress, key="outline", state="done", index=i, outline=outline.outer.astype(int).tolist(),
                  circles=[{"cx": c.cx, "cy": c.cy, "d": c.d} for c in outline.circles])
            values = []
            if not reads or label.input_kind == "photo" or not item.numbers:
                _emit(progress, key="read", state="skipped", index=i)
            else:
                _emit(progress, key="read", state="running", index=i)
                values = link(read_values(bgr, outline, service), outline)
                _emit(progress, key="read", state="done", index=i,
                      reads=[{"text": v.reading.text, "value_mm": v.reading.value_mm, "kind": v.reading.kind,
                              "bbox": v.reading.bbox, "confidence": v.reading.confidence} for v in values])
            obs = Observation(face=label.face, kind=label.input_kind, outline=outline, values=values,
                              mm_per_px=mm_per_px, scale_confirmed=item.scale_confirmed or not given,
                              confidence=label.confidence * (RESCUE_PENALTY if rescued else 1.0),
                              stroke=line_width(bgr, outline, mask_out) if outline.line_art and not rescued else 0.0)
            attach_label(obs, label)
            if rescued:
                observed.warnings.append(f"{label.face}: sketch cleaned by Qwen-Image, check it")
            observed.observations.append(obs)
            observed.images.append(bgr)
            observed.labels.append(label)
            excluded.append(mask_out)
        if self.depth is not None:
            observed.warnings += self._depths(observed, excluded)
        return self._merge(observed)

    def _outline(self, bgr: np.ndarray, mask_out, kind: str,
                 line_art: bool = False) -> tuple[PixelOutline | S.MvAbstain, bool]:
        """The outline, and whether Qwen-Image had to redraw the sketch (spec 2026-09-23 section 8)."""
        outline = extract(bgr, mask_out, drawing=kind == "drawing", line_art=line_art)
        if (isinstance(outline, S.MvAbstain) and outline.reason == "no_outline" and kind != "photo"
                and self.rescue_enabled and self.image_gen is not None):
            fixed = rescue_sketch(bgr, self.image_gen, self.seed)
            if fixed is not None:
                return fixed, True
        return outline, False

    def _depths(self, observed: Observed, excluded: list[tuple]) -> list[str]:
        """Solaria once per face, on its most confident photo that shows a hole (spec 2026-09-23 section 9)."""
        best: dict[str, int] = {}
        for k, o in enumerate(observed.observations):
            if o.kind == "photo" and o.outline.circles and (
                    o.face not in best or o.confidence > observed.observations[best[o.face]].confidence):
                best[o.face] = k
        warnings = []
        for face, k in best.items():
            try:
                depth = self.depth(observed.images[k])
                warnings += apply_depth(observed.observations[k], depth, excluded[k])
            except Exception as e:  # noqa: BLE001 - a failed provider falls back, never breaks the request
                log.warning("Solaria failed on %s: %s", face, e)
                warnings.append(f"{face}: depth unavailable")
        return warnings

    @staticmethod
    def _merge(observed: Observed) -> Observed:
        """One observation per face: several photos of a face are merged (spec 2026-09-23 section 5)."""
        merged, images, warnings = merge_same_face(observed.observations, observed.images)
        observed.observations, observed.images = merged, images
        observed.warnings += warnings
        observed.masks = {o.face: input_mask(o.outline) for o in merged}
        return observed

    def fuse(self, observed: Observed, user_values: dict | None = None, accepted=(), rejected=(),
             geometry: GeometrySettings | None = None,
             progress: Progress | None = None) -> S.MultiViewSpec | S.MvAbstain:
        geometry = geometry or GeometrySettings()
        env_result = fuse_envelope(observed.observations, user_values)
        if isinstance(env_result, S.MvAbstain):
            return env_result
        env, env_prov, warnings = env_result
        outlines, more = canonical_outlines(observed.observations, env)
        warnings = observed.warnings + warnings + more
        if all(o.line_art for o in observed.observations):  # drawing-sheet spec 3.5, for line drawings only
            turned_views, more = complete_turned({f: ol for f, (ol, _) in outlines.items()}, env)
            outlines.update({f: (ol, "inferred") for f, ol in turned_views.items() if f not in outlines})
            warnings += more
        edges, more = classify_drawn_circles(observed.observations, env)
        warnings += more
        for k, o in enumerate(observed.observations):
            if o.line_art:  # on every fuse, so a circle read as a hole again gets its hole back
                observed.masks[o.face] = input_mask(o.outline, edges.get(k, ()))
        best = max(range(len(observed.observations)), key=lambda i: observed.observations[i].confidence)
        target = observed.observations[best]
        image = observed.images[best] if best < len(observed.images) else None  # None once forget_images dropped them
        pairs = sorted(zip(observed.observations, observed.images), key=lambda p: -p[0].confidence)
        refs = [(o.face, img) for o, img in pairs][:MAX_REFS]
        _emit(progress, key="draw", state="running")
        full, more, observed.mesh = complete({f: ol for f, (ol, _) in outlines.items()}, env, target.face,
                                             observed.masks[target.face], image, self.mesh_provider,
                                             observed.mesh, tuple(rejected),
                                             gen=self.image_gen if self.draw_faces else None, refs=refs,
                                             qwen_cache=observed.qwen_cache, filled_by=observed.filled_by,
                                             seed=self.seed, attempts=self.attempts)
        warnings += more
        _emit(progress, key="draw", state="done", filled_by=dict(observed.filled_by))
        with_prov = {f: (ol, outlines[f][1] if f in outlines else ("inferred" if ol.source == "inferred" else "default"))
                     for f, ol in full.items()}
        feats, feat_prov = features_from(observed.observations, env, edges)
        if len(observed.observations) >= 2 and all(o.line_art for o in observed.observations):
            views = {f: ol for f, (ol, _) in with_prov.items()}
            pins, more = bosses(observed.observations, edges, views, env)
            pockets, notes = relief(observed.observations, views, env)
            drawn = all(views[f].source in ("observed", "mirrored") for f in S.CANONICAL_FACES if f in views)
            feat_prov.update(pocket_provenance(pins + pockets, len(feats), inferred=not drawn))
            feats += pins + pockets
            warnings += more + notes
        _emit(progress, key="fuse", state="running")
        try:
            spec = assemble(env, env_prov, with_prov, feats, feat_prov, warnings, user_values, accepted,
                            snap_values=geometry.snap, clearance=geometry.clearance,
                            kinds=outline_kinds(observed.observations))
        except ValidationError as e:
            log.warning("spec rejected: %s", e)
            return S.MvAbstain(stage="dimensions", reason="invalid_value",
                               remedy="A value is out of range. Check the numbers you entered.")
        try:
            axis = turned_axis(spec)
        except Exception as e:  # noqa: BLE001 - the turn is a refinement; its check must never break a fuse
            log.warning("turned check failed: %s", e)
            axis = None
        note = axis and TURNED_WARNING.format(axis=axis)
        if note and note not in spec.warnings:
            spec = spec.model_copy(update={"warnings": [*spec.warnings, note]})
        _emit(progress, key="fuse", state="done")
        return spec

    def build(self, spec: S.MultiViewSpec, out_dir: Path, masks: dict | None = None) -> BuildResult | S.MvAbstain:
        notes: list[str] = []
        try:
            solid = build_solid(spec, notes)
        except BuildError as e:
            return S.MvAbstain(stage="build", reason=e.reason, remedy=e.remedy)
        step, stl = export(solid, out_dir)
        sliced = slice_solid(solid, out_dir, self.profile, self.slicer)
        if isinstance(sliced, S.MvAbstain):
            return sliced
        mesh = solid_mesh(solid)
        views = {f: normalize_mask(face_mask(mesh, f, spec.envelope)[0]) for f in S.FACES}
        scores = {f: round(iou(views[f], m), 3) for f, m in (masks or {}).items()}
        warnings = list(spec.warnings) + notes + sliced.warnings
        warnings += [f"Low confidence on {f}, check the dimensions." for f, s in scores.items() if s < IOU_GREEN]
        return BuildResult(step, stl, sliced.print_stl, sliced.gcode, sliced.print_time_s, sliced.filament_g,
                           views, scores, warnings)


def _warm_readers(reader: object | None, read_chat: Chat | None) -> None:
    """TrOCR first, then Qwen: Ollama only sees TrOCR's GPU share once it has actually claimed it, so it
    can decide how many layers to keep on the GPU with that share already gone."""
    if reader is not None:
        warm = getattr(reader, "warm", None)
        if warm is not None:
            t0 = time.perf_counter()
            try:
                warm()
                log.info("reader %s ready in %.1f s", reader.name, time.perf_counter() - t0)
            except Exception as e:  # noqa: BLE001 - a failed warm-up only means a slower first read
                log.warning("reader %s warm-up failed: %s", reader.name, e)
    if read_chat is not None:
        qwen_reader.warm_chat(read_chat)


def default_pipeline() -> MvPipeline:
    """Qwen-VL, Qwen-Image and Solaria from the environment; TrOCR and TripoSR when the ai extra is installed."""
    reader = provider = None
    try:
        from s2c.reading.trocr import TrocrReader
        reader = TrocrReader()
    except Exception as e:  # noqa: BLE001 - transformers missing: fall back without TrOCR
        log.warning("TrOCR unavailable: %s", e)
    try:
        from s2c.multiview.hf3d import default_provider
        provider = default_provider()
    except Exception as e:  # noqa: BLE001 - a failed provider falls back, never breaks the request
        log.warning("TripoSR unavailable: %s", e)
    read_chat = env_chat(stage="mv_read")
    if reader is not None or read_chat is not None:
        threading.Thread(target=_warm_readers, args=(reader, read_chat), name="reader-warmup",
                          daemon=True).start()
    space = os.environ.get("SOLARIA_SPACE")
    image_gen = default_gen()
    if image_gen is None:
        log.warning("Qwen-Image unavailable: set QWEN_IMAGE_SPACE, or QWEN_IMAGE_BACKEND=dashscope with its "
                    "settings; missing faces fall back to TripoSR or an assumed rectangle")
    if not space:
        log.warning("Solaria unavailable: set SOLARIA_SPACE; every hole stays through unless the user makes it blind")
    return MvPipeline(chat=env_chat(), reader=reader, mesh_provider=provider,
                      batch_reader=qwen_batch_reader(read_chat) if read_chat else None, image_gen=image_gen,
                      depth=solaria_depth(space, os.environ.get("HF_TOKEN")) if space else None)
