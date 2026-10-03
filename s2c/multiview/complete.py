"""Fill the canonical faces nobody photographed: drawn by Qwen-Image, else predicted from a TripoSR mesh, else
assumed rectangular. The mirror rule already ran in fuse. Spec 2026-09-22 section 6.3, spec 2026-09-23 section 7.
Neither the drawn image nor the mesh is exported; both are only read back as outlines."""
from __future__ import annotations

import itertools
import logging
from collections.abc import Callable

import numpy as np

from s2c import obs
from s2c.multiview.qwen_faces import SEED, TRIES, qwen_face
from s2c.multiview.qwen_image import ImageGen
from s2c.multiview.raster import Mesh, face_mask, iou, mask_to_mm, normalize_mask
from s2c.multiview.spec import CANONICAL_FACES, Envelope, Outline, face_size

log = logging.getLogger(__name__)
MeshProvider = Callable[[np.ndarray], Mesh]
MIN_ORIENTATION_IOU = 0.6
SEARCH_PX = 128


def rotations() -> list[np.ndarray]:
    """The 24 axis-aligned rotations: signed permutation matrices with determinant +1."""
    out = []
    for perm in itertools.permutations(range(3)):
        for signs in itertools.product((1.0, -1.0), repeat=3):
            m = np.zeros((3, 3))
            for row, (col, sign) in enumerate(zip(perm, signs)):
                m[row, col] = sign
            if round(np.linalg.det(m)) == 1:
                out.append(m)
    return out


def _at_origin(v: np.ndarray) -> np.ndarray:
    return v - v.min(axis=0)


def _own_envelope(v: np.ndarray) -> Envelope:
    span = np.maximum(v.max(axis=0) - v.min(axis=0), 1e-6)
    return Envelope(x_mm=float(span[0]), y_mm=float(span[1]), z_mm=float(span[2]))


def orient(mesh: Mesh, face: str, target_mask: np.ndarray) -> tuple[np.ndarray, float]:
    """Rotation that makes the mesh, seen from `face` at its own proportions, look most like the target."""
    target = normalize_mask(target_mask, SEARCH_PX)
    best_r, best = np.eye(3), -1.0
    for r in rotations():
        v = _at_origin(mesh.vertices @ r.T)
        mask, _ = face_mask(Mesh(v, mesh.faces), face, _own_envelope(v), SEARCH_PX)
        score = iou(normalize_mask(mask, SEARCH_PX), target)
        if score > best:
            best_r, best = r, score
    return best_r, best


def fit_to_envelope(mesh: Mesh, r: np.ndarray, env: Envelope) -> Mesh:
    """Rotate, then scale each axis so the bounding box equals the trusted envelope."""
    v = _at_origin(mesh.vertices @ r.T)
    span = np.maximum(v.max(axis=0), 1e-9)
    return Mesh(v / span * np.array([env.x_mm, env.y_mm, env.z_mm]), mesh.faces)


def _clamp(pts, a_len, b_len):
    return [(min(max(a, 0.0), a_len), min(max(b, 0.0), b_len)) for a, b in pts]


def predicted_outline(mesh: Mesh, face: str, env: Envelope, confidence: float) -> Outline:
    mask, s = face_mask(mesh, face, env, 512)
    outer, inner = mask_to_mm(mask, s, 512)
    a_len, b_len = face_size(face, env)
    return Outline(outer=_clamp(outer, a_len, b_len), inner=[_clamp(loop, a_len, b_len) for loop in inner],
                   source="inferred", confidence=round(float(confidence), 3))


def assumed_outline(face: str, env: Envelope) -> Outline:
    a, b = face_size(face, env)
    return Outline(outer=[(0.0, 0.0), (a, 0.0), (a, b), (0.0, b)], source="assumed", confidence=0.3)


def complete(outlines: dict[str, Outline], env: Envelope, target_face: str, target_mask: np.ndarray | None,
             image: np.ndarray | None, provider: MeshProvider | None, mesh: Mesh | None = None, rejected=(),
             gen: ImageGen | None = None, refs=(), qwen_cache: dict | None = None,
             filled_by: dict | None = None, seed: int = SEED, attempts: int = TRIES
             ) -> tuple[dict[str, Outline], list[str], Mesh | None]:
    """All three canonical outlines, the warnings, and the mesh so the caller can cache it.
    filled_by, when given, receives who filled each face: observed, mirrored, qwen-image, triposr or assumed."""
    result, warnings = dict(outlines), []
    filled_by = {} if filled_by is None else filled_by
    qwen_cache = {} if qwen_cache is None else qwen_cache
    filled_by.update({f: ol.source for f, ol in outlines.items()})
    observed = list(outlines)
    missing = [f for f in CANONICAL_FACES if f not in outlines]
    tried_qwen = []
    for face in missing:
        if face in rejected:
            continue
        drawn = qwen_face(result, env, face, observed, list(refs), gen, qwen_cache, seed=seed, attempts=attempts)
        if drawn is not None:
            result[face], filled_by[face] = drawn, "qwen-image"
        elif any(v is not None for k, v in qwen_cache.items() if k[0] == face):
            tried_qwen.append(face)  # Qwen returned an image for this face and it was rejected, not just never called
    wanted = [f for f in missing if f not in rejected and f not in result]
    if wanted and mesh is None and provider is not None and image is not None:
        try:
            mesh = provider(image)
        except Exception as e:  # noqa: BLE001 - a failed provider falls back, never breaks the request
            obs.fallback("triposr", e)
            warnings.append("3D predictor unavailable")
    fitted, score = None, 0.0
    if wanted and mesh is not None:
        r, score = orient(mesh, target_face, target_mask)
        if score >= MIN_ORIENTATION_IOU:
            fitted = fit_to_envelope(mesh, r, env)
        else:
            warnings.append("predicted view unreliable")
    for face in missing:
        if face in result:
            continue
        if fitted is not None and face in wanted:
            try:
                result[face], filled_by[face] = predicted_outline(fitted, face, env, score), "triposr"
            except ValueError:
                log.warning("predicted %s view was empty", face)
        if face not in result:
            result[face], filled_by[face] = assumed_outline(face, env), "assumed"
            warnings.append(f"assumed rectangular {face}, check it")
        if face in tried_qwen:
            warnings.append(f"{face}: Qwen-Image view rejected, {filled_by[face]} used")
    return result, warnings, mesh
