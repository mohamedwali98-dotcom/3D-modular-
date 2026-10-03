"""Qwen-Image draws a face nobody photographed; a voxel check keeps it only if it never removes material that a
photographed face shows. Spec 2026-09-23 section 7. Holes come only from photographed faces, where they are
editable features, so circles on a drawn face are ignored."""
from __future__ import annotations

import logging

import cv2
import numpy as np

from s2c import obs
from s2c.multiview.outline import PixelOutline, extract, foreground, resize_long_side, to_face_mm
from s2c.multiview.qwen_image import ImageGen, ImageGenError
from s2c.multiview.raster import iou, polygon_mask
from s2c.multiview.spec import CANONICAL_FACES, Envelope, MvAbstain, Outline, face_size

log = logging.getLogger(__name__)
VOXELS = 128
GATE_IOU = 0.95
TRIES = 2
SEED = 7
CONFIDENCE = 0.6
FAILED = ("qwen-image-failed",)  # cache key: a call failed, so this request stops asking
VIEWER = {
    "front": "the front, looking straight along the depth axis, with the top of the part at the top of the image",
    "top": "directly above, looking down, with the front edge of the part at the bottom of the image",
    "right": "the right side, looking straight at it, with the top of the part at the top of the image",
}
FACE_PROMPT = ("Images 1 to {n} show one mechanical part: {which}. Draw the orthographic {face} view of the same "
               "part, as seen from {viewer}. Solid black silhouette on a pure white background. Through-holes are "
               "white. No text, no dimension lines, no shading, no perspective, no background objects.")


def face_prompt(ref_faces: list[str], face: str) -> str:
    which = ", ".join(f"image {k} is the {f} view" for k, f in enumerate(ref_faces, 1))
    return FACE_PROMPT.format(n=len(ref_faces), which=which, face=face, viewer=VIEWER[face])


def outline_from_image(img: np.ndarray, face: str, env: Envelope) -> Outline | None:
    """A drawn silhouette, stretched onto the face's envelope rectangle."""
    found = extract(resize_long_side(img))
    if isinstance(found, MvAbstain):
        return None
    a_len, b_len = face_size(face, env)
    _, _, w, h = found.bbox
    sa, sb = a_len / max(w - 1, 1), b_len / max(h - 1, 1)

    def mm(points):
        return [(min(max(a, 0.0), a_len), min(max(b, 0.0), b_len)) for a, b in to_face_mm(points, found.bbox, sa, sb)]

    outer = mm(found.outer)
    if len(set(outer)) < 3:
        return None
    return Outline(outer=outer, inner=[], source="inferred", confidence=CONFIDENCE)


def grid_mask(outline: Outline, a_len: float, b_len: float, n: int = VOXELS) -> np.ndarray:
    """Boolean [ib, ia] samples of an outline on an n x n grid over its face rectangle; ib grows upward."""
    def px(points):
        return [(a / a_len * n, n - b / b_len * n) for a, b in points]

    img = polygon_mask(px(outline.outer), [px(loop) for loop in outline.inner], (n, n))
    return img[::-1] > 127


def carve(front: np.ndarray, top: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Voxels [ix, iy, iz] inside all three extruded outlines. Face frames as in the multi-view spec section 3:
    front (a=X, b=Y), top (a=X, b=depth-Z), right (a=depth-Z, b=Y)."""
    return front.T[:, :, None] & top[::-1, :].T[:, None, :] & right[:, ::-1][None, :, :]


def project(vox: np.ndarray, face: str) -> np.ndarray:
    """Silhouette of the voxels seen from a canonical face, as [ib, ia] like grid_mask."""
    if face == "front":
        return vox.any(axis=2).T
    if face == "top":
        return vox.any(axis=1).T[::-1]
    return vox.any(axis=0)[:, ::-1]


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    union = np.logical_or(a, b).sum()
    return 1.0 if union == 0 else float(np.logical_and(a, b).sum() / union)


def consistent(outlines: dict[str, Outline], env: Envelope, observed: list[str], n: int = VOXELS) -> bool:
    """True when carving with every outline still shows each observed face as it was photographed."""
    masks = {f: grid_mask(outlines[f], *face_size(f, env), n) if f in outlines else np.ones((n, n), bool)
             for f in CANONICAL_FACES}
    vox = carve(masks["front"], masks["top"], masks["right"])
    return all(_iou(project(vox, f), masks[f]) >= GATE_IOU for f in observed)


def _drawn(gen: ImageGen | None, refs, face: str, cache: dict, seed: int) -> np.ndarray | None:
    """The generated image for (face, seed), from the cache when possible. After one failed call the request
    stops asking: a slow or broken Space would otherwise cost a full timeout per face and per seed."""
    key = (face, seed)
    if key not in cache:
        if gen is None or not refs or cache.get(FAILED):
            return None
        try:
            cache[key] = gen([img for _, img in refs], face_prompt([f for f, _ in refs], face), seed, "mv_face")
        except ImageGenError as e:
            obs.fallback("qwen_image", e)
            cache[key], cache[FAILED] = None, True
    return cache[key]


def qwen_face(outlines: dict[str, Outline], env: Envelope, face: str, observed: list[str], refs,
              gen: ImageGen | None, cache: dict, seed: int = SEED, attempts: int = TRIES) -> Outline | None:
    """The first drawing of `face` that passes the voxel check, trying seeds seed .. seed + attempts - 1."""
    for attempt in range(attempts):
        img = _drawn(gen, refs, face, cache, seed + attempt)
        if img is None and cache.get(FAILED):
            return None  # the call failed: another seed would only wait again
        candidate = None if img is None else outline_from_image(img, face, env)
        if candidate is not None and consistent({**outlines, face: candidate}, env, observed):
            return candidate
    return None


RESCUE_PROMPT = ("Redraw this hand sketch of a mechanical part face as a clean solid black silhouette on a pure "
                 "white background. Keep the proportions and position exactly. Remove all text, numbers, arrows "
                 "and dimension lines. Holes are white.")
RESCUE_IOU = 0.85
RESCUE_ASPECT = 0.05
RESCUE_PENALTY = 0.8
BRIDGE_FRACTION = 0.03  # of the long side: enough to close a gap in a pen line


def raw_region(image_bgr: np.ndarray) -> np.ndarray:
    """The pen strokes with small gaps bridged, largest region filled: what the sketch outline encloses."""
    ink = foreground(image_bgr)
    k = max(3, int(BRIDGE_FRACTION * max(ink.shape)) | 1)
    closed = cv2.morphologyEx(ink, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    region = np.zeros_like(ink)
    if contours:
        cv2.drawContours(region, [max(contours, key=cv2.contourArea)], -1, 255, -1)
    return region


def rescue_sketch(image_bgr: np.ndarray, gen: ImageGen, seed: int = SEED) -> PixelOutline | None:
    """Qwen-Image redraws a sketch whose outline is not closed. Kept only if it matches the raw strokes."""
    h, w = image_bgr.shape[:2]
    try:
        drawn = gen([image_bgr], RESCUE_PROMPT, seed, "mv_rescue")
    except ImageGenError as e:
        obs.fallback("qwen_image_rescue", e)
        return None
    outline = extract(cv2.resize(drawn, (w, h), interpolation=cv2.INTER_AREA))
    if isinstance(outline, MvAbstain):
        return None
    raw = raw_region(image_bgr)
    if iou(polygon_mask(outline.outer, outline.inner, (h, w)), raw) < RESCUE_IOU:
        return None
    _, _, rw, rh = cv2.boundingRect(raw)
    _, _, cw, ch = outline.bbox
    if rh == 0 or ch == 0 or abs((cw / ch) / (rw / rh) - 1) > RESCUE_ASPECT:
        return None
    return outline
