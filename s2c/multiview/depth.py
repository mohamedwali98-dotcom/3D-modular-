"""Solaria (Marigold V2 depth on a Hugging Face Space) -> through or blind holes, and blind depth, on photos only.
Spec 2026-09-23 section 9. Marigold depth is right only up to scale and shift, and Solaria rescales it again, so
only ratios of depth differences are used: they do not depend on scale, shift or which way depth grows.
The point cloud is never exported."""
from __future__ import annotations

import logging
import tempfile
import time
from collections.abc import Callable
from pathlib import Path

import cv2
import numpy as np

from s2c.multiview.fuse import Observation
from s2c.multiview.outline import PixelOutline, resize_long_side
from s2c.multiview.qwen_image import log_call
from s2c.multiview.raster import polygon_mask

log = logging.getLogger(__name__)
DepthProvider = Callable[[np.ndarray], np.ndarray]  # BGR image -> depth at the same size, NaN where unknown
STRIDE = 4
SEND_LONG_SIDE = 1024
TIMEOUT_S = 180
THROUGH, MARK, FLAT = 0.9, 0.1, 0.05
NEAR, FAR = 0.05, 0.15  # table band, as fractions of the part's bounding-box diagonal
HTTP_TIMEOUT_S = 30


def read_depth(ply_path, width: int, height: int, stride: int = STRIDE) -> np.ndarray:
    """Invert Solaria's point cloud: x = (col - W/2) / W, y = -(row - H/2) / H, z = depth, on a stride grid."""
    lines = Path(ply_path).read_text(encoding="utf-8").splitlines()
    body = lines[lines.index("end_header") + 1:]
    out = np.full((-(-height // stride), -(-width // stride)), np.nan, np.float32)
    if not body:
        return out
    pts = np.loadtxt(body, ndmin=2, usecols=(0, 1, 2))
    cols = np.rint((pts[:, 0] + 0.5) * width).astype(int) // stride
    rows = np.rint((0.5 - pts[:, 1]) * height).astype(int) // stride
    ok = (rows >= 0) & (rows < out.shape[0]) & (cols >= 0) & (cols < out.shape[1])
    out[rows[ok], cols[ok]] = pts[ok, 2]
    return out


def solaria_depth(space: str, token: str | None = None, client_factory=None, timeout_s: float = TIMEOUT_S,
                  log_path=None) -> DepthProvider:
    def run(img: np.ndarray) -> np.ndarray:
        from gradio_client import Client, handle_file
        h, w = img.shape[:2]
        client = (client_factory or Client)(space, token=token, httpx_kwargs={"timeout": HTTP_TIMEOUT_S})
        with tempfile.TemporaryDirectory() as d:
            src, mask = Path(d) / "image.png", Path(d) / "mask.png"
            cv2.imwrite(str(src), img)
            cv2.imwrite(str(mask), np.full((h, w), 255, np.uint8))  # all white: depth for the whole frame
            job = client.submit(handle_file(str(src)), handle_file(str(mask)), api_name="/gerar_3d")
            try:
                _, ply, status = job.result(timeout=timeout_s)
            except TimeoutError:
                job.cancel()  # stop the Space job instead of leaving it to burn quota
                raise
        ply = ply.get("path") if isinstance(ply, dict) else ply
        if not ply:
            raise RuntimeError(f"Solaria returned no point cloud: {status}")
        return read_depth(ply, w, h)

    def provide(image_bgr: np.ndarray) -> np.ndarray:
        img = resize_long_side(image_bgr, SEND_LONG_SIDE) if max(image_bgr.shape[:2]) > SEND_LONG_SIDE else image_bgr
        t0, ok = time.perf_counter(), False
        try:
            small = run(img)
            ok = True
        finally:
            log_call(log_path, "hf-space", space, "mv_depth", t0, ok)
        h, w = image_bgr.shape[:2]
        return cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)

    return provide


def _median(depth: np.ndarray, where: np.ndarray) -> float | None:
    values = depth[where]
    values = values[np.isfinite(values)]
    return float(np.median(values)) if values.size else None


def _crop_bounds(c, h: int, w: int) -> tuple[int, int, int, int]:
    pad = int(np.ceil(1.5 * c.d / 2)) + 1
    y0, y1 = max(round(c.cy) - pad, 0), min(round(c.cy) + pad + 1, h)
    x0, x1 = max(round(c.cx) - pad, 0), min(round(c.cx) + pad + 1, w)
    return y0, y1, x0, x1


def hole_depths(depth: np.ndarray, outline: PixelOutline, exclude=()) -> tuple[dict[int, float | None], list[str]]:
    """Per circle: None when it goes through, else its depth as a fraction of the axis length. A mark, or a part
    too flat to measure, gets no entry and a warning. The part lies flat, so the table is one axis length down.
    Ring and hole stats are read from each circle's own bounding-box crop, not a full-frame array per circle."""
    h, w = depth.shape
    filled = polygon_mask(outline.outer, (), (h, w))
    part = polygon_mask(outline.outer, outline.inner, (h, w))
    for c in outline.circles:
        cv2.circle(part, (round(c.cx), round(c.cy)), round(c.d / 2), 0, -1)
    diag = float(np.hypot(outline.bbox[2], outline.bbox[3]))
    away = cv2.distanceTransform(255 - filled, cv2.DIST_L2, 5)
    table = (away > NEAR * diag) & (away <= FAR * diag)
    for x, y, bw, bh in exclude:
        table[max(y, 0): y + bh, max(x, 0): x + bw] = False
    finite = depth[np.isfinite(depth)]
    span = float(np.percentile(finite, 98) - np.percentile(finite, 2)) if finite.size else 0.0
    d_table = _median(depth, table)
    found, warnings = {}, []
    for i, c in enumerate(outline.circles):
        r = c.d / 2
        y0, y1, x0, x1 = _crop_bounds(c, h, w)
        d_crop, part_crop = depth[y0:y1, x0:x1], part[y0:y1, x0:x1] > 127
        yy, xx = np.mgrid[y0:y1, x0:x1]
        d2 = (xx - c.cx) ** 2 + (yy - c.cy) ** 2
        d_face = _median(d_crop, part_crop & (d2 >= r * r) & (d2 <= (1.5 * r) ** 2))
        d_hole = _median(d_crop, d2 <= (0.7 * r) ** 2)
        if d_table is None or d_face is None or d_hole is None or abs(d_table - d_face) <= FLAT * span:
            warnings.append("depth: part too flat to measure, check hole depths")
            continue
        ratio = (d_hole - d_face) / (d_table - d_face)
        if ratio >= THROUGH:
            found[i] = None
        elif ratio >= MARK:
            found[i] = float(ratio)
        else:
            warnings.append(f"hole {i + 1} looks like a mark, not a hole")
    return found, list(dict.fromkeys(warnings))


def apply_depth(obs: Observation, depth: np.ndarray, exclude=()) -> list[str]:
    """Write Solaria's verdicts into the observation; only Solaria (or the user) makes a hole blind."""
    found, warnings = hole_depths(depth, obs.outline, exclude)
    for i, ratio in found.items():
        obs.depth_from_image.add(i)
        obs.blind[i] = ratio is not None
        obs.depth_estimates.pop(i, None)
        if ratio is None:
            obs.depth_ratio.pop(i, None)
        else:
            obs.depth_ratio[i] = ratio
    return [f"{obs.face}: {w}" for w in warnings]
