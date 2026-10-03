"""Single image -> mesh with TripoSR, on the local GPU or through the Hugging Face Space. Spec section 6.3.
The mesh is only rendered to silhouettes by complete.py; it is never exported."""
from __future__ import annotations

import concurrent.futures as cf
import inspect
import logging
import os
import sys
import tempfile
import types
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np

from s2c import obs
from s2c.multiview.raster import Mesh

log = logging.getLogger(__name__)
LOCAL_TIMEOUT_S = 60
SPACE_TIMEOUT_S = 90
DEFAULT_SPACE = "stabilityai/TripoSR"
REPO_DIR = Path(os.environ.get("TRIPOSR_DIR", Path(__file__).resolve().parents[2] / "vendor" / "TripoSR"))
_pool = cf.ThreadPoolExecutor(max_workers=1)


def _install_mcubes_shim(force: bool = False) -> None:
    """TripoSR imports torchmcubes, which needs a CUDA compiler on Windows; scikit-image does the same on CPU."""
    if not force:
        try:
            import torchmcubes  # noqa: F401
            return
        except ImportError:
            pass
    import torch
    from skimage.measure import marching_cubes as sk_marching_cubes

    def marching_cubes(vol, thresh):
        verts, faces, _, _ = sk_marching_cubes(vol.detach().cpu().numpy(), level=thresh)
        verts = np.ascontiguousarray(verts[:, ::-1])  # torchmcubes returns (x, y, z) for a volume indexed [z][y][x]
        return (torch.from_numpy(verts.copy()).float().to(vol.device),
                torch.from_numpy(faces.astype(np.int64)).to(vol.device))

    module = types.ModuleType("torchmcubes")
    module.marching_cubes = marching_cubes
    sys.modules["torchmcubes"] = module


@lru_cache(maxsize=1)
def _rembg_session():
    import rembg
    return rembg.new_session()


def _prepare(image_bgr: np.ndarray):
    """Background removed, part centred at 85 percent of a square, composited on grey, as TripoSR expects."""
    import rembg
    from PIL import Image
    rgba = np.asarray(rembg.remove(Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)),
                                   session=_rembg_session()))
    ys, xs = np.nonzero(rgba[:, :, 3] > 127)
    if len(xs) == 0:
        raise RuntimeError("background removal left nothing")
    crop = rgba[ys.min(): ys.max() + 1, xs.min(): xs.max() + 1]
    side = int(max(crop.shape[:2]) / 0.85)
    canvas = np.zeros((side, side, 4), np.uint8)
    y0, x0 = (side - crop.shape[0]) // 2, (side - crop.shape[1]) // 2
    canvas[y0: y0 + crop.shape[0], x0: x0 + crop.shape[1]] = crop
    rgb, alpha = canvas[:, :, :3] / 255.0, canvas[:, :, 3:4] / 255.0
    return Image.fromarray(((rgb * alpha + (1 - alpha) * 0.5) * 255).astype(np.uint8))


def _to_mesh(tm) -> Mesh:
    return Mesh(np.asarray(tm.vertices, np.float64), np.asarray(tm.faces, np.int64))


@lru_cache(maxsize=1)
def _local_model():
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("no CUDA device")
    if not REPO_DIR.exists():
        raise RuntimeError(f"TripoSR not found at {REPO_DIR}; run scripts/setup_triposr.ps1")
    if str(REPO_DIR) not in sys.path:  # appended: the checkout never shadows an installed package
        sys.path.append(str(REPO_DIR))
    _install_mcubes_shim()
    from tsr.system import TSR
    model = TSR.from_pretrained("stabilityai/TripoSR", config_name="config.yaml", weight_name="model.ckpt")
    model.renderer.set_chunk_size(4096)  # smaller chunks fit a 6 GB GPU
    return model.to("cuda")


def _local_run(image_bgr: np.ndarray) -> Mesh:
    import torch
    model = _local_model()
    image = _prepare(image_bgr)
    with torch.no_grad():
        codes = model([image], device="cuda")
        kwargs = {"resolution": 256}
        if "has_vertex_color" in inspect.signature(model.extract_mesh).parameters:
            kwargs["has_vertex_color"] = False
        meshes = model.extract_mesh(codes, **kwargs)
    return _to_mesh(meshes[0])


def local_triposr(image_bgr: np.ndarray) -> Mesh:
    _local_model()  # loads outside the timeout: the first call downloads the weights
    return _pool.submit(_local_run, image_bgr).result(timeout=LOCAL_TIMEOUT_S)


def space_triposr(image_bgr: np.ndarray) -> Mesh:
    import trimesh
    from gradio_client import Client, handle_file

    def run() -> Mesh:
        client = Client(os.environ.get("TRIPOSR_SPACE", DEFAULT_SPACE))
        with tempfile.TemporaryDirectory() as d:
            src = Path(d) / "input.png"
            _prepare(image_bgr).save(src)
            processed = client.predict(handle_file(str(src)), False, 0.85, api_name="/preprocess")
            result = client.predict(handle_file(processed), 256, api_name="/generate")
            path = result[0] if isinstance(result, (list, tuple)) else result
            return _to_mesh(trimesh.load(path, force="mesh"))

    with cf.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(run).result(timeout=SPACE_TIMEOUT_S)


def default_provider():
    """Local GPU first, the Hugging Face Space second. Raises when both fail; complete.py then assumes."""
    def provide(image_bgr: np.ndarray) -> Mesh:
        errors = []
        for name, fn in (("local", local_triposr), ("space", space_triposr)):
            try:
                return fn(image_bgr)
            except Exception as e:  # noqa: BLE001 - a failed provider falls back, never breaks the request
                if name == "local":
                    obs.fallback("triposr_local", e)  # the Space is tried next
                else:
                    log.warning("TripoSR %s failed: %s", name, e)  # the caller counts the whole fallback
                errors.append(f"{name}: {e}")
        raise RuntimeError("; ".join(errors))
    return provide
