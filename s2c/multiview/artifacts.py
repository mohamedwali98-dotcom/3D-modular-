"""Built parts and their files, cached by content. Spec 2026-09-23-studio section 5.
A part is keyed by its geometry (never by warnings or provenance); files by mesh quality or print settings. Files live
under ROOT/<key>/ and are deleted an hour after their last use. Nothing the user did not change is recomputed."""
from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import shutil
import tempfile
import threading
import time
import zipfile
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import cadquery as cq
import numpy as np

from s2c import obs
from s2c.config import data_path
from s2c.multiview import spec as S
from s2c.multiview.blend import write_blend
from s2c.multiview.build import BuildError, build, volume
from s2c.multiview.drawing import DRAWING_FILES, DRAWING_FORMATS, write_drawings
from s2c.multiview.exporters import FILE_NAMES, MESH_FORMATS, export_mesh_formats, write_glb
from s2c.multiview.finish import apply_geometry
from s2c.multiview.print_settings import slicer_flags
from s2c.multiview.raster import face_mask, normalize_mask, solid_mesh
from s2c.multiview.settings import FORMATS, GeometrySettings, MeshSettings, PrintSettings, settings_hash
from s2c.multiview.slice import parse_gcode_stats, slice_solid

ROOT = data_path("tmp", "mv_gradio")
TTL_S = 3600
LRU_SIZE = 16
_KEY_FIELDS = {"envelope", "views", "features", "finishes"}


@dataclass
class Part:
    key: str
    spec: S.MultiViewSpec  # after the geometry settings
    solid: cq.Workplane
    volume_mm3: float
    bbox_mm: tuple[float, float, float]
    views: dict[str, np.ndarray]
    preview: Path
    folder: Path
    warnings: list[str] = field(default_factory=list)


@dataclass
class ExportResult:
    files: dict[str, Path]
    sizes: dict[str, int]
    print_time_s: float | None = None
    filament_g: float | None = None
    warnings: list[str] = field(default_factory=list)


class _Lru:
    def __init__(self, size: int):
        self.size, self._items, self._lock = size, OrderedDict(), threading.Lock()

    def get(self, key):
        with self._lock:
            value = self._items.get(key)
            if value is not None:
                self._items.move_to_end(key)
            return value

    def put(self, key, value) -> None:
        with self._lock:
            self._items[key] = value
            self._items.move_to_end(key)
            while len(self._items) > self.size:
                self._items.popitem(last=False)

    def clear(self) -> None:
        with self._lock:
            self._items.clear()


_parts = _Lru(LRU_SIZE)
_key_locks: dict[tuple[str, str], threading.Lock] = {}
_key_locks_guard = threading.Lock()


def clear_cache() -> None:
    _parts.clear()


def _lock_for(cache_key: tuple[str, str]) -> threading.Lock:
    # one lock per (root, key); a key's lock stays here after it leaves the LRU, bounded by distinct keys per process
    with _key_locks_guard:
        lock = _key_locks.get(cache_key)
        if lock is None:
            lock = _key_locks[cache_key] = threading.Lock()
        return lock


def geometry_key(spec: S.MultiViewSpec, geometry: GeometrySettings | None = None) -> str:
    body = json.dumps(spec.model_dump(mode="json", include=_KEY_FIELDS), sort_keys=True)
    return hashlib.sha256(f"{body}|{settings_hash(geometry or GeometrySettings())}".encode()).hexdigest()[:20]


def _touch(folder: Path) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    now = time.time()
    os.utime(folder, (now, now))


def build_part(
    spec: S.MultiViewSpec, geometry: GeometrySettings | None = None, root: Path = ROOT
) -> Part | S.MvAbstain:
    geometry = geometry or GeometrySettings()
    key = geometry_key(spec, geometry)
    folder = Path(root) / key
    final, warnings = apply_geometry(spec, geometry)
    cache_key = (str(root), key)
    cached = _parts.get(cache_key)
    if cached is not None and cached.preview.exists():  # the sweep may have removed the files
        _touch(folder)
        # same solid and files; this request's provenance and warnings, which the manifest records
        return dataclasses.replace(cached, spec=final, warnings=warnings)
    with _lock_for(cache_key):
        cached = _parts.get(cache_key)  # someone else may have built it while we waited for the lock
        if cached is not None and cached.preview.exists():
            _touch(folder)
            return dataclasses.replace(cached, spec=final, warnings=warnings)
        try:
            solid = build(final)
        except BuildError as e:
            return S.MvAbstain(stage="build", reason=e.reason, remedy=e.remedy)
        bb = solid.val().BoundingBox()
        mesh = solid_mesh(solid)
        views = {f: normalize_mask(face_mask(mesh, f, final.envelope)[0]) for f in S.FACES}
        _touch(folder)
        preview = folder / "preview.glb"
        tmp = folder / f"preview.{uuid4().hex[:8]}.tmp.glb"
        try:
            write_glb(solid, tmp, "normal")
            os.replace(tmp, preview)
        except Exception:
            tmp.unlink(missing_ok=True)
            raise
        part = Part(key, final, solid, volume(solid), (bb.xlen, bb.ylen, bb.zlen), views, preview, folder, warnings)
        _parts.put(cache_key, part)
        return part


def _mesh_files(part: Part, wanted: list[str], quality: str) -> dict[str, Path]:
    folder = part.folder / f"mesh-{quality}"
    missing = [f for f in wanted if not (folder / FILE_NAMES[f]).exists()]
    if missing:
        export_mesh_formats(part.solid, folder, missing, quality)
    return {f: folder / FILE_NAMES[f] for f in wanted}


def _gcode(
    part: Part, printing: PrintSettings, slicer, profile
) -> tuple[Path | None, float | None, float | None, list[str]]:
    folder = part.folder / f"print-{settings_hash(printing)}"
    gcode = folder / "part.gcode"
    if gcode.exists():
        seconds, grams = parse_gcode_stats(gcode.read_text(encoding="utf-8", errors="replace"))
        return gcode, seconds, grams, []
    res = slice_solid(part.solid, folder, profile, slicer, overrides=slicer_flags(printing),
                      scale=printing.scale_pct / 100)
    if isinstance(res, S.MvAbstain):
        return None, None, None, [f"G-code skipped: {res.remedy}"]
    return res.gcode, res.print_time_s, res.filament_g, list(res.warnings)


def export_part(part: Part, formats, mesh: MeshSettings | None = None, printing: PrintSettings | None = None,
                slicer: Path | None = None, profile: Path | None = None) -> ExportResult:
    """Two exports of the same part (e.g. two /mv requests in FastAPI's thread pool) must not both miss the
    cache and both write the same files at once, so the whole body is serialised per part folder."""
    mesh, printing = mesh or MeshSettings(), printing or PrintSettings()
    unknown = sorted(set(formats) - set(FORMATS))
    if unknown:
        raise ValueError(f"unknown formats {unknown}")
    wanted = [f for f in FORMATS if f in formats]
    with _lock_for(("export", str(part.folder))):
        need = [f for f in MESH_FORMATS if f in wanted or (f == "obj" and "blend" in wanted)]
        made = _mesh_files(part, need, mesh.quality)
        files = {f: made[f] for f in MESH_FORMATS if f in wanted}
        warnings: list[str] = []
        seconds = grams = None
        if "blend" in wanted:
            try:
                files["blend"], more = write_blend(made["obj"], part.folder / f"mesh-{mesh.quality}")
                warnings += more
            except Exception as e:  # noqa: BLE001 - one writer failing loses its own file, never the whole export
                obs.fallback("blend_writer", e)
                warnings.append("The Blender scene could not be written; the other files are ready.")
        drawings = [f for f in DRAWING_FORMATS if f in wanted]
        if drawings:
            folder = part.folder / "drawing"
            missing = [f for f in drawings if not (folder / DRAWING_FILES[f]).exists()]
            try:
                if missing:
                    write_drawings(part.solid, part.spec, folder, missing)
                files.update({f: folder / DRAWING_FILES[f] for f in drawings})
            except Exception as e:  # noqa: BLE001 - an unusual solid can defeat the 2D projection: the rest still goes
                obs.fallback("drawing_writer", e)
                names = ", ".join(f.upper() for f in drawings)
                warnings.append(f"The {names} drawing could not be written for this part; the other files are ready.")
        if "gcode" in wanted:
            gcode, seconds, grams, more = _gcode(part, printing, slicer, profile)
            warnings += more
            if gcode is not None:
                files["gcode"] = gcode
        _touch(part.folder)
        ordered = {f: files[f] for f in FORMATS if f in files}
        return ExportResult(ordered, {f: p.stat().st_size for f, p in ordered.items()}, seconds, grams, warnings)


def bundle(part: Part, result: ExportResult, settings: dict | None = None) -> Path:
    """One zip per exported file set, settings and spec, so a later export never overwrites a zip already handed out."""
    manifest = {
        "part": {"key": part.key, "volume_mm3": round(part.volume_mm3, 3),
                 "bbox_mm": [round(v, 3) for v in part.bbox_mm]},
        "files": {f: p.name for f, p in result.files.items()},
        "print": {"time_s": result.print_time_s, "filament_g": result.filament_g},
        "settings": settings or {},
        "warnings": [*part.spec.warnings, *result.warnings],
        "spec": part.spec.model_dump(mode="json"),
    }
    paths = sorted(p.relative_to(part.folder).as_posix() if p.is_relative_to(part.folder) else p.name
                   for p in result.files.values())  # mesh-fine/part.stl and mesh-normal/part.stl differ
    body = json.dumps({"paths": paths, "manifest": manifest}, sort_keys=True, default=str)
    target = part.folder / f"sketch-to-cad-{part.key[:8]}-{hashlib.sha256(body.encode()).hexdigest()[:8]}.zip"
    manifest = {"generated_at": datetime.now(UTC).isoformat(timespec="seconds"), **manifest}
    part.folder.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(suffix=".part", dir=part.folder)  # unique: concurrent exports never share it
    os.close(fd)
    tmp = Path(name)
    try:
        with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as z:
            for path in result.files.values():
                z.write(path, arcname=path.name)
            z.writestr("manifest.json", json.dumps(manifest, indent=2))
        os.replace(tmp, target)
    finally:
        tmp.unlink(missing_ok=True)
    return target


def sweep(root: Path = ROOT, ttl_s: float = TTL_S) -> None:
    root = Path(root)
    if not root.exists():
        return
    now = time.time()
    for d in root.iterdir():
        if d.is_dir() and now - d.stat().st_mtime > ttl_s:
            shutil.rmtree(d, ignore_errors=True)
