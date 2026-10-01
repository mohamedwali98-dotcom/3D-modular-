"""Score the multi-view pipeline against reverse-engineered parts with known STLs and clean renders.
The true envelope is passed as user values (CLAUDE.md rule 2): this measures modeling accuracy given size,
not size recovery. Never uses trimesh.Trimesh.contains (rtree is not installed); voxel IoU uses voxelize+fill.

Sheet mode (drawing-sheet spec section 6) feeds the same part as one first-angle line-art sheet of its front, top and
right renders, through the Studio's split, name and crop, to measure what a drawing sheet costs against per-face
uploads."""
from __future__ import annotations

import json
import re
import statistics
import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeoutError
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import trimesh
from PIL import Image

from s2c.multiview import spec as S
from s2c.multiview.artifacts import build_part
from s2c.multiview.pipeline import ImageInput, MvPipeline
from s2c.multiview.raster import Mesh, face_mask, iou, normalize_mask, solid_mesh
from s2c.multiview.sheet import AUTO, SKIP, Box, Naming, Sheet, crop_views, split_sheet
from s2c.multiview.sheet_read import choose_naming

MAX_TRIANGLES = 200_000
ROW_FIELDS = ("frame_iou", "vol_true", "vol_built", "vol_err", "voxel_iou", "view_iou", "features")

SHEET_LONG = 400        # px of the envelope's longest edge on a composed sheet
SHEET_GAP = 60          # px between views
SHEET_LINE = 2          # px line width
SHEET_LABEL = 0.07      # label height, share of the median view's long side
CANNY = (50, 150)
# (column, row) of each view around the front in first-angle projection (ISO 5456-2): the top view below the front,
# the view from the right on its left. Row +1 is below the front.
SHEET_GRID = {"front": (0, 0), "top": (0, 1), "right": (-1, 0)}
SHEET_LABELS = {"front": "FRONT VIEW", "top": "TOP VIEW", "right": "RIGHT SIDE VIEW"}
FONT = cv2.FONT_HERSHEY_SIMPLEX


@dataclass
class RefPart:
    name: str          # "<category>/<dir>"
    category: str
    mesh: Mesh         # our frame, min corner at 0
    envelope: S.Envelope
    renders: dict[str, Path]
    watertight: bool
    triangles: int


def _to_ours(v: np.ndarray) -> np.ndarray:
    """Dataset is Z-up, ours is Y-up. The proper rotation only; the reflection flips the volume sign."""
    return np.stack([v[:, 0], v[:, 2], -v[:, 1]], axis=1)


def load_part(part_dir: Path) -> RefPart:
    part_dir = Path(part_dir)
    category = part_dir.parent.name
    meta = json.loads((part_dir / "metadata.json").read_text())
    stl = trimesh.load(part_dir / "model.stl", force="mesh", process=False)
    v = _to_ours(np.asarray(stl.vertices, dtype=np.float64))
    v = v - v.min(axis=0)
    mesh = Mesh(v, np.asarray(stl.faces))
    env = S.Envelope(x_mm=float(v[:, 0].max()), y_mm=float(v[:, 1].max()), z_mm=float(v[:, 2].max()))
    renders = {f: part_dir / f"{f}.png" for f in S.FACES}
    return RefPart(f"{category}/{part_dir.name}", category, mesh, env, renders, bool(meta["watertight"]),
                   len(mesh.faces))


def render_mask(png: Path) -> np.ndarray:
    """255 where a pixel differs from white by more than 30 in the sum of |RGB - 255|."""
    return _part_pixels(np.array(Image.open(png).convert("RGB")))


def _part_pixels(rgb: np.ndarray) -> np.ndarray:
    return np.where(np.abs(rgb.astype(int) - 255).sum(-1) > 30, 255, 0).astype(np.uint8)


def _cells(mesh: trimesh.Trimesh, pitch: float) -> set[tuple[int, int, int]]:
    vox = mesh.voxelized(pitch).fill()
    return set(map(tuple, np.round(vox.points / pitch).astype(int)))


def voxel_iou(a: trimesh.Trimesh, b: trimesh.Trimesh, n: int = 64) -> float:
    pitch = float(max(b.extents)) / n
    ca, cb = _cells(a, pitch), _cells(b, pitch)
    union = len(ca | cb)
    return len(ca & cb) / union if union else 0.0


def bare_row(part: str, category: str, result: str, secs: float = 0.0) -> dict:
    """A results.jsonl row for a part that never became a RefPart (a bad metadata.json or STL, a crash)."""
    row = {"part": part, "category": category, "result": result, "secs": round(secs, 1)}
    row.update(dict.fromkeys(ROW_FIELDS))
    return row


def _row(ref: RefPart, result: str, secs: float = 0.0, **extra) -> dict:
    row = bare_row(ref.name, ref.category, result, secs)
    row.update(extra)
    return row


def _frame_iou(ref: RefPart, env: S.Envelope) -> float:
    def one(face: str) -> float:
        ours = normalize_mask(face_mask(ref.mesh, face, env)[0])
        theirs = normalize_mask(render_mask(ref.renders[face]))
        return iou(ours, theirs)

    return float(np.mean([one(f) for f in S.FACES]))


def line_view(png: Path, size: tuple[int, int]) -> np.ndarray:
    """A render as a line drawing (ink 255 on 0): cropped to the part, resized to `size` (w, h) so every view of a
    sheet shares one scale, then Canny edges SHEET_LINE px wide. Edges at shading steps are the visible edges."""
    rgb = np.array(Image.open(png).convert("RGB"))
    ys, xs = np.nonzero(_part_pixels(rgb))
    if len(xs) == 0:
        raise ValueError(f"empty render {png}")
    part = rgb[ys.min(): ys.max() + 1, xs.min(): xs.max() + 1]
    interp = cv2.INTER_AREA if size[0] * size[1] < part.shape[0] * part.shape[1] else cv2.INTER_CUBIC
    gray = cv2.cvtColor(cv2.resize(part, size, interpolation=interp), cv2.COLOR_RGB2GRAY)
    pad = 2 * SHEET_LINE
    gray = cv2.copyMakeBorder(gray, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=255)
    edges = cv2.dilate(cv2.Canny(gray, *CANNY), np.ones((SHEET_LINE, SHEET_LINE), np.uint8))
    ys, xs = np.nonzero(edges)
    return edges[ys.min(): ys.max() + 1, xs.min(): xs.max() + 1]


def _text_ink(text: str, height: int) -> np.ndarray:
    """`text` in SHEET_LINE px strokes, cropped to its ink, about `height` px tall."""
    def draw(scale: float) -> np.ndarray:
        (tw, th), base = cv2.getTextSize(text, FONT, scale, SHEET_LINE)
        pad = 2 * SHEET_LINE + 2
        canvas = np.zeros((th + base + 2 * pad, tw + 2 * pad), np.uint8)
        cv2.putText(canvas, text, (pad, th + pad), FONT, scale, 255, SHEET_LINE, cv2.LINE_8)
        on = canvas >= 128  # OpenCV 5 anti-aliases text; keep the ink crisp
        ys, xs = np.nonzero(on)
        return np.where(on[ys.min(): ys.max() + 1, xs.min(): xs.max() + 1], 255, 0).astype(np.uint8)

    return draw(height / draw(1.0).shape[0])


def _place(views: dict[str, np.ndarray], texts: dict[str, np.ndarray]):
    """Top-left corner of each view in its SHEET_GRID cell, rows and columns centred on each other, with room for a
    label and a gap of its height under each view; and the sheet size (h, w)."""
    cols = sorted({SHEET_GRID[f][0] for f in views})
    rows = sorted({SHEET_GRID[f][1] for f in views})
    col_w = {c: max(max(v.shape[1], texts[f].shape[1] if f in texts else 0)
                    for f, v in views.items() if SHEET_GRID[f][0] == c) for c in cols}
    row_h = {r: max(v.shape[0] for f, v in views.items() if SHEET_GRID[f][1] == r) for r in rows}
    row_label = {r: max((2 * t.shape[0] for f, t in texts.items() if SHEET_GRID[f][1] == r), default=0) for r in rows}
    col_x, x = {}, SHEET_GAP
    for c in cols:
        col_x[c], x = x, x + col_w[c] + SHEET_GAP
    row_y, y = {}, SHEET_GAP
    for r in rows:
        row_y[r], y = y, y + row_h[r] + row_label[r] + SHEET_GAP
    spots = {}
    for f, v in views.items():
        c, r = SHEET_GRID[f]
        spots[f] = (col_x[c] + (col_w[c] - v.shape[1]) // 2, row_y[r] + (row_h[r] - v.shape[0]) // 2)
    return spots, (y, x)


def compose_sheet(views: dict[str, np.ndarray], labels: bool) -> tuple[np.ndarray, dict[str, Box]]:
    """A white BGR first-angle sheet of line-drawn views (face -> ink layer, one shared scale), and the ink box of
    each view. A label sits one label height under its view, farther than split_sheet's dilation reaches, so it stays
    a label and never joins its view."""
    label_h = max(12, round(SHEET_LABEL * float(np.median([max(v.shape) for v in views.values()])))) if labels else 0
    for _ in range(4):
        texts = {f: _text_ink(SHEET_LABELS[f], label_h) for f in views} if labels else {}
        spots, (h, w) = _place(views, texts)
        short = max(3, round(0.008 * max(h, w))) + 3 - min((t.shape[0] for t in texts.values()), default=10 ** 6)
        if short <= 0:
            break
        label_h += short
    ink = np.zeros((h, w), np.uint8)
    truth: dict[str, Box] = {}
    for f, v in views.items():
        x, y = spots[f]
        vh, vw = v.shape
        ink[y: y + vh, x: x + vw] |= v
        truth[f] = (x, y, vw, vh)
        if f in texts:
            t = texts[f]
            tx, ty = x + (vw - t.shape[1]) // 2, y + vh + t.shape[0]
            ink[ty: ty + t.shape[0], tx: tx + t.shape[1]] |= t
    return cv2.cvtColor(255 - ink, cv2.COLOR_GRAY2BGR), truth


def _labelled(name: str) -> bool:
    """Labels on even-numbered parts only, so half the sheets are named by layout alone (spec section 6)."""
    m = re.search(r"(\d+)$", name)
    return bool(m) and int(m.group(1)) % 2 == 0


def part_sheet(ref: RefPart) -> tuple[np.ndarray, dict[str, Box]]:
    """The part's front, top and right renders as one first-angle line-art sheet, the envelope's longest edge
    SHEET_LONG px, and the ink box of each view."""
    env = ref.envelope
    s = SHEET_LONG / max(env.x_mm, env.y_mm, env.z_mm)
    views = {}
    for f in SHEET_GRID:
        w, h = S.face_size(f, env)
        views[f] = line_view(ref.renders[f], (max(2, round(s * w)), max(2, round(s * h))))
    return compose_sheet(views, _labelled(ref.name))


def _named_ok(sheet: Sheet, naming: Naming, truth: dict[str, Box]) -> bool:
    """Each true view found once and named with its true face; views named "skip" (marks) are ignored."""
    if naming.drawing < 0:
        return False
    found = []
    for view, face in zip(sheet.drawings[naming.drawing].views, naming.faces, strict=True):
        if face == SKIP:
            continue
        cx, cy = view.box[0] + view.box[2] / 2, view.box[1] + view.box[3] / 2
        true = next((f for f, (x, y, w, h) in truth.items() if x <= cx <= x + w and y <= cy <= y + h), None)
        if true != face:
            return False
        found.append(face)
    return sorted(found) == sorted(truth)


def sheet_inputs(ref: RefPart, root: Path | None = None) -> tuple[list[ImageInput], bool, str | None]:
    """The part's sheet split, named by layout with the projection on Auto (no label reader, so the drawing decides
    and a tie keeps first-angle) and cropped as the Studio does (spec 3.1, 3.2; sheet-reading spec 2.4).

    Returns the crops as kind "drawing" with their named faces; whether every view was named with its true face; and
    "split" (the part drawing has under 3 views) or "naming" (under 3 named views, or one left "auto") when the
    sheet cannot go on. The sheet is saved under `root`/sheets for inspection."""
    image, truth = part_sheet(ref)
    if root is not None:
        folder = Path(root) / "sheets"
        folder.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(folder / f"{ref.name.replace('/', '__')}.png"), image)
    sheet = split_sheet(image)
    naming = choose_naming(sheet, image, "auto")
    ok = _named_ok(sheet, naming, truth)
    if naming.drawing < 0 or len(sheet.drawings[naming.drawing].views) < len(SHEET_GRID):
        return [], ok, "split"
    crops = crop_views(sheet, image, naming)
    if len(crops) < len(SHEET_GRID) or any(face == AUTO for _, face in crops):
        return [], ok, "naming"
    return [ImageInput(data, face, "drawing") for data, face in crops], ok, None


def _abstain(row: dict, stage: str, reason: str, t0: float) -> dict:
    row["result"] = f"abstain {stage}:{reason}"
    row["secs"] = round(time.time() - t0, 1)
    return row


def _score_built(ref: RefPart, pipe: MvPipeline, root: Path, t0: float, sheet: bool = False) -> dict:
    env = ref.envelope
    truth = trimesh.Trimesh(ref.mesh.vertices, ref.mesh.faces, process=False)
    true_vol = abs(float(truth.volume))
    row = _row(ref, "built", frame_iou=round(_frame_iou(ref, env), 3), vol_true=round(true_vol, 3))
    if sheet:
        imgs, row["named_ok"], failed = sheet_inputs(ref, root)
        if failed:
            return _abstain(row, "sheet", failed, t0)
    else:
        imgs = [ImageInput(ref.renders[f].read_bytes(), f, "drawing") for f in S.FACES]
    observed = pipe.observe(imgs)
    if isinstance(observed, S.MvAbstain):
        return _abstain(row, observed.stage, observed.reason, t0)
    user_values = {"envelope.x_mm": env.x_mm, "envelope.y_mm": env.y_mm, "envelope.z_mm": env.z_mm}
    fused = pipe.fuse(observed, user_values)
    if isinstance(fused, S.MvAbstain):
        return _abstain(row, fused.stage, fused.reason, t0)
    part = build_part(fused, root=Path(root))
    if isinstance(part, S.MvAbstain):
        return _abstain(row, part.stage, part.reason, t0)
    built_mesh = solid_mesh(part.solid)
    rebuilt = trimesh.Trimesh(built_mesh.vertices, built_mesh.faces, process=False)
    view_iou = {f: round(iou(normalize_mask(face_mask(built_mesh, f, env)[0]),
                             normalize_mask(face_mask(ref.mesh, f, env)[0])), 3) for f in S.FACES}
    built_vol = float(part.volume_mm3)
    row.update(vol_built=round(built_vol, 3), vol_err=round(built_vol / true_vol - 1, 4),
              voxel_iou=round(voxel_iou(rebuilt, truth), 3), view_iou=view_iou, features=len(fused.features))
    row["secs"] = round(time.time() - t0, 1)
    return row


def sheet_fields(sheet: bool) -> dict:
    """Sheet-mode rows always carry named_ok, so the summary knows the mode even when no sheet was scored."""
    return {"named_ok": None} if sheet else {}


def _run(ref: RefPart, pipe: MvPipeline, root: Path, sheet: bool = False) -> dict:
    t0 = time.time()
    try:
        return _score_built(ref, pipe, root, t0, sheet)
    except Exception as e:  # noqa: BLE001 - one part's failure must not stop the whole run (review focus 1)
        return _row(ref, f"error {type(e).__name__}", secs=time.time() - t0, **sheet_fields(sheet))


def score_part(ref: RefPart, pipe: MvPipeline, root: Path, timeout_s: float = 120, sheet: bool = False) -> dict:
    extra = sheet_fields(sheet)
    if not ref.watertight:
        return _row(ref, "skipped not_watertight", **extra)
    if ref.triangles > MAX_TRIANGLES:
        return _row(ref, "skipped too_large", **extra)
    ex = ThreadPoolExecutor(max_workers=1)
    future = ex.submit(_run, ref, pipe, root, sheet)
    try:
        result = future.result(timeout=timeout_s)
    except FutureTimeoutError:
        ex.shutdown(wait=False, cancel_futures=True)  # a real hang must free this worker, not block on shutdown
        return _row(ref, "skipped timeout", secs=timeout_s, **extra)
    ex.shutdown(wait=False)
    return result


def _reason(result: str) -> tuple[str, str | None]:
    kind, _, rest = result.partition(" ")
    return kind, rest or None


def _stats(rows: list[dict]) -> dict:
    built = [r for r in rows if r["result"] == "built"]
    abstained: dict[str, int] = {}
    skipped = errors = 0
    for r in rows:
        kind, rest = _reason(r["result"])
        if kind == "abstain":
            abstained[rest] = abstained.get(rest, 0) + 1
        elif kind == "skipped":
            skipped += 1
        elif kind == "error":
            errors += 1
    abs_vol_err = [abs(r["vol_err"]) for r in built if r.get("vol_err") is not None]
    voxel_ious = [r["voxel_iou"] for r in built if r.get("voxel_iou") is not None]
    mean_view_ious = [statistics.mean(r["view_iou"].values()) for r in built if r.get("view_iou")]
    named = [r["named_ok"] for r in rows if r.get("named_ok") is not None]
    return {
        "n": len(rows),
        "built": len(built),
        "abstained": abstained,
        "skipped": skipped,
        "errors": errors,
        "median_abs_vol_err": statistics.median(abs_vol_err) if abs_vol_err else None,
        "median_voxel_iou": statistics.median(voxel_ious) if voxel_ious else None,
        "median_view_iou": statistics.median(mean_view_ious) if mean_view_ious else None,
        "within_5pct_vol_err": (sum(1 for e in abs_vol_err if e <= 0.05) / len(built)) if built else None,
        "named_ok": (sum(named) / len(named)) if named else None,  # share of the sheets that were split and named
    }


def summarize(rows: list[dict]) -> dict:
    categories = {cat: _stats([r for r in rows if r["category"] == cat])
                 for cat in sorted({r["category"] for r in rows})}
    return {"overall": _stats(rows), "categories": categories, "sheet": any("named_ok" in r for r in rows)}


def _fmt(v, pct: bool = False) -> str:
    if v is None:
        return "-"
    return f"{v * 100:.1f}%" if pct else f"{v:.3f}"


def _md_row(name: str, s: dict, sheet: bool = False) -> str:
    named = f" {_fmt(s.get('named_ok'), pct=True)} |" if sheet else ""
    return (f"| {name} | {s['n']} | {s['built']} | {sum(s['abstained'].values())} | {s['skipped']} | "
           f"{s['errors']} | {_fmt(s['median_abs_vol_err'])} | {_fmt(s['median_voxel_iou'])} | "
           f"{_fmt(s['median_view_iou'])} | {_fmt(s['within_5pct_vol_err'], pct=True)} |{named}")


def summary_markdown(summary: dict) -> str:
    sheet = bool(summary.get("sheet"))
    header = ("| category | n | built | abstained | skipped | errors | median abs(vol_err) | median voxel IoU | "
             "median view IoU | built within 5% |" + (" named correctly |" if sheet else ""))
    title = ("Reverse-engineering benchmark, sheet mode: one first-angle line-art sheet per part (Canny edges of the "
             "front, top and right clean renders, labels on even-numbered parts), split and named by layout, true "
             "envelope given as user values." if sheet else
             "Reverse-engineering benchmark: clean renders, true envelope given as user values.")
    lines = [
        title,
        "",
        header,
        "|---" * (11 if sheet else 10) + "|",
        _md_row("overall", summary["overall"], sheet),
    ]
    lines += [_md_row(cat, s, sheet) for cat, s in summary["categories"].items()]
    return "\n".join(lines)
