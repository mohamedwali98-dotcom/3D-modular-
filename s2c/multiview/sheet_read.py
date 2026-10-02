"""One drawing sheet, read end to end (sheet-reading spec 2026-10-01): its views split from their annotations and
named (the projection chosen by the drawing when no symbol or label says it), its dimensions read into one scale,
and the crops the multi-view pipeline builds from. The isometric picture a sheet often carries is the target part,
but it is no view: it never gets a face."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import cv2
import numpy as np

from s2c.multiview import spec as S
from s2c.multiview.dimensions import SheetScale, read_dimensions
from s2c.multiview.sheet import (
    SKIP,
    Naming,
    Sheet,
    ink_mask,
    is_sheet,
    name_views,
    named_count,
    round_view,
    split_sheet,
    view_body,
)

log = logging.getLogger(__name__)

CROP_MARGIN = 0.04
EXPLAINED = 0.15     # a projection that leaves at most this line mismatch explains the drawing...
MARGIN = 0.3         # ...and wins when the other leaves at least this much more; otherwise ISO first-angle stays


@dataclass
class SheetCrop:
    view: int                 # index in the part drawing's views
    png: bytes
    face: str
    origin: tuple[int, int]   # the crop's top-left corner on the sheet
    long: int                 # the crop's long side, px


@dataclass
class SheetRead:
    sheet: Sheet
    naming: Naming
    crops: list[SheetCrop]
    scale: SheetScale
    warnings: list[str] = field(default_factory=list)

    def inputs(self):
        """The views as pipeline inputs: kind "drawing", the named face, and the sheet's scale when it was read."""
        from s2c.multiview.pipeline import ImageInput
        return [ImageInput(c.png, c.face, "drawing", mm_per_px=self.scale.mm_per_px,
                           scale_confirmed=self.scale.confirmed) for c in self.crops]


def _crops(sheet: Sheet, image: np.ndarray, naming: Naming, keep_unnamed: bool = False) -> list[SheetCrop]:
    """Each named view's body only, on white, at the sheet's scale, with a small margin. Views that stay "auto" (an
    isometric picture, an unnamed view) are left out unless `keep_unnamed` (the Studio, which has a face picker):
    a face is never guessed."""
    ink = ink_mask(image)
    long = max(image.shape[:2])
    out = []
    for i, (view, face) in enumerate(zip(sheet.drawings[naming.drawing].views, naming.faces, strict=True)):
        if face == SKIP or (face == "auto" and not keep_unnamed):
            continue
        (x, y, w, h), mask = view_body(ink, view.box, long)
        keep = cv2.dilate(mask.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
        m = max(2, round(CROP_MARGIN * max(w, h)))
        crop = np.full((h + 2 * m, w + 2 * m, 3), 255, np.uint8)
        body = image[y: y + h, x: x + w].copy()
        body[(ink[y: y + h, x: x + w] == 0) | ~keep] = 255
        crop[m: m + h, m: m + w] = body
        out.append(SheetCrop(i, cv2.imencode(".png", crop)[1].tobytes(), face, (x - m, y - m), max(crop.shape[:2])))
    return out


def part_bodies(sheet: Sheet, image: np.ndarray, naming: Naming):
    """The sheet's ink and the bodies of the part drawing's views: named ones and unnamed ones (a picture), never
    a mark (dimension text split off on its own: the loops of a 0, 6, 8 or 9 would fill into a "body" and the
    number would be erased before it is read)."""
    ink = ink_mask(image)
    long = max(image.shape[:2])
    views = sheet.drawings[naming.drawing].views if naming.drawing >= 0 else []
    return ink, [view_body(ink, v.box, long) for v, f in zip(views, naming.faces, strict=True)
                 if f != SKIP and v.line_art]


def _envelope(sheet: Sheet, naming: Naming, image: np.ndarray) -> S.Envelope | None:
    """The part's proportions in sheet pixels, from the named views' bodies (any one unit serves the comparison)."""
    ink = ink_mask(image)
    long = max(image.shape[:2])
    sizes: dict[str, list[float]] = {"x": [], "y": [], "z": []}
    for view, face in zip(sheet.drawings[naming.drawing].views, naming.faces, strict=True):
        if face not in S.FACE_AXES:
            continue
        _, _, w, h = view_body(ink, view.box, long)[0]
        a_axis, b_axis, _ = S.FACE_AXES[face]
        sizes[a_axis].append(w)
        sizes[b_axis].append(h)
    if not all(sizes.values()):
        return None
    return S.Envelope(**{f"{a}_mm": float(np.median(v)) for a, v in sizes.items()})


def _mismatch(sheet: Sheet, image: np.ndarray, naming: Naming) -> float | None:
    """How badly the drawn lines agree with each other when the views carry these faces (relief.mismatch)."""
    from s2c.multiview.fuse import Observation, canonical_outlines
    from s2c.multiview.outline import extract, resize_long_side
    from s2c.multiview.relief import mismatch
    env = _envelope(sheet, naming, image)
    if env is None:
        return None
    observations = []
    for c in _crops(sheet, image, naming):
        bgr = resize_long_side(cv2.imdecode(np.frombuffer(c.png, np.uint8), cv2.IMREAD_COLOR))
        outline = extract(bgr, drawing=True)
        if isinstance(outline, S.MvAbstain) or not outline.line_art:
            return None
        observations.append(Observation(face=c.face, kind="drawing", outline=outline))
    outlines, _ = canonical_outlines(observations, env)
    return mismatch(observations, {f: ol for f, (ol, _) in outlines.items()}, env)


def _choose(sheet: Sheet, image: np.ndarray, reader) -> Naming:
    """The symbol, then consistent labels, decide the projection; otherwise the drawing does: the reading whose views
    explain each other's lines clearly better wins. A tie keeps first-angle (ISO), the default convention."""
    first = name_views(sheet, image, "first", reader)
    if first.projection_source in ("symbol", "labels"):
        return first
    third = name_views(sheet, image, "third", reader)
    if third.projection_source in ("symbol", "labels") or first.faces == third.faces:
        return third if third.projection_source != "setting" else first
    a, b = _mismatch(sheet, image, first), _mismatch(sheet, image, third)
    log.info("projection mismatch: first %s, third %s", a, b)
    if a is None or b is None:
        return first
    if b <= EXPLAINED and a - b >= MARGIN:
        third.projection_source = "drawing"
        third.warnings.append("Read as third-angle (US): the views agree with each other that way.")
        return third
    if a <= EXPLAINED and b - a >= MARGIN:
        first.projection_source = "drawing"
    return first


def choose_naming(sheet: Sheet, image: np.ndarray, projection: str, reader=None) -> Naming:
    """Name the views: "auto" lets the drawing choose the projection, "first" or "third" set it."""
    return _choose(sheet, image, reader) if projection == "auto" else name_views(sheet, image, projection, reader)


def read_sheet(image_bgr: np.ndarray, projection: str = "auto", reader=None, service=None,
               keep_unnamed: bool = False) -> SheetRead | None:
    """The sheet read end to end, or None when the image is not a sheet of line-drawn views. `projection` is
    "auto" (the drawing decides), "first" or "third" (a symbol or labels still win)."""
    image = image_bgr if image_bgr.ndim == 3 else cv2.cvtColor(image_bgr, cv2.COLOR_GRAY2BGR)
    sheet = split_sheet(image)
    if not is_sheet(sheet):
        return None
    naming = choose_naming(sheet, image, projection, reader)
    if named_count(naming) < 2:
        return None
    ink = ink_mask(image)
    views = sheet.drawings[naming.drawing].views
    if all(round_view(ink, v.box) for v, f in zip(views, naming.faces, strict=True) if f not in ("auto", SKIP)):
        return None
    warnings = list(naming.warnings)
    if any(f == "auto" for f in naming.faces) and not keep_unnamed:  # no face picker: the view is left out instead
        warnings = [w for w in warnings if "pick the face by hand" not in w]
        warnings.append("A view that could not be named (an isometric picture, a detail) was left out; it is "
                        "only a picture of the part.")
    ink, bodies = part_bodies(sheet, image, naming)
    scale = read_dimensions(image, ink, bodies, service)
    warnings += scale.warnings
    if scale.mm_per_px:
        warnings.append(f"Sizes read from the drawing's dimensions ({len(scale.used)} used); check them.")
    return SheetRead(sheet, naming, _crops(sheet, image, naming, keep_unnamed), scale, warnings)


def link_diameters(read: SheetRead, observed) -> None:
    """A ⌀ or R written next to a drawn circle is that hole's size, written by the user: attach it to the circle
    of the observation it was drawn in (rule 2: user_written)."""
    from s2c.multiview.ocr import Linked, Reading
    dims = [d for d in read.scale.dimensions if d.kind in ("diameter", "radius")]
    if not dims or not observed.observations:
        return
    from s2c.multiview.outline import LONG_SIDE
    circles = []  # (sheet x, sheet y, sheet radius, observation index, circle index)
    for k, (crop, o) in enumerate(zip(read.crops, observed.observations, strict=False)):
        f = LONG_SIDE / max(crop.long, 1)
        for i, c in enumerate(o.outline.circles):
            circles.append((crop.origin[0] + c.cx / f, crop.origin[1] + c.cy / f, c.d / f / 2, k, i))
    for d in dims:
        x, y, w, h = d.box
        cx, cy = x + w / 2, y + h / 2
        best = min(circles, key=lambda c: np.hypot(c[0] - cx, c[1] - cy) - c[2], default=None)
        if best is None or np.hypot(best[0] - cx, best[1] - cy) - best[2] > 3 * max(w, h):
            continue
        o = observed.observations[best[3]]
        value = d.value_mm * (2 if d.kind == "radius" else 1)
        o.values.append(Linked(Reading(value, "diameter", d.box, 0.95, d.text, d.confirmed), None, best[4]))
