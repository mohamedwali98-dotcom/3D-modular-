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
    SCALE,
    SKIP,
    Naming,
    Sheet,
    _remove_border,
    body_of,
    ink_mask,
    is_sheet,
    name_views,
    named_count,
    round_view,
    split_by_outlines,
    split_sheet,
)

log = logging.getLogger(__name__)

CROP_MARGIN = 0.04
EXPLAINED = 0.15     # a projection that leaves at most this line mismatch explains the drawing...
MARGIN = 0.3         # ...and wins when the other leaves at least this much more; otherwise ISO first-angle stays


CLEAN_SHARE = 0.97   # a drawing: this share of its pixels is paper or ink...
CLEAN_GREY = 40      # ...within this many grey levels of the paper or of the darkest ink...
CLEAN_SAT = 25       # ...and its paper has no colour cast (mean HSV saturation under this)
SKETCH_SCALE = 0.2   # hand sketches are not drawn to scale: the naming's size check allows this share (spec 3.2)
SKETCH_ALIGN = 0.2   # ...and hand-placed views line up within this share of their size
SKETCH_EXTENT = 0.4  # ...sharing an extent within this share
TURN_MAX = 6.0       # a sketch view turned by up to this many degrees is turned back square
SNAP = 5.0           # a fitted line within this many degrees of the axes is drawn exactly along them
FIT_COVER = 0.8      # the fitted lines and circles cover this share of a view's ink, or it is kept as drawn
DASH = 6.0           # a line shorter than this many line widths may be a dash of a hidden line


@dataclass
class Page:
    """The image the five steps read: black ink on white. A photo of a sketch is its rectified page; a drawing is
    used as drawn. `to_photo` maps page pixels back to the upload (for overlays)."""
    image: np.ndarray
    kind: str            # "sketch" | "drawing"
    to_photo: np.ndarray
    stroke_px: float


def _clean(image: np.ndarray) -> bool:
    gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    g = gray.astype(np.int16)
    paper = float(np.median(np.concatenate([g[0], g[-1], g[:, 0], g[:, -1]])))
    ink = float(np.percentile(g, 1))
    if paper < 150 or paper - ink < 3 * CLEAN_GREY:  # no light paper, or nothing drawn on it
        return False
    papery = np.abs(g - paper) <= CLEAN_GREY
    if np.mean(papery | (np.abs(g - ink) <= CLEAN_GREY)) < CLEAN_SHARE:
        return False
    if image.ndim == 3 and papery.any():
        return float(cv2.cvtColor(image, cv2.COLOR_BGR2HSV)[..., 1][papery].mean()) < CLEAN_SAT
    return True


def page_of(image_bgr: np.ndarray) -> Page | S.MvAbstain:
    """Step 0: a clean drawing is read as drawn; a photo of a sketch goes through the sketch reader's capture (the
    sheet found, rectified, shadows flattened, ink binarised). A photo it cannot use abstains with its remedy."""
    image = image_bgr if image_bgr.ndim == 3 else cv2.cvtColor(image_bgr, cv2.COLOR_GRAY2BGR)
    if _clean(image):
        ink = ink_mask(image) > 0
        dist = cv2.distanceTransform(ink.astype(np.uint8), cv2.DIST_L2, 3)
        stroke = 2 * float(np.percentile(dist[ink], 75)) if ink.any() else 2.0
        return Page(image, "drawing", np.eye(3), max(1.0, stroke))
    from s2c.sketch.capture import capture
    from s2c.sketch.models import SketchAbstain
    captured = capture(image)
    if isinstance(captured, SketchAbstain):
        return S.MvAbstain(stage="outline", reason=captured.reason, remedy=captured.remedy)
    page = cv2.cvtColor(np.where(captured.ink > 0, 0, 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    return Page(page, "sketch", np.asarray(captured.to_original, np.float64), float(captured.stroke_px))


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
    kind: str = "drawing"       # "sketch" | "drawing" (read_drawing)
    page: Page | None = None    # the page read, with its mapping back to the upload

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
        (x, y, w, h), mask = body_of(ink, view, long)
        keep = cv2.dilate(mask.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
        m = max(2, round(CROP_MARGIN * max(w, h)))
        crop = np.full((h + 2 * m, w + 2 * m, 3), 255, np.uint8)
        body = image[y: y + h, x: x + w].copy()
        body[(ink[y: y + h, x: x + w] == 0) | ~keep] = 255
        crop[m: m + h, m: m + w] = body
        if sheet.stroke_px:
            crop = _straighten(crop, sheet.stroke_px)
        out.append(SheetCrop(i, cv2.imencode(".png", crop)[1].tobytes(), face, (x - m, y - m), max(crop.shape[:2])))
    return out


def _straighten(crop: np.ndarray, stroke: float) -> np.ndarray:
    """A sketch view drawn again from the lines, circles, arcs and curves fitted to its strokes (spec 3.3): turned
    back square by its lines' median slant, lines near the axes drawn exactly along them with their ends on the
    lines they nearly meet (a hand corner that stops short or overshoots closes exactly), circles round, arcs and curves as drawn. Kept as drawn when
    the fit covers under FIT_COVER of its ink (hatching, lettering, a shaded view)."""
    from s2c.sketch.vectorize import vectorize
    ink = (cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) < 128).astype(np.uint8) * 255
    prims = vectorize(ink, stroke, "view")
    if not prims or not ink.any():
        return crop
    width = max(1, round(float(np.median([p.width for p in prims]))))
    reach = np.zeros_like(ink)
    for p in prims:
        cv2.polylines(reach, [p.pts.round().astype(np.int32)], False, 255, width + 2 * max(2, round(stroke)))
    if np.count_nonzero(reach & ink) < FIT_COVER * np.count_nonzero(ink):
        return crop
    slants = [(_slant(p.p1 - p.p0), p.length) for p in prims if p.kind == "line"]
    slants = sorted((a, n) for a, n in slants if abs(a) <= TURN_MAX)
    turn = 0.0
    if slants:
        weights = np.cumsum([n for _, n in slants])
        turn = slants[int(np.searchsorted(weights, weights[-1] / 2))][0]
    h, w = ink.shape
    pad = round(0.05 * max(h, w))
    rot = cv2.getRotationMatrix2D((w / 2, h / 2), turn, 1.0)
    rot[:, 2] += pad
    out = np.zeros((h + 2 * pad, w + 2 * pad), np.uint8)
    square = []  # (axis, position across, start, end, a dash?) of the lines near the axes
    for p in prims:
        pts = cv2.transform(p.pts[None].astype(np.float64), rot)[0]
        if p.kind == "circle":
            c = cv2.transform(np.array([[p.center]], np.float64), rot)[0, 0]
            cv2.circle(out, (round(c[0]), round(c[1])), round(p.radius), 255, width)
        elif p.kind == "line" and (abs(_slant(pts[1] - pts[0])) <= SNAP or (
                p.length < DASH * stroke and min(np.abs(pts[1] - pts[0])) <= stroke)):  # a dash's slant is noise
            (x0, y0), (x1, y1) = pts
            if abs(x1 - x0) >= abs(y1 - y0):
                square.append(["h", (y0 + y1) / 2, min(x0, x1), max(x0, x1)])
            else:
                square.append(["v", (x0 + x1) / 2, min(y0, y1), max(y0, y1)])
        else:
            cv2.polylines(out, [pts.round().astype(np.int32)], False, 255, width)
    for axis in "hv":  # the pieces of one edge (split at junctions, or dashes) share its position: no 1 px steps
        line = sorted((sq for sq in square if sq[0] == axis), key=lambda sq: sq[1])
        groups, group = [], []
        for sq in line:
            if group and sq[1] - group[0][1] > 2 * stroke:
                groups.append(group)
                group = []
            group.append(sq)
        for run in groups + [group]:
            at = float(np.average([sq[1] for sq in run], weights=[sq[3] - sq[2] + 1 for sq in run]))
            for sq in run:
                sq[1] = at
    square = _join(square, stroke)
    reach = 2 * stroke
    solid = [sq for sq in square if not sq[4]]

    def meets(axis, at, e, near=reach):
        """The line across (axis, at) that the end e nearly meets, or None."""
        across = [t[1] for t in solid if t[0] != axis and t[2] - reach <= at <= t[3] + reach]
        return min((t for t in across if abs(t - e) <= near), key=lambda t: abs(t - e), default=None)

    for axis, at, lo, hi, dash in solid:  # each end on the line it nearly meets: a hand corner closes exactly
        lo, hi = (e if t is None else t for e, t in ((lo, meets(axis, at, lo)), (hi, meets(axis, at, hi))))
        _segment(out, axis, at, lo, hi, width)
    run_reach = DASH * stroke  # a hidden line's first dash often runs into the outline: a dash and a gap short
    for axis, at, lo, hi in _dash_runs([sq for sq in square if sq[4]], stroke):
        _hidden(out, axis, at, lo, meets(axis, at, lo, run_reach), hi, meets(axis, at, hi, run_reach), width)
    return cv2.cvtColor(255 - out, cv2.COLOR_GRAY2BGR)


def _join(square: list, stroke: float) -> list:
    """The pieces of one line that meet joined (a skeleton splits a line where another meets it), each marked as
    a possible dash when what stays is short: [axis, position, start, end, dash?]."""
    out = []
    for sq in sorted(square, key=lambda sq: (sq[0], sq[1], sq[2])):
        last = out[-1] if out else None
        if last and last[0] == sq[0] and last[1] == sq[1] and sq[2] - last[3] <= max(2.0, 0.6 * stroke):
            last[3] = max(last[3], sq[3])
        else:
            out.append(list(sq))
    return [sq + [sq[3] - sq[2] < DASH * stroke] for sq in out]


def _segment(out: np.ndarray, axis: str, at: float, lo: float, hi: float, width: int) -> None:
    ends = [(lo, at), (hi, at)] if axis == "h" else [(at, lo), (at, hi)]
    cv2.line(out, *(tuple(round(v) for v in e) for e in ends), 255, width)


def _dash_runs(dashes: list, stroke: float) -> list[tuple[str, float, float, float]]:
    """Dashes on one line with gaps under three line widths: the hidden lines, (axis, position, start, end)."""
    runs = []
    for axis in "hv":
        line = sorted((d for d in dashes if d[0] == axis), key=lambda d: (d[1], d[2]))
        run = []
        for d in line:
            if run and (d[1] != run[-1][1] or d[2] - run[-1][3] > 3 * stroke):
                runs.append(run)
                run = []
            run.append(d)
        runs += [run] if run else []
    return [(r[0][0], r[0][1], r[0][2], r[-1][3]) for r in runs]


def _hidden(out: np.ndarray, axis: str, at: float, lo: float, lo_line: float | None, hi: float,
            hi_line: float | None, width: int) -> None:
    """A hidden line drawn again as even dashes. An end on an outline starts a clear gap from it (the dash must
    not join the outline, or it is not seen as a dash); a free end stays where it was drawn."""
    dash, gap = 4 * width, 2 * width
    if lo_line is not None:
        lo = lo_line + width / 2 + gap
    if hi_line is not None:
        hi = hi_line - width / 2 - gap
    n = max(1, round((hi - lo + gap) / (dash + gap)))
    dash = (hi - lo - (n - 1) * gap) / n
    if dash < 2 * width:  # too short for a dash pattern: one stroke as drawn
        _segment(out, axis, at, lo, hi, width)
        return
    half = width / 2
    for k in range(n):
        a = lo + k * (dash + gap)
        x0, y0, x1, y1 = (a, at - half, a + dash, at + half) if axis == "h" else (at - half, a, at + half, a + dash)
        cv2.rectangle(out, (round(x0), round(y0)), (round(x1), round(y1)), 255, cv2.FILLED)


def _slant(d: np.ndarray) -> float:
    """A direction's angle from the nearest axis, in degrees, in (-45, 45]."""
    return float((np.degrees(np.arctan2(d[1], d[0])) + 45) % 90 - 45)


def part_bodies(sheet: Sheet, image: np.ndarray, naming: Naming):
    """The sheet's ink and the bodies of the part drawing's views: named ones and unnamed ones (a picture), never
    a mark (dimension text split off on its own: the loops of a 0, 6, 8 or 9 would fill into a "body" and the
    number would be erased before it is read)."""
    ink = ink_mask(image)
    long = max(image.shape[:2])
    views = sheet.drawings[naming.drawing].views if naming.drawing >= 0 else []
    return ink, [body_of(ink, v, long) for v, f in zip(views, naming.faces, strict=True)
                 if f != SKIP and v.line_art]


def _envelope(sheet: Sheet, naming: Naming, image: np.ndarray) -> S.Envelope | None:
    """The part's proportions in sheet pixels, from the named views' bodies (any one unit serves the comparison)."""
    ink = ink_mask(image)
    long = max(image.shape[:2])
    sizes: dict[str, list[float]] = {"x": [], "y": [], "z": []}
    for view, face in zip(sheet.drawings[naming.drawing].views, naming.faces, strict=True):
        if face not in S.FACE_AXES:
            continue
        _, _, w, h = body_of(ink, view, long)[0]
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


def _choose(sheet: Sheet, image: np.ndarray, reader, scale_tol: float = SCALE) -> Naming:
    """The symbol, then consistent labels, decide the projection; otherwise the drawing does: the reading whose views
    explain each other's lines clearly better wins. A tie keeps first-angle (ISO), the default convention."""
    first = name_views(sheet, image, "first", reader, scale_tol)
    if first.projection_source in ("symbol", "labels"):
        return first
    third = name_views(sheet, image, "third", reader, scale_tol)
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


def choose_naming(sheet: Sheet, image: np.ndarray, projection: str, reader=None, scale_tol: float = SCALE) -> Naming:
    """Name the views: "auto" lets the drawing choose the projection, "first" or "third" set it."""
    if projection == "auto":
        return _choose(sheet, image, reader, scale_tol)
    return name_views(sheet, image, projection, reader, scale_tol)


def read_sheet(image_bgr: np.ndarray, projection: str = "auto", reader=None, service=None,
               keep_unnamed: bool = False) -> SheetRead | None:
    """The sheet read end to end, or None when the image is not a sheet of line-drawn views. `projection` is
    "auto" (the drawing decides), "first" or "third" (a symbol or labels still win)."""
    image = image_bgr if image_bgr.ndim == 3 else cv2.cvtColor(image_bgr, cv2.COLOR_GRAY2BGR)
    sheet = split_sheet(image)
    if not is_sheet(sheet):
        return None
    return _read(sheet, image, projection, reader, service, keep_unnamed, SCALE)


def read_drawing(image_bgr: np.ndarray, projection: str = "auto", reader=None, service=None,
                 keep_unnamed: bool = False) -> SheetRead | S.MvAbstain | None:
    """One image to named views and their numbers (sketch-to-model spec 3): a clean drawing is read exactly as
    `read_sheet` reads it; a photo of a hand sketch becomes a clean page (an abstention when it cannot), its views
    are its closed outlines, and naming allows for a sketch not drawn to scale. None when no sheet of views is
    found."""
    page = page_of(image_bgr)
    if isinstance(page, S.MvAbstain):
        return page
    if page.kind == "drawing":
        read = read_sheet(page.image, projection, reader, service, keep_unnamed)
    else:
        sheet = split_by_outlines(page.image, page.stroke_px)
        if sum(len(d.views) for d in sheet.drawings) < 2:
            sheet = split_sheet(page.image)
        if not is_sheet(sheet, SKETCH_ALIGN, SKETCH_EXTENT):
            return None
        sheet.stroke_px = page.stroke_px
        read = _read(sheet, page.image, projection, reader, service, keep_unnamed, SKETCH_SCALE)
    if read is not None:
        read.kind, read.page = page.kind, page
    return read


def _read(sheet: Sheet, image: np.ndarray, projection: str, reader, service, keep_unnamed: bool,
          scale_tol: float) -> SheetRead | None:
    """Steps 2 to 4 on a split sheet: name the views, read the numbers, crop the named views."""
    naming = choose_naming(sheet, image, projection, reader, scale_tol)
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
    _remove_border(ink)  # the sheet's frame is no dimension line
    scale = read_dimensions(image, ink, bodies, service)
    warnings += scale.warnings
    if scale.mm_per_px:
        warnings.append(f"Sizes read from the drawing's dimensions ({len(scale.used)} used); check them.")
    return SheetRead(sheet, naming, _crops(sheet, image, naming, keep_unnamed), scale, warnings)


DIAMETER_OFF = 0.10  # a written Ø further than this from the measured circle is flagged, not trusted


def link_diameters(read: SheetRead, observed) -> None:
    """A Ø written next to a drawn circle is that hole's size, written by the user: attach it to the circle of the
    observation it was drawn in (rule 2: user_written). An R is a corner or an arc, never linked to a hole. With
    the sheet's scale known, a Ø more than DIAMETER_OFF from the measured circle is kept as a check, with a note."""
    from s2c.multiview.ocr import Linked, Reading
    from s2c.multiview.outline import LONG_SIDE
    dims = [d for d in read.scale.dimensions if d.kind == "diameter"]
    if not dims or not observed.observations:
        return
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
        confirmed = d.confirmed
        if read.scale.mm_per_px:
            measured = 2 * best[2] * read.scale.mm_per_px
            if abs(d.value_mm - measured) > DIAMETER_OFF * measured:
                confirmed = False
                observed.warnings.append(f"{d.text} is written by a hole drawn about {measured:.1f} mm wide; "
                                         "check the diameter.")
        o = observed.observations[best[3]]
        o.values.append(Linked(Reading(d.value_mm, "diameter", d.box, 0.95, d.text, confirmed), None, best[4]))
