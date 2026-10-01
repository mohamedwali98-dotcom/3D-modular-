"""Image -> outer outline, openings and circles, in pixels. Spec section 6.2.
One rule covers pen sketches (a drawn ring) and photos (a filled part). Clean drawings made of lines follow
the line-art rules of the drawing-sheet spec (2026-09-27) section 3.3."""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from s2c.multiview.spec import MvAbstain

LONG_SIDE = 1600
MIN_OUTLINE_FRACTION = 0.02
MIN_OPENING_FRACTION = 0.0003
CIRCULARITY = 0.85
EDGE_BAND_PX = 31  # the inside of a drawn outline touches this band next to the edge; a real opening does not
MIN_THIN_FRACTION = 0.002  # a thin part's edge view can be this small and still be a real outline
MIN_THIN_SPAN = 0.10       # ...provided it is long relative to the page...
MIN_THIN_FILL = 0.5        # ...and filled, not a hollow or broken stroke...
MIN_THIN_MARGIN = 0.01     # ...and fully in frame, not a table edge or ruler crossing the border
LINE_ART_FILL = 0.35       # a drawing whose ink covers less of its filled outline than this is drawn in lines...
LINE_ART_STROKE = 0.015    # ...if the ink is also this thin (share of the long side); a thin-walled render is not
LINE_ART_DARK = 200        # ...or, ink this far from the paper (grey levels, median), a tiny view blown up...
LINE_ART_DARK_STROKE = 0.05  # ...whose lines grew up to this thick; rendered faces are shaded, not this dark
SPUR_FRACTION = 0.006      # lines thinner than this share of the long side (and SPUR_MIN_PX) are not the part
SPUR_MIN_PX = 5
SPUR_MAX = 0.025           # the kernel widened for thick lines stays under this share of the long side
SPUR_KEEP = 0.5            # cutting spurs keeps most of the view, or the view itself was that thin
SPUR_MIN_REACH = 3         # px: a piece cut off that sticks out less is a sliver of a curve, not a line
SPUR_TURN = 45             # degrees: under a line sticking out the outline runs on; at a corner it turns
SPUR_FEW = 8               # centre lines overhang in a few places; more short pieces are teeth
ROUND_ASPECT = 0.85        # a drawn circle is about as wide as it is tall...
ROUND_FILL = 0.9           # ...fills this share of its hull...
SPLIT_FILL = 0.6           # ...and the pieces of one cut by centre lines fill at least this share
DASH_RUN = 3               # a hidden line is at least this many dashes...
DASH_MAX = 0.15            # ...each shorter than this share of the view
RING_SPAN = 1.6            # a concentric ring lies within this many radii of a drawn circle...
RING_HITS = 0.85           # ...and this share of rays from the centre meets it...
RING_SPREAD = 0.05         # ...at radii this close (share of the radius): a circle, not other lines
LINE_MIN = 0.04            # a visible line is at least this share of the view's long side...
LINE_JOIN = 0.025          # ...and each of its ends meets another line or the outline within this share


@dataclass
class PixelCircle:
    cx: float
    cy: float
    d: float
    ring: float = 0.0  # diameter of a concentric circle drawn just around it (a pin's tip chamfer, a counterbore)


@dataclass
class PixelOutline:
    outer: np.ndarray                                    # (n, 2) image pixels
    inner: list[np.ndarray] = field(default_factory=list)
    circles: list[PixelCircle] = field(default_factory=list)
    bbox: tuple[int, int, int, int] = (0, 0, 1, 1)        # x, y, w, h of the outer outline
    circular: bool = False                               # the outer outline is itself a circle
    shape: tuple[int, int] = (1, 1)                      # image height, width
    line_art: bool = False                               # a drawing in lines: no openings, circles only
    hidden: list[tuple[str, float, float, float]] = field(default_factory=list)  # see find_hidden_lines
    lines: list[tuple[str, float, float, float]] = field(default_factory=list)   # see find_visible_lines


def resize_long_side(image: np.ndarray, long_side: int = LONG_SIDE) -> np.ndarray:
    h, w = image.shape[:2]
    s = long_side / max(h, w)
    interp = cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC
    return cv2.resize(image, (round(w * s), round(h * s)), interpolation=interp)


def circularity(contour) -> float:
    perimeter = cv2.arcLength(contour, True)
    return 0.0 if perimeter == 0 else float(4 * np.pi * cv2.contourArea(contour) / perimeter ** 2)


def ink_mask(image_bgr: np.ndarray, mask_out=()) -> np.ndarray:
    """Ink or part pixels are 255, before small breaks are closed, so the gaps of dashed lines survive.
    The polarity is chosen so the image border is background."""
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    background = int(np.median(np.concatenate([gray[0], gray[-1], gray[:, 0], gray[:, -1]])))
    for x, y, w, h in mask_out:
        gray[max(y, 0): y + h, max(x, 0): x + w] = background
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    if np.concatenate([th[0], th[-1], th[:, 0], th[:, -1]]).mean() > 127:
        th = 255 - th
    return th


def _closed(ink: np.ndarray) -> np.ndarray:
    return cv2.morphologyEx(ink, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))


def foreground(image_bgr: np.ndarray, mask_out=()) -> np.ndarray:
    """Ink or part pixels are 255, small breaks closed. The polarity is chosen so the image border is background."""
    return _closed(ink_mask(image_bgr, mask_out))


def is_thin_edge_view(contour, area: float, h: int, w: int) -> bool:
    """Small but long, filled and fully in frame: a thin part's edge-on silhouette, not a speck,
    a broken stroke, or a table edge / ruler / shadow crossing the image border."""
    if area < MIN_THIN_FRACTION * h * w:
        return False
    x, y, bw, bh = cv2.boundingRect(contour)
    if max(bw, bh) < MIN_THIN_SPAN * max(h, w):
        return False
    if area < MIN_THIN_FILL * bw * bh:
        return False
    margin = MIN_THIN_MARGIN * max(h, w)
    return x >= margin and y >= margin and (w - (x + bw)) >= margin and (h - (y + bh)) >= margin


def extract(image_bgr: np.ndarray, mask_out=(), band: int = EDGE_BAND_PX,
            drawing: bool = False) -> PixelOutline | MvAbstain:
    """`drawing` (input kind "drawing") allows the line-art rules. Line art is told by its ink fill, never by
    the kind alone, so a filled render sent as a drawing keeps the rules below."""
    ink = ink_mask(image_bgr, mask_out)
    fg = _closed(ink)
    h, w = fg.shape
    contours, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    outer = max(contours, key=cv2.contourArea) if contours else None
    outer_area = cv2.contourArea(outer) if outer is not None else 0.0
    if outer is None or (outer_area < MIN_OUTLINE_FRACTION * h * w
                          and not is_thin_edge_view(outer, outer_area, h, w)):
        return MvAbstain(stage="outline", reason="no_outline",
                         remedy="Retake on a plain background with the whole part in frame.")
    filled = np.zeros_like(fg)
    cv2.drawContours(filled, [outer], -1, 255, -1)
    if drawing and cv2.countNonZero(cv2.bitwise_and(fg, filled)) < LINE_ART_FILL * cv2.countNonZero(filled):
        stroke = _stroke(cv2.bitwise_and(ink, filled))
        if stroke <= LINE_ART_STROKE * max(h, w) or (
                stroke <= LINE_ART_DARK_STROKE * max(h, w) and _contrast(image_bgr, ink, filled) >= LINE_ART_DARK):
            return _line_art(ink, fg, outer, filled, band, stroke)
    edge_band = cv2.subtract(filled, cv2.erode(filled, np.ones((band, band), np.uint8)))
    gaps = cv2.morphologyEx(cv2.bitwise_and(filled, cv2.bitwise_not(fg)), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(gaps, connectivity=4)
    inner, circles = [], []
    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < MIN_OPENING_FRACTION * h * w:
            continue
        comp = np.where(labels == i, 255, 0).astype(np.uint8)
        if cv2.countNonZero(cv2.bitwise_and(comp, edge_band)):
            continue  # the inside of a drawn outline, not an opening
        c = max(cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0], key=cv2.contourArea)
        if circularity(c) >= CIRCULARITY:
            m = cv2.moments(comp, binaryImage=True)
            circles.append(PixelCircle(m["m10"] / m["m00"], m["m01"] / m["m00"], float(2 * np.sqrt(area / np.pi))))
        else:
            inner.append(cv2.approxPolyDP(c, 2.0, True).reshape(-1, 2))
    return PixelOutline(outer=cv2.approxPolyDP(outer, 2.0, True).reshape(-1, 2), inner=inner, circles=circles,
                        bbox=tuple(int(v) for v in cv2.boundingRect(outer)),
                        circular=circularity(outer) >= CIRCULARITY, shape=(h, w))


def _contrast(image_bgr: np.ndarray, ink: np.ndarray, filled: np.ndarray) -> float:
    """Median distance of the ink from the paper, in grey levels."""
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    paper = float(np.median(np.concatenate([gray[0], gray[-1], gray[:, 0], gray[:, -1]])))
    return float(np.median(np.abs(gray[cv2.bitwise_and(ink, filled) > 0].astype(float) - paper)))


def _stroke(ink: np.ndarray) -> float:
    """Typical line width: the ink area over half its edge length."""
    contours, _ = cv2.findContours(ink, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    edge = sum(cv2.arcLength(c, True) for c in contours)
    return 2 * cv2.countNonZero(ink) / edge if edge else 0.0


def _spur_kernel(shape, stroke: float) -> int:
    """Spec 3.3's kernel, widened for thick lines, such as those of a small sheet view blown up to full size."""
    return max(SPUR_MIN_PX, round(SPUR_FRACTION * max(shape)), min(2 * round(stroke) + 1, round(SPUR_MAX * max(shape))))


def _circle(solid: np.ndarray, dx: int, dy: int, stroke: float) -> PixelCircle:
    """A drawn circle's edge is the middle of its line: the region inside it grows by one line width."""
    m = cv2.moments(solid, binaryImage=True)
    d = 2 * np.sqrt(m["m00"] / np.pi) + stroke
    return PixelCircle(m["m10"] / m["m00"] + dx, m["m01"] / m["m00"] + dy, float(d))


def _ring(ink: np.ndarray, c: PixelCircle, stroke: float) -> float:
    """The diameter of a concentric circle drawn just outside c, line middle to line middle, or 0. Each ray from
    the centre leaves c's own line, then meets the next line; a ring meets them all at about one radius."""
    h, w = ink.shape
    radii = np.arange(max(1.0, c.d / 2 - stroke), RING_SPAN * c.d / 2, 0.5)
    if len(radii) < 4:
        return 0.0
    mids, outs = [], []
    for t in np.linspace(0, 2 * np.pi, 64, endpoint=False):
        xs = np.round(c.cx + radii * np.cos(t)).astype(int)
        ys = np.round(c.cy + radii * np.sin(t)).astype(int)
        ok = (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h)
        on = np.zeros(len(radii), bool)
        on[ok] = ink[ys[ok], xs[ok]] > 0
        edges = np.flatnonzero(np.diff(on.astype(np.int8)))  # where ink starts or stops along the ray
        starts = [e + 1 for e in edges if not on[e] and on[e + 1]]
        gap = max(6, round(stroke))  # samples of 0.5 px: white this long ends the band of ink around c
        own_end = next((e for e in edges if on[e] and not on[e + 1] and radii[e] >= c.d / 2
                        and not on[e + 1: e + 1 + gap].any()), None)
        if own_end is not None:
            outs.append(radii[own_end])
        if on[0]:  # inside c's own line: its end comes first, the next start is the ring
            starts = [s for s in starts if s > (edges[0] if len(edges) else len(on))]
        else:
            starts = starts[1:]  # the first start is c's own line
        if not starts:
            continue
        stops = [e for e in edges if on[e] and not on[e + 1] and e >= starts[0]]
        stop = stops[0] if stops else len(on) - 1
        mids.append((radii[starts[0]] + radii[stop]) / 2)
    if len(mids) < RING_HITS * 64:
        # two circles a line apart print as one thick line (a small view, a screenshot): its outer edge is the ring
        if len(outs) >= RING_HITS * 64 and np.median(outs) - c.d / 2 > 1.2 * stroke:
            return 2 * (float(np.median(outs)) - stroke / 2)
        return 0.0
    r = float(np.median(mids))
    if np.percentile(np.abs(np.array(mids) - r), 85) > max(2.0, RING_SPREAD * r):
        return 0.0
    return 2 * r


def _round(points, area: float, fill: float) -> bool:
    """Round by the hull, which a staircase edge of a blown-up drawing does not spoil; an ellipse is no circle."""
    hull = cv2.convexHull(points)
    _, (a, b), _ = cv2.minAreaRect(hull)
    return (circularity(hull) >= CIRCULARITY and min(a, b) >= ROUND_ASPECT * max(a, b)
            and area >= fill * cv2.contourArea(hull))


def _solid(comp: np.ndarray, k: int) -> np.ndarray:
    """The region padded by k, with the centre-line dashes inside it filled and the notches cut into it by
    lines thinner than k closed, so that a circle crossed by lines still reads as round."""
    pad = cv2.copyMakeBorder(comp, k, k, k, k, cv2.BORDER_CONSTANT, value=0)
    solid = np.zeros_like(pad)
    cv2.drawContours(solid, cv2.findContours(pad, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0], -1, 255, -1)
    return cv2.morphologyEx(solid, cv2.MORPH_CLOSE, np.ones((k, k), np.uint8))


def _line_art(ink: np.ndarray, fg: np.ndarray, outer, filled: np.ndarray, band: int, stroke: float) -> PixelOutline:
    """Spec 3.3. Centre, extension and dimension lines sticking out are opened away. Regions enclosed by
    visible lines are faces, not openings, so only circles are kept, as candidate holes."""
    h, w = fg.shape
    k = _spur_kernel(fg.shape, stroke)
    opened = cv2.morphologyEx(filled, cv2.MORPH_OPEN, np.ones((k, k), np.uint8))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(opened, connectivity=8)
    body = filled  # also when the view itself is about as thin as a line: it stays as drawn
    if n > 1:
        core = np.where(labels == 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA])), 255, 0).astype(np.uint8)
        spurs = _spurs(filled, core, k)
        if cv2.countNonZero(spurs) and cv2.countNonZero(core) >= SPUR_KEEP * cv2.countNonZero(filled):
            outer = max(cv2.findContours(cv2.subtract(filled, spurs), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0],
                        key=cv2.contourArea)
            body = np.zeros_like(filled)
            cv2.drawContours(body, [outer], -1, 255, -1)
    bbox = tuple(int(v) for v in cv2.boundingRect(outer))
    edge_band = cv2.subtract(body, cv2.erode(body, np.ones((band, band), np.uint8)))
    gaps = cv2.morphologyEx(cv2.bitwise_and(body, cv2.bitwise_not(fg)), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(gaps, connectivity=4)
    min_area = MIN_OPENING_FRACTION * h * w
    circles, pieces = [], np.zeros_like(gaps)
    for i in range(1, n):
        x, y, bw, bh, area = (int(v) for v in stats[i])
        if area < min_area / 8:
            continue
        comp = np.where(labels[y: y + bh, x: x + bw] == i, 255, 0).astype(np.uint8)
        if cv2.countNonZero(cv2.bitwise_and(comp, edge_band[y: y + bh, x: x + bw])):
            continue  # a face between visible lines that reaches the outline
        solid = _solid(comp, k)
        c = max(cv2.findContours(solid, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0], key=cv2.contourArea)
        if not _round(c, cv2.contourArea(c), ROUND_FILL):
            pieces[y: y + bh, x: x + bw] |= comp
        elif cv2.countNonZero(solid) >= min_area:
            circles.append(_circle(solid, x - k, y - k, stroke))
    circles += _split_circles(pieces, k, min_area, stroke)
    drawn = cv2.bitwise_and(ink, body)
    for c in circles:
        c.ring = _ring(drawn, c, stroke)
    return PixelOutline(outer=cv2.approxPolyDP(outer, 2.0, True).reshape(-1, 2), circles=circles, bbox=bbox,
                        circular=circularity(outer) >= CIRCULARITY, shape=(h, w), line_art=True,
                        hidden=find_hidden_lines(drawn, bbox),
                        lines=find_visible_lines(drawn, body, bbox, circles, stroke))


def _spurs(filled: np.ndarray, core: np.ndarray, k: int) -> np.ndarray:
    """What the opening cut off that is a line sticking out: any piece reaching further than k, and a shorter
    one standing on a stretch where the outline runs on. A piece where the outline turns is a corner's own tip,
    which a square kernel shaves off an acute corner; it stays. So do short pieces all round the outline: the
    part's own teeth, not the few places where centre lines overhang."""
    cut = cv2.subtract(filled, core)
    reach = cv2.distanceTransform(cv2.bitwise_not(core), cv2.DIST_L2, 3)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(np.where(reach > SPUR_MIN_REACH, cut, 0).astype(np.uint8))
    spurs, short = np.zeros_like(cut), []
    if n < 2:
        return spurs
    pts = max(cv2.findContours(core, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0], key=cv2.contourArea).reshape(-1, 2)
    g = SPUR_MIN_REACH + 2  # the piece's foot, nearer the core than SPUR_MIN_REACH, goes with it
    for i in range(1, n):
        x, y, w, h = (int(v) for v in stats[i, :4])
        x0, y0, x1, y1 = max(x - g, 0), max(y - g, 0), min(x + w + g, cut.shape[1]), min(y + h + g, cut.shape[0])
        seed = np.where(labels[y0: y1, x0: x1] == i, 255, 0).astype(np.uint8)
        piece = cv2.bitwise_and(cv2.dilate(seed, np.ones((2 * g + 1, 2 * g + 1), np.uint8)), cut[y0: y1, x0: x1])
        if reach[y0: y1, x0: x1][piece > 0].max() > k:
            spurs[y0: y1, x0: x1] |= piece
            continue
        near = cv2.dilate(piece, np.ones((3, 3), np.uint8))
        inside = np.flatnonzero((pts[:, 0] >= x0) & (pts[:, 0] < x1) & (pts[:, 1] >= y0) & (pts[:, 1] < y1))
        idx = inside[near[pts[inside, 1] - y0, pts[inside, 0] - x0] > 0]
        if _turn(pts, idx, k) < SPUR_TURN:
            short.append((x0, y0, x1, y1, piece))
    if len(short) <= SPUR_FEW:
        for x0, y0, x1, y1, piece in short:
            spurs[y0: y1, x0: x1] |= piece
    return spurs


def _turn(pts: np.ndarray, idx: np.ndarray, m: int) -> float:
    """How far, in degrees, the closed outline `pts` turns across the stretch of points `idx`, each side
    measured over m points."""
    if len(idx) == 0:
        return 0.0
    n = len(pts)
    j = int(np.argmax(np.diff(np.r_[idx, idx[0] + n])))  # the stretch starts after its widest gap
    i0, i1 = idx[(j + 1) % len(idx)], idx[j]
    a, b = pts[i0] - pts[(i0 - m) % n], pts[(i1 + m) % n] - pts[i1]
    cos = float(np.dot(a, b)) / (float(np.linalg.norm(a) * np.linalg.norm(b)) or 1.0)
    return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))


def _split_circles(pieces: np.ndarray, k: int, min_area: float, stroke: float,
                   dx: int = 0, dy: int = 0) -> list[PixelCircle]:
    """Circles cut into sectors by centre lines: pieces that touch across a line thinner than k and whose
    hull is round. The pieces that reach the hull are its rim; the rest are tried again (a bore in a boss)."""
    circles = []
    n, groups, stats, _ = cv2.connectedComponentsWithStats(cv2.dilate(pieces, np.ones((k, k), np.uint8)))
    for g in range(1, n):
        x, y, bw, bh = (int(v) for v in stats[g, :4])
        members = np.where(groups[y: y + bh, x: x + bw] == g, pieces[y: y + bh, x: x + bw], 0).astype(np.uint8)
        count, labels = cv2.connectedComponents(members, connectivity=4)
        if count < 3:
            continue
        points = cv2.findNonZero(members)
        hull = cv2.convexHull(points)
        disk = np.zeros_like(members)
        cv2.fillConvexPoly(disk, hull, 255)
        if _round(points, cv2.countNonZero(members), SPLIT_FILL) and cv2.countNonZero(disk) >= min_area:
            circles.append(_circle(disk, x + dx, y + dy, stroke))
        rim = np.unique(labels[hull[:, 0, 1], hull[:, 0, 0]])
        inside = np.where(np.isin(labels, rim), 0, members).astype(np.uint8)
        if cv2.countNonZero(inside):
            circles += _split_circles(inside, k, min_area, stroke, x + dx, y + dy)
    return circles


def _runs(pieces: list[tuple[float, int, int, int]]) -> list[list[tuple[float, int, int, int]]]:
    """Pieces (across centre, start, length, thickness) of one line, split into runs with regular gaps."""
    runs, run, ref = [], [], None
    for p in sorted(pieces, key=lambda q: q[1]):
        if not run:
            run = [p]
            continue
        last = run[-1]
        gap = p[1] - (last[1] + last[2])
        if gap <= 0 or gap > 2 * max(p[2], last[2]):
            runs.append(run)
            run, ref = [p], None
        elif ref is None or abs(gap - ref) <= max(3.0, 0.5 * ref):
            ref = gap if ref is None else ref
            run.append(p)
        elif any(abs(gap - ref - m * (last[2] + ref)) <= max(3.0, 0.5 * ref) for m in (1, 2)):
            run.append(p)  # a dash or two lost where another line crosses
        else:
            runs.append(run)
            run, ref = [last, p], gap
    return runs + [run] if run else runs


def _is_dashed(run) -> bool:
    """Even dashes, of which only one end may be cut short. Short pieces between long dashes make a chain
    (centre) line, and a run of dots is no hidden line either."""
    lengths = [p[2] for p in run]
    core = lengths[1:-1]
    if min(core) * 1.5 + 2 < max(core) or np.median(lengths) < 2 * np.median([p[3] for p in run]):
        return False
    ends = lengths[0], lengths[-1]
    return all(e <= max(core) + 2 for e in ends) and sum(e < 0.5 * min(core) for e in ends) <= 1


def find_hidden_lines(ink: np.ndarray, bbox) -> list[tuple[str, float, float, float]]:
    """Dashed hidden edges (spec 3.3): runs of 3 or more collinear, axis-parallel short dashes with regular
    gaps, inside the bbox. Each is ("h" | "v", pos, start, end) in fractions of the bbox, measured from its
    left and top edges as in the image: "h" is a horizontal line at y = pos from x = start to x = end."""
    x0, y0, bw, bh = bbox
    thin = _spur_kernel(ink.shape, _stroke(ink))
    longest = DASH_MAX * max(bw, bh)
    _, _, stats, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
    boxes = [(x, y, w, h) for x, y, w, h, area in (tuple(int(v) for v in s) for s in stats[1:])
             if x >= x0 and y >= y0 and x + w <= x0 + bw and y + h <= y0 + bh
             and max(w, h) <= longest and area >= 0.5 * w * h]
    hidden = []
    for axis in "hv":
        size, origin, across_size, across_origin = (bw, x0, bh, y0) if axis == "h" else (bh, y0, bw, x0)
        pieces = []
        for x, y, w, h in boxes:
            start, length, across, thick = (x, w, y, h) if axis == "h" else (y, h, x, w)
            if thick <= thin and 2 * length >= thick:
                pieces.append((across + thick / 2, start, length, thick))
        lines, line = [], []
        for p in sorted(pieces):
            if line and p[0] - line[-1][0] > thin / 2:
                lines.append(line)
                line = []
            line.append(p)
        for run in (r for ln in lines + [line] for r in _runs(ln)):
            if len(run) >= DASH_RUN and _is_dashed(run):
                pos = float(np.median([p[0] for p in run]))
                a, b = run[0][1], run[-1][1] + run[-1][2]
                hidden.append((axis, float(np.clip((pos - across_origin) / across_size, 0, 1)),
                               float(np.clip((a - origin) / size, 0, 1)), float(np.clip((b - origin) / size, 0, 1))))
    return hidden


def find_visible_lines(ink: np.ndarray, body: np.ndarray, bbox, circles: list[PixelCircle],
                       stroke: float) -> list[tuple[str, float, float, float]]:
    """Long continuous axis-parallel visible edges (complex-parts spec 3), in the format of find_hidden_lines. Circles
    are erased first; dashes are shorter than a line and vanish; a segment that does not meet another line or the
    outline at both ends (a centre-line dash) is dropped, since an edge always ends on another edge."""
    x0, y0, bw, bh = bbox
    ink = ink.copy()
    pad = max(2, round(stroke)) + 2
    for c in circles:
        cv2.circle(ink, (round(c.cx), round(c.cy)), round(c.d / 2), 0, 2 * pad)
    length = max(9, round(LINE_MIN * max(bw, bh)))
    reach = max(pad, LINE_JOIN * max(bw, bh))
    rim = cv2.subtract(body, cv2.erode(body, np.ones((3, 3), np.uint8)))
    rim_pts = np.flip(np.argwhere(rim), 1).astype(np.float64)
    found = []
    for axis, kernel in (("h", (1, length)), ("v", (length, 1))):
        runs = cv2.morphologyEx(ink, cv2.MORPH_OPEN, np.ones(kernel, np.uint8))
        _, _, stats, _ = cv2.connectedComponentsWithStats(runs, connectivity=8)
        for x, y, w, h, _ in stats[1:]:
            if axis == "h":
                found.append(("h", y + h / 2, float(x), float(x + w), (x, y + h / 2), (x + w, y + h / 2)))
            else:
                found.append(("v", x + w / 2, float(y), float(y + h), (x + w / 2, y), (x + w / 2, y + h)))

    def meets(pt, own) -> bool:
        px, py = pt
        if len(rim_pts) and np.min(np.abs(rim_pts - (px, py)).max(1)) <= reach:
            return True
        for other in found:
            ax, pos, s, e = other[:4]
            if other is own or ax == own[0]:  # a collinear piece (the next dash of a chain line) is no junction
                continue
            u, v = (px, py) if ax == "h" else (py, px)
            if abs(v - pos) <= reach and s - reach <= u <= e + reach:
                return True
        return False

    lines = []
    for seg in found:
        if not (meets(seg[4], seg) and meets(seg[5], seg)):
            continue
        axis, pos, s, e = seg[:4]
        size, origin, across_size, across_origin = (bw, x0, bh, y0) if axis == "h" else (bh, y0, bw, x0)
        lines.append((axis, float(np.clip((pos - across_origin) / across_size, 0, 1)),
                      float(np.clip((s - origin) / size, 0, 1)), float(np.clip((e - origin) / size, 0, 1))))
    return lines


def to_face_mm(points_px, bbox, sa: float, sb: float) -> list[tuple[float, float]]:
    """Image pixels -> face-frame millimetres: a from the left of the bbox, b up from its bottom row."""
    x, y, _, h = bbox
    pts = np.asarray(points_px, np.float64).reshape(-1, 2)
    return [(float((u - x) * sa), float((y + h - 1 - v) * sb)) for u, v in pts]
