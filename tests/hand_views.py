"""Photo-like hand sketches for the sketch-to-model tests: an exact dimensioned line sheet (line_views), made
hand-like (wobbly, slightly turned strokes, script digits, pen colour), then photographed (pasted on a desk, seen
at an angle, under a shadow). Word boxes follow every transform so a stand-in reader can read them."""
from __future__ import annotations

import cadquery as cq
import cv2
import numpy as np

from tests.line_views import dashed_line, dimension, drawing_sheet

PAPER = (230, 233, 235)
PEN = (70, 30, 20)
DESK = (50, 70, 90)


def _miter(ink: np.ndarray, boxes: dict, layout: str) -> None:
    """The 45 degree construction line between the plan view and the side view, dashed, in their free corner."""
    side = boxes.get("right") or boxes.get("left")
    plan = boxes.get("top") or boxes.get("bottom")
    if side is None or plan is None:
        return
    length = 0.8 * min(side[2], plan[3])
    x0 = side[0] - 25
    if layout == "third":   # the plan view above the front: the free corner is up and to the right
        y0 = plan[1] + plan[3] + 25
        end = (x0 + length, y0 - length)
    else:                   # the plan view below the front
        y0 = plan[1] - 25
        end = (x0 + length, y0 + length)
    dashed_line(ink, np.array([[x0, y0], end], float), 2, 18, 10)


def _extra(ink: np.ndarray, boxes: dict, extra, font) -> list:
    """More dimensions, between two points of a view's top edge: (face, start_mm, end_mm, axis, text) at 4 px/mm,
    drawn above the view."""
    words = []
    for face, start, end, axis, text in extra:
        x, y, _, _ = boxes[face]
        pad = 3
        if axis == "a":
            p0, p1 = (x + pad + round(start * 4), y + pad), (x + pad + round(end * 4), y + pad)
            words.append((dimension(ink, p0, p1, -45, text, font=font), text))
    return words


def hand_photo(part: cq.Workplane, faces=("front", "top", "right"), layout: str = "third", dims: bool = True,
               seed: int = 0, miter: bool = False, extra=()):
    """The part as a hand sketch on a photographed page. Returns the BGR photo and the words on it: [(box, text)]."""
    font = cv2.FONT_HERSHEY_SCRIPT_SIMPLEX
    sheet, boxes, words = drawing_sheet(part, faces=faces, layout=layout, dims=dims, gap=140, font=font)
    ink = 255 - cv2.cvtColor(sheet, cv2.COLOR_BGR2GRAY)
    if miter:
        _miter(ink, boxes, layout)
    words = words + _extra(ink, boxes, extra, font)
    rng = np.random.default_rng(seed)
    h, w = ink.shape
    pad = 60
    ink = cv2.copyMakeBorder(ink, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)
    h, w = ink.shape
    # wobble: a smooth displacement field of about 2.5 px
    field = [cv2.GaussianBlur(rng.normal(size=(h, w)).astype(np.float32), (0, 0), 25) for _ in range(2)]
    field = [f * (2.5 / max(1e-6, float(np.abs(f).max()))) for f in field]
    gx, gy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    ink = cv2.remap(ink, gx + field[0], gy + field[1], cv2.INTER_LINEAR)
    turn = cv2.getRotationMatrix2D((w / 2, h / 2), 1.5, 1.0)
    ink = cv2.warpAffine(ink, turn, (w, h))
    alpha = (ink.astype(np.float32) / 255)[..., None]
    page = (np.array(PAPER, np.float32) * (1 - alpha) + np.array(PEN, np.float32) * alpha).astype(np.uint8)
    # the photo: the page on a desk, seen at an angle, under a shadow
    W, H = round(w * 1.3), round(h * 1.3)
    ox, oy = (W - w) // 2, (H - h) // 2
    corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    jitter = rng.uniform(-0.04, 0.04, size=(4, 2)).astype(np.float32) * np.float32([w, h])
    target = corners + np.float32([ox, oy]) + jitter
    persp = cv2.getPerspectiveTransform(corners, target)
    photo = np.full((H, W, 3), DESK, np.uint8)
    warped = cv2.warpPerspective(page, persp, (W, H), borderMode=cv2.BORDER_CONSTANT, borderValue=DESK)
    mask = cv2.warpPerspective(np.full((h, w), 255, np.uint8), persp, (W, H)) > 0
    photo[mask] = warped[mask]
    shade = np.linspace(1.0, 0.6, W, dtype=np.float32)[None, :, None]
    photo = (photo.astype(np.float32) * shade).clip(0, 255).astype(np.uint8)

    def move(box):
        x, y, bw, bh = box
        pts = np.float32([[x, y], [x + bw, y], [x + bw, y + bh], [x, y + bh]]) + pad
        pts = cv2.transform(pts[None], turn)[0]
        pts = cv2.perspectiveTransform(pts[None], persp)[0]
        x0, y0 = pts.min(0) - 3
        x1, y1 = pts.max(0) + 3
        return int(x0), int(y0), int(x1 - x0), int(y1 - y0)

    return photo, [(move(b), t) for b, t in words]
