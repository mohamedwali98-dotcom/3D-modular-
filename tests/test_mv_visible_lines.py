"""Visible inner lines of a line drawing (complex-parts spec, section 3)."""
import cadquery as cq
import cv2
import numpy as np

from s2c.multiview.outline import extract, resize_long_side
from tests.line_views import draw_view


def _page(ink: np.ndarray, margin: int = 40) -> np.ndarray:
    ink = cv2.copyMakeBorder(ink, margin, margin, margin, margin, cv2.BORDER_CONSTANT, value=0)
    return resize_long_side(cv2.cvtColor(255 - ink, cv2.COLOR_GRAY2BGR))


def _notched() -> cq.Workplane:
    """80 x 60 x 70 block with a 15 x 20 notch at the front-top-left corner, 20 deep."""
    blk = cq.Workplane("XY").box(80, 60, 70, centered=False)
    return blk.cut(cq.Workplane("XY").box(15, 20, 20, centered=False).translate((0, 40, 50)))


def _near(lines, axis, pos, start, end, tol=0.03):
    return any(a == axis and abs(p - pos) < tol and abs(s - start) < tol and abs(e - end) < tol for a, p, s, e in lines)


def test_the_notch_lines_are_found():
    o = extract(_page(draw_view(_notched(), "front", 4.0)), drawing=True)
    assert o.line_art
    # notch: x 0..15 of 80, y 40..60 of 60 -> image rows from the top: 0..20 of 60
    assert _near(o.lines, "h", 20 / 60, 0, 15 / 80), o.lines
    assert _near(o.lines, "v", 15 / 80, 0, 20 / 60), o.lines
    assert _near(o.lines, "h", 1.0, 0, 1.0)  # the outline is drawn too


def test_dashes_circles_and_centre_lines_are_not_visible_lines():
    blk = cq.Workplane("XY").box(80, 60, 70, centered=False)
    blk = blk.cut(cq.Workplane("XY").box(30, 20, 70, centered=False).translate((25, 0, 0)).translate((0, 0, -50)))
    blk = blk.faces(">Z").workplane(centerOption="CenterOfBoundBox").center(-20, 10).hole(16)
    ink = draw_view(blk, "front", 4.0)
    h, w = ink.shape
    cy, cx = h // 2 - round(10 * 4.0), w // 2 - round(20 * 4.0)
    for x in range(cx - 70, cx + 70, 30):  # a chain centre line through the hole: long and short dashes
        cv2.line(ink, (x, cy), (x + 18, cy), 255, 1)
        cv2.line(ink, (x + 22, cy), (x + 25, cy), 255, 1)
    o = extract(_page(ink), drawing=True)
    assert o.line_art and len(o.circles) == 1
    inner = [ln for ln in o.lines if 0.02 < ln[1] < 0.98]
    assert inner == [], inner
