"""Dimensioned sheets: annotations stay out of the views; dimensions give the sheet's scale (sheet-reading spec)."""
import cv2
import numpy as np
import pytest

from s2c.multiview.sheet import crop_views, name_views, split_sheet, view_body
from tests.line_views import drawing_sheet
from tests.test_mv_relief import _block, _cut, _pins


def _part():
    return _pins(_cut(_cut(_block(), 0, 40, 50, 15, 60, 70), 65, 40, 50, 80, 60, 70), z=20)


@pytest.fixture(scope="module")
def third():
    return drawing_sheet(_part(), layout="third", faces=("front", "top", "right"), iso=True)


def _close(a, b, tol=6):
    return all(abs(p - q) <= tol for p, q in zip(a, b, strict=True))


def test_a_view_body_leaves_its_dimensions_out(third):
    img, boxes, _ = third
    sheet = split_sheet(img)
    ink = (cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) < 128).astype(np.uint8) * 255
    for view in sheet.drawings[0].views:
        body, _ = view_body(ink, view.box, max(img.shape[:2]))
        hits = [f for f, b in boxes.items() if _close(body, b)]
        assert hits or body[2] * body[3] < 2000, (view.box, body)


def test_named_views_of_a_dimensioned_sheet(third):
    img, boxes, _ = third
    sheet = split_sheet(img)
    naming = name_views(sheet, img, "third")
    faces = sorted(f for f in naming.faces if f not in ("auto", "skip"))
    assert faces == ["front", "right", "top"], naming.faces


def test_crops_hold_no_dimension_lines(third):
    img, boxes, _ = third
    sheet = split_sheet(img)
    naming = name_views(sheet, img, "third")
    for png, face in crop_views(sheet, img, naming):
        if face in ("auto", "skip"):
            continue
        crop = cv2.imdecode(np.frombuffer(png, np.uint8), cv2.IMREAD_GRAYSCALE)
        ys, xs = np.nonzero(crop < 128)
        w, h = xs.max() - xs.min() + 1, ys.max() - ys.min() + 1
        tw, th = boxes[face][2], boxes[face][3]
        assert abs(w - tw) <= 6 and abs(h - th) <= 6, (face, (w, h), (tw, th))
