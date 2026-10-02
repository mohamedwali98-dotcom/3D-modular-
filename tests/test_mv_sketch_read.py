"""One sketch to a 3D model in five steps: page, faces, labels, generate the rest, numbers, model
(sketch-to-model spec 2026-10-02)."""
from pathlib import Path

import cv2
import numpy as np

from s2c.multiview import spec as S
from s2c.multiview.sheet_read import page_of
from tests.hand_views import hand_photo
from tests.line_views import drawing_sheet
from tests.test_mv_relief import _block, _cut, _pins

THIRD = ("front", "top", "right")
GOLDEN = Path(__file__).resolve().parent / "golden_sketch"


def _part():
    return _pins(_cut(_cut(_block(), 0, 40, 50, 15, 60, 70), 65, 40, 50, 80, 60, 70), z=20)


def test_a_clean_drawing_is_used_as_drawn():
    img, _, _ = drawing_sheet(_part(), faces=THIRD, layout="third")
    page = page_of(img)
    assert page.kind == "drawing" and np.array_equal(page.image, img)


def test_a_hand_photo_becomes_a_clean_page():
    photo, _ = hand_photo(_part(), faces=THIRD, layout="third")
    page = page_of(photo)
    assert page.kind == "sketch" and max(page.image.shape[:2]) == 1600
    gray = cv2.cvtColor(page.image, cv2.COLOR_BGR2GRAY)
    assert set(np.unique(gray)) <= {0, 255}          # black ink on white
    corner = cv2.perspectiveTransform(np.float32([[[0, 0]]]), page.to_photo)[0, 0]
    assert 0 <= corner[0] < photo.shape[1] and 0 <= corner[1] < photo.shape[0]


def test_a_dark_photo_abstains_with_a_remedy():
    page = page_of(np.full((900, 1200, 3), 20, np.uint8))
    assert isinstance(page, S.MvAbstain) and page.remedy
