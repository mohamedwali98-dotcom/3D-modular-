"""One sketch to a 3D model in five steps: page, faces, labels, generate the rest, numbers, model
(sketch-to-model spec 2026-10-02)."""
from pathlib import Path

import cv2
import numpy as np

from s2c.multiview import spec as S
from s2c.multiview.pipeline import MvPipeline
from s2c.multiview.sheet import split_by_outlines
from s2c.multiview.sheet_read import page_of, read_drawing, read_sheet
from tests.hand_views import hand_photo
from tests.line_views import drawing_sheet
from tests.test_mv_relief import _block, _cut, _pins
from tests.test_mv_sheet_read import _service

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


def test_views_joined_by_dimension_and_miter_lines_are_three_faces():
    photo, _ = hand_photo(_part(), faces=THIRD, layout="third", miter=True)
    page = page_of(photo)
    sheet = split_by_outlines(page.image, page.stroke_px)
    big = [v for d in sheet.drawings for v in d.views if v.box[2] * v.box[3] > 0.02 * page.image.size / 3]
    assert len(big) == 3


def test_the_real_sketch_has_three_faces():
    page = page_of(cv2.imread(str(GOLDEN / "real_bracket_1" / "image.jpg")))
    sheet = split_by_outlines(page.image, page.stroke_px)
    area = page.image.shape[0] * page.image.shape[1]
    big = [v for d in sheet.drawings for v in d.views if v.box[2] * v.box[3] > 0.01 * area]
    assert len(big) == 3


def test_dimension_strips_are_not_part_of_a_face():
    """A dimension line whose extension lines meet the outline closes a strip onto the view; with its number beside
    it, the strip is taken off again, so views that share a size have the same box size."""
    for seed in range(3):
        photo, _ = hand_photo(_part(), faces=THIRD, layout="third", miter=True, seed=seed)
        page = page_of(photo)
        views = [v.box for d in split_by_outlines(page.image, page.stroke_px).drawings for v in d.views]
        top = min(views, key=lambda b: b[1])
        front, right = sorted((b for b in views if b != top), key=lambda b: b[0])
        assert abs(top[2] - front[2]) <= 0.03 * front[2]
        assert abs(front[3] - right[3]) <= 0.03 * front[3]


def test_a_hand_sketch_is_read_into_named_faces():
    photo, _ = hand_photo(_part(), faces=THIRD, layout="third", miter=True)
    read = read_drawing(photo, "third")
    assert read.kind == "sketch" and sorted(c.face for c in read.crops) == sorted(THIRD)


def test_a_clean_drawing_reads_exactly_as_before():
    img, _, words = drawing_sheet(_part(), faces=THIRD, layout="third", iso=True)
    a = read_sheet(img, "auto", service=_service(words))
    b = read_drawing(img, "auto", service=_service(words))
    assert b.kind == "drawing" and [c.face for c in a.crops] == [c.face for c in b.crops]
    assert a.scale.mm_per_px == b.scale.mm_per_px


def test_the_real_sketch_names_its_three_faces():
    """Third-angle set: top, front, right. On auto, the ISO default with the setting as the source is accepted
    when the drawing does not decide (spec 2)."""
    img = cv2.imread(str(GOLDEN / "real_bracket_1" / "image.jpg"))
    third = read_drawing(img, "third")
    assert sorted(f for f in third.naming.faces if f != "auto") == sorted(THIRD)
    auto = read_drawing(img, "auto")
    faces = sorted(auto.naming.faces)
    assert faces == sorted(THIRD) or (faces == sorted(("bottom", "front", "left"))
                                      and auto.naming.projection_source == "setting")


def test_a_wobbly_notched_block_reads_its_notch():
    part = _cut(_block(), 0, 40, 50, 15, 60, 70)
    for seed in range(3):  # three photos: wobble, slant and the dash pattern differ
        photo, _ = hand_photo(part, faces=THIRD, layout="third", seed=seed)
        read = read_drawing(photo, "third")
        pipe = MvPipeline()
        spec = pipe.fuse(pipe.observe(read.inputs()), {"envelope.x_mm": 80, "envelope.y_mm": 60, "envelope.z_mm": 70})
        pockets = [f for f in spec.features if f.type == "pocket"]
        assert len(pockets) == 1 and abs(pockets[0].depth_mm - 20) <= 2, seed


def test_straightening_keeps_a_hole_round():
    part = _block().faces(">Z").workplane(centerOption="CenterOfBoundBox").hole(12)
    photo, _ = hand_photo(part, faces=THIRD, layout="third")
    read = read_drawing(photo, "third")
    obs = MvPipeline().observe(read.inputs())
    front = next(o for o in obs.observations if o.face == "front")
    assert len(front.outline.circles) == 1
