import cv2
import numpy as np

from s2c.multiview.outline import extract, resize_long_side, to_face_mm
from s2c.multiview.spec import MvAbstain


def page(h=1200, w=1600, color=255):
    return np.full((h, w, 3), color, np.uint8)


def test_sketch_rectangle_outline():
    img = page()
    cv2.rectangle(img, (400, 300), (1000, 700), (0, 0, 0), 4)
    o = extract(img)
    x, _, w, h = o.bbox
    assert abs(x - 400) <= 5 and abs(w - 600) <= 10 and abs(h - 400) <= 10
    assert o.circles == [] and o.inner == []


def test_drawn_circles_are_holes():
    img = page()
    cv2.rectangle(img, (400, 300), (1000, 700), (0, 0, 0), 4)
    cv2.circle(img, (500, 400), 40, (0, 0, 0), 3)
    cv2.circle(img, (900, 600), 40, (0, 0, 0), 3)
    o = extract(img)
    assert len(o.circles) == 2
    assert all(68 <= c.d <= 82 for c in o.circles)  # inner edge of the pen stroke around an 80 px circle


def test_photo_of_a_part_with_a_hole():
    img = page()
    cv2.rectangle(img, (400, 300), (1000, 700), (60, 60, 60), -1)
    cv2.circle(img, (500, 400), 30, (255, 255, 255), -1)
    o = extract(img)
    assert len(o.circles) == 1 and abs(o.circles[0].d - 60) <= 3


def test_a_slot_is_an_opening_not_a_circle():
    img = page()
    cv2.rectangle(img, (400, 300), (1000, 700), (60, 60, 60), -1)
    cv2.rectangle(img, (600, 480), (800, 520), (255, 255, 255), -1)
    o = extract(img)
    assert o.circles == [] and len(o.inner) == 1


def test_light_part_on_a_dark_background():
    img = page(color=30)
    cv2.rectangle(img, (400, 300), (1000, 700), (220, 220, 220), -1)
    assert abs(extract(img).bbox[2] - 600) <= 6


def test_blank_page_abstains():
    res = extract(page())
    assert isinstance(res, MvAbstain) and res.reason == "no_outline"


def test_the_edge_view_of_a_thin_washer_is_an_outline():
    img = page(1600, 1600)
    cv2.rectangle(img, (75, 785), (75 + 1450, 785 + 30), (128, 128, 128), -1)
    o = extract(img)
    _, _, w, h = o.bbox
    assert abs(w - 1450) <= 6 and abs(h - 30) <= 6


def test_the_edge_view_of_a_thin_l_shaped_part_is_an_outline():
    """A hinge leaf seen edge-on: a long strip with a short arm at one end. It fills little of its box, yet it is a
    solid band far wider than a pen stroke (benchmark hinge_003/004 abstained here)."""
    img = page(1600, 1600)
    cv2.rectangle(img, (760, 454), (760 + 26, 454 + 1074), (128, 128, 128), -1)  # the strip
    cv2.rectangle(img, (760, 454), (760 + 236, 454 + 26), (128, 128, 128), -1)   # the arm
    o = extract(img)
    assert not isinstance(o, MvAbstain), o
    _, _, w, h = o.bbox
    assert abs(w - 237) <= 6 and abs(h - 1075) <= 6


def test_a_small_speck_still_abstains():
    img = page(1600, 1600)
    cv2.rectangle(img, (780, 780), (820, 820), (60, 60, 60), -1)
    res = extract(img)
    assert isinstance(res, MvAbstain) and res.reason == "no_outline"


def test_a_strip_across_the_whole_photo_still_abstains():
    img = page(1600, 1600)
    cv2.rectangle(img, (0, 785), (1600, 785 + 30), (128, 128, 128), -1)
    res = extract(img)
    assert isinstance(res, MvAbstain) and res.reason == "no_outline"


def test_a_strip_touching_one_border_still_abstains():
    img = page(1600, 1600)
    cv2.rectangle(img, (0, 785), (1000, 785 + 30), (128, 128, 128), -1)
    res = extract(img)
    assert isinstance(res, MvAbstain) and res.reason == "no_outline"


def test_masked_region_is_ignored():
    img = page()
    cv2.rectangle(img, (100, 300), (500, 700), (60, 60, 60), -1)    # the part
    cv2.rectangle(img, (900, 100), (1550, 1100), (60, 60, 60), -1)  # a bigger object, masked out
    assert abs(extract(img, mask_out=[(880, 80, 700, 1050)]).bbox[0] - 100) <= 3


def test_resize_and_face_millimetres():
    assert resize_long_side(page(600, 800)).shape[:2] == (1200, 1600)
    pts = to_face_mm(np.array([[100, 300], [700, 100]]), (100, 100, 601, 201), 0.1, 0.2)
    assert pts == [(0.0, 0.0), (60.0, 40.0)]
