"""FacePocket and FaceBoss: the contract and the builder (complex-parts spec, section 2)."""
import math

import pytest

from s2c.multiview.build import build, volume
from tests.mv_helpers import make_spec


def test_a_blind_pocket_on_top_removes_its_box():
    spec = make_spec((40, 20, 30), features=[
        {"type": "pocket", "face": "top", "a_mm": 20, "b_mm": 15, "width_mm": 10, "height_mm": 8, "depth_mm": 5}])
    assert volume(build(spec)) == pytest.approx(40 * 20 * 30 - 10 * 8 * 5, rel=1e-4)


def test_a_corner_notch_is_a_pocket_running_off_the_outline():
    """A notch at the front-top-left corner: 8 wide, 6 tall, 12 deep from the front face."""
    spec = make_spec((40, 20, 30), features=[
        {"type": "pocket", "face": "front", "a_mm": 4, "b_mm": 17, "width_mm": 8, "height_mm": 6, "depth_mm": 12}])
    assert volume(build(spec)) == pytest.approx(40 * 20 * 30 - 8 * 6 * 12, rel=1e-4)


def test_a_through_pocket_is_a_window():
    spec = make_spec((40, 20, 30), features=[
        {"type": "pocket", "face": "front", "a_mm": 20, "b_mm": 10, "width_mm": 10, "height_mm": 6}])
    assert volume(build(spec)) == pytest.approx(40 * 20 * 30 - 10 * 6 * 30, rel=1e-4)


def test_a_boss_rounds_a_square_pin():
    """The hull of a pin seen as bumps in two views is a square prism; the boss keeps only its cylinder."""
    from tests.mv_helpers import outline, rect
    pin = [(0, 0), (40, 0), (40, 20), (25, 20), (25, 26), (15, 26), (15, 20), (0, 20)]   # front: bump 10 wide, 6 up
    side = [(0, 0), (30, 0), (30, 20), (20, 20), (20, 26), (10, 26), (10, 20), (0, 20)]  # right: bump 10 wide
    spec = make_spec((40, 26, 30), front=outline(pin), right=outline(side), top=outline(rect(40, 30)), features=[
        {"type": "boss", "face": "top", "a_mm": 20, "b_mm": 15, "diameter_mm": 10, "height_mm": 6}])
    body, pin_v = 40 * 20 * 30, math.pi * 5 ** 2 * 6
    assert volume(build(spec)) == pytest.approx(body + pin_v, rel=2e-3)


def test_the_new_features_need_provenance():
    from pydantic import ValidationError
    spec = make_spec((40, 20, 30), features=[
        {"type": "pocket", "face": "top", "a_mm": 20, "b_mm": 15, "width_mm": 10, "height_mm": 8, "depth_mm": 5}])
    data = spec.model_dump()
    del data["provenance"]["features[0].width_mm"]
    with pytest.raises(ValidationError):
        type(spec).model_validate(data)


def test_review_names_pockets_and_pins():
    from s2c.studio.handlers import field_label
    spec = make_spec((40, 26, 30), features=[
        {"type": "pocket", "face": "front", "a_mm": 4, "b_mm": 17, "width_mm": 8, "height_mm": 6, "depth_mm": 12},
        {"type": "boss", "face": "top", "a_mm": 20, "b_mm": 15, "diameter_mm": 8, "height_mm": 6}])
    assert field_label(spec, "features[0].height_mm") == "Pocket 1 (front) · height"
    assert field_label(spec, "features[1].diameter_mm") == "Pin 2 (top) · diameter"
