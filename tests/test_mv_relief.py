"""Notches, steps, pockets and windows read from the inner lines of exact line drawings (complex-parts spec 3)."""
import cadquery as cq
import cv2
import pytest

from s2c.multiview.build import build, volume
from s2c.multiview.pipeline import ImageInput, MvPipeline
from tests.builders import add_pins, cut_box, page_png, pockets_of, relief_spec, solid_block
from tests.line_views import draw_view

FACES = ("front", "top", "right")


def _true(part) -> float:
    return float(part.val().Volume())


def _left_in(solid, x0, y0, z0, x1, y1, z1, inset=0.05) -> float:
    """Material the built part keeps inside a box (shrunk by `inset`): 0 when a notch is fully open."""
    box = cq.Workplane("XY").box(x1 - x0 - 2 * inset, y1 - y0 - 2 * inset, z1 - z0 - 2 * inset, centered=False)
    return volume(solid.intersect(box.translate((x0 + inset, y0 + inset, z0 + inset))))


def test_a_corner_notch_is_read_and_built():
    part = cut_box(solid_block(), 0, 40, 50, 15, 60, 70)
    spec = relief_spec(part)
    (p,) = pockets_of(spec)
    assert p.face in ("front", "top", "left")
    built = build(spec)
    assert volume(built) == pytest.approx(_true(part), rel=0.03)
    assert any("notch" in w for w in spec.warnings)
    assert _left_in(built, 0.6, 40.6, 50.6, 14.4, 60, 70) < 1.0  # no skin left where the notch opens


def test_a_notch_under_pins_leaves_no_skin():
    """Pins make the envelope taller than the block, so the notch's top is not the envelope's: snapping its size to
    0.5 mm must not leave a sliver of the block's top over it."""
    part = add_pins(cut_box(solid_block(80, 60.8, 70), 0, 40, 50, 15, 60.8, 70), top=60.8)
    built = build(relief_spec(part))
    assert _left_in(built, 0.6, 40.6, 50.6, 14.4, 60.8, 70) < 1.0


def test_a_slot_across_the_top_front_is_read():
    """Front and top show it in lines; the right view sees it only as hidden lines. No outline changes."""
    part = cut_box(solid_block(), 20, 40, 40, 60, 60, 70)
    spec = relief_spec(part)
    assert pockets_of(spec)
    assert volume(build(spec)) == pytest.approx(_true(part), rel=0.03)


def test_a_blind_pocket_takes_its_depth_from_the_hidden_lines():
    part = cut_box(solid_block(), 20, 45, 20, 60, 60, 50)
    spec = relief_spec(part)
    (p,) = pockets_of(spec)
    assert p.face == "top" and p.depth_mm == pytest.approx(15, abs=1.5)
    assert volume(build(spec)) == pytest.approx(_true(part), rel=0.03)


def test_a_through_window_is_through():
    part = cut_box(solid_block(), 25, 20, 0, 55, 40, 70)
    spec = relief_spec(part)
    (p,) = pockets_of(spec)
    assert p.depth_mm is None
    assert volume(build(spec)) == pytest.approx(_true(part), rel=0.03)


def test_two_notches_and_a_step():
    part = cut_box(cut_box(cut_box(solid_block(), 0, 40, 50, 15, 60, 70), 65, 40, 50, 80, 60, 70), 0, 0, 0, 80, 15, 20)
    spec = relief_spec(part)
    assert volume(build(spec)) == pytest.approx(_true(part), rel=0.03)


def test_a_plain_block_has_no_pockets():
    spec = relief_spec(solid_block())
    assert pockets_of(spec) == []
    assert not any("notch" in w for w in spec.warnings)


def test_filled_renders_never_get_pockets():
    """A render's faces are filled; relief runs only on line art."""
    part = cut_box(solid_block(), 0, 40, 50, 15, 60, 70)
    pipe = MvPipeline()
    images = []
    for f in FACES:
        ink = draw_view(part, f, 4.0, hidden=False)
        filled = ink.copy()
        contours, _ = cv2.findContours(ink, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(filled, contours, -1, 140, -1)
        images.append(ImageInput(page_png(filled), f, "drawing"))
    observed = pipe.observe(images)
    assert not any(o.line_art for o in observed.observations)
    spec = pipe.fuse(observed, {"envelope.x_mm": 80, "envelope.y_mm": 60, "envelope.z_mm": 70})
    assert pockets_of(spec) == []


def _section_area(solid, y) -> float:
    """Area of the built part cut by the plane at height y."""
    bb = solid.val().BoundingBox()
    slab = cq.Workplane("XY").box(bb.xlen + 2, 0.01, bb.zlen + 2, centered=False).translate((bb.xmin - 1, y, bb.zmin - 1))
    return solid.intersect(slab).val().Volume() / 0.01


def test_pins_come_out_round():
    part = add_pins(solid_block())
    spec = relief_spec(part)
    pins = [f for f in spec.features if f.type == "boss"]
    assert len(pins) == 2 and all(f.face == "top" for f in pins)
    assert all(f.diameter_mm == pytest.approx(8, abs=0.6) and f.height_mm == pytest.approx(6, abs=0.6) for f in pins)
    assert not [f for f in spec.features if f.type == "hole"]
    built = build(spec)
    assert _section_area(built, 63) == pytest.approx(2 * 3.14159 * 16, rel=0.12)  # two discs, not two squares


def test_a_block_with_notches_pins_and_a_side_hole():
    """The layout of a real user sheet: two front corner notches, two pins on top, a hole in a notch wall."""
    part = cut_box(cut_box(solid_block(), 0, 40, 50, 15, 60, 70), 65, 40, 50, 80, 60, 70)
    part = add_pins(part, z=20)
    part = part.cut(cq.Workplane("YZ", origin=(65, 0, 0)).center(50, 60).circle(3).extrude(-20))
    spec = relief_spec(part)
    kinds = sorted(f.type for f in spec.features)
    assert kinds.count("boss") == 2 and kinds.count("pocket") >= 2, kinds
    assert volume(build(spec)) == pytest.approx(_true(part), rel=0.03)


def test_chamfered_pins_drawn_as_two_circles_are_pins():
    """A pin with a chamfered tip shows two concentric circles from above; the outer one is the pin."""
    part = solid_block()
    for x in (20, 60):
        pin = cq.Workplane("XZ", origin=(0, 60, 0)).center(x, 15).circle(4).extrude(-6).faces(">Y").chamfer(1.2)
        part = part.union(pin)
    spec = relief_spec(part)
    pins = [f for f in spec.features if f.type == "boss"]
    assert len(pins) == 2, spec.warnings
    assert all(f.diameter_mm == pytest.approx(8, abs=0.6) for f in pins)
    assert not [f for f in spec.features if f.type == "hole"]


def test_carving_never_splits_the_part():
    """A drawing a carve could match by cutting the part in two (the middle drawn as empty): it stays one piece."""
    import numpy as np

    from s2c.multiview.relief import ReliefEvidence, carve
    occ = np.ones((3, 1, 1), bool)
    obs_v = np.ones((4, 1), bool)                                   # every vertical grid edge drawn
    obs_h = np.array([[True, True], [False, False], [True, True]])  # the middle column has no top or bottom
    ev = ReliefEvidence("front", obs_v, obs_h, np.zeros_like(obs_v), np.zeros_like(obs_h),
                   np.ones(obs_v.shape), np.ones(obs_h.shape))
    carved = carve(occ, [ev])[0]
    assert carved[0].any() and carved[1].any() and carved[2].any()
