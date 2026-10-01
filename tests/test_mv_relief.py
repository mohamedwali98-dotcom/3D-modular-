"""Notches, steps, pockets and windows read from the inner lines of exact line drawings (complex-parts spec 3)."""
import cadquery as cq
import cv2
import pytest

from s2c.multiview import spec as S
from s2c.multiview.build import build, volume
from s2c.multiview.pipeline import ImageInput, MvPipeline
from tests.line_views import draw_view

FACES = ("front", "top", "right")


def _page(ink, margin=40) -> bytes:
    ink = cv2.copyMakeBorder(ink, margin, margin, margin, margin, cv2.BORDER_CONSTANT, value=0)
    return cv2.imencode(".png", 255 - ink)[1].tobytes()


def _block(x=80, y=60, z=70) -> cq.Workplane:
    return cq.Workplane("XY").box(x, y, z, centered=False)


def _cut(part, x0, y0, z0, x1, y1, z1) -> cq.Workplane:
    return part.cut(cq.Workplane("XY").box(x1 - x0, y1 - y0, z1 - z0, centered=False).translate((x0, y0, z0)))


def _spec(part, faces=FACES, kind="drawing", hidden=True) -> S.MultiViewSpec:
    pipe = MvPipeline()
    observed = pipe.observe([ImageInput(_page(draw_view(part, f, 4.0, hidden=hidden)), f, kind) for f in faces])
    bb = part.val().BoundingBox()
    spec = pipe.fuse(observed, {"envelope.x_mm": round(bb.xlen, 3), "envelope.y_mm": round(bb.ylen, 3),
                                "envelope.z_mm": round(bb.zlen, 3)})
    assert isinstance(spec, S.MultiViewSpec), spec
    return spec


def _pockets(spec):
    return [f for f in spec.features if f.type == "pocket"]


def _true(part) -> float:
    return float(part.val().Volume())


def test_a_corner_notch_is_read_and_built():
    part = _cut(_block(), 0, 40, 50, 15, 60, 70)
    spec = _spec(part)
    (p,) = _pockets(spec)
    assert p.face in ("front", "top", "left")
    assert volume(build(spec)) == pytest.approx(_true(part), rel=0.03)
    assert any("notch" in w for w in spec.warnings)


def test_a_slot_across_the_top_front_is_read():
    """Front and top show it in lines; the right view sees it only as hidden lines. No outline changes."""
    part = _cut(_block(), 20, 40, 40, 60, 60, 70)
    spec = _spec(part)
    assert _pockets(spec)
    assert volume(build(spec)) == pytest.approx(_true(part), rel=0.03)


def test_a_blind_pocket_takes_its_depth_from_the_hidden_lines():
    part = _cut(_block(), 20, 45, 20, 60, 60, 50)
    spec = _spec(part)
    (p,) = _pockets(spec)
    assert p.face == "top" and p.depth_mm == pytest.approx(15, abs=1.5)
    assert volume(build(spec)) == pytest.approx(_true(part), rel=0.03)


def test_a_through_window_is_through():
    part = _cut(_block(), 25, 20, 0, 55, 40, 70)
    spec = _spec(part)
    (p,) = _pockets(spec)
    assert p.depth_mm is None
    assert volume(build(spec)) == pytest.approx(_true(part), rel=0.03)


def test_two_notches_and_a_step():
    part = _cut(_cut(_cut(_block(), 0, 40, 50, 15, 60, 70), 65, 40, 50, 80, 60, 70), 0, 0, 0, 80, 15, 20)
    spec = _spec(part)
    assert volume(build(spec)) == pytest.approx(_true(part), rel=0.03)


def test_a_plain_block_has_no_pockets():
    spec = _spec(_block())
    assert _pockets(spec) == []
    assert not any("notch" in w for w in spec.warnings)


def test_filled_renders_never_get_pockets():
    """A render's faces are filled; relief runs only on line art."""
    part = _cut(_block(), 0, 40, 50, 15, 60, 70)
    pipe = MvPipeline()
    images = []
    for f in FACES:
        ink = draw_view(part, f, 4.0, hidden=False)
        filled = ink.copy()
        contours, _ = cv2.findContours(ink, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(filled, contours, -1, 140, -1)
        images.append(ImageInput(_page(filled), f, "drawing"))
    observed = pipe.observe(images)
    assert not any(o.line_art for o in observed.observations)
    spec = pipe.fuse(observed, {"envelope.x_mm": 80, "envelope.y_mm": 60, "envelope.z_mm": 70})
    assert _pockets(spec) == []


def _pins(part, xs=(20, 60), z=15, d=8, h=6, top=60):
    for x in xs:
        part = part.union(cq.Workplane("XZ", origin=(0, top, 0)).center(x, z).circle(d / 2).extrude(-h))
    return part


def _section_area(solid, y) -> float:
    """Area of the built part cut by the plane at height y."""
    bb = solid.val().BoundingBox()
    slab = cq.Workplane("XY").box(bb.xlen + 2, 0.01, bb.zlen + 2, centered=False).translate((bb.xmin - 1, y, bb.zmin - 1))
    return solid.intersect(slab).val().Volume() / 0.01


def test_pins_come_out_round():
    part = _pins(_block())
    spec = _spec(part)
    pins = [f for f in spec.features if f.type == "boss"]
    assert len(pins) == 2 and all(f.face == "top" for f in pins)
    assert all(f.diameter_mm == pytest.approx(8, abs=0.6) and f.height_mm == pytest.approx(6, abs=0.6) for f in pins)
    assert not [f for f in spec.features if f.type == "hole"]
    built = build(spec)
    assert _section_area(built, 63) == pytest.approx(2 * 3.14159 * 16, rel=0.12)  # two discs, not two squares


def test_a_block_with_notches_pins_and_a_side_hole():
    """The layout of a real user sheet: two front corner notches, two pins on top, a hole in a notch wall."""
    part = _cut(_cut(_block(), 0, 40, 50, 15, 60, 70), 65, 40, 50, 80, 60, 70)
    part = _pins(part, z=20)
    part = part.cut(cq.Workplane("YZ", origin=(65, 0, 0)).center(50, 60).circle(3).extrude(-20))
    spec = _spec(part)
    kinds = sorted(f.type for f in spec.features)
    assert kinds.count("boss") == 2 and kinds.count("pocket") >= 2, kinds
    assert volume(build(spec)) == pytest.approx(_true(part), rel=0.03)


def test_chamfered_pins_drawn_as_two_circles_are_pins():
    """A pin with a chamfered tip shows two concentric circles from above; the outer one is the pin."""
    part = _block()
    for x in (20, 60):
        pin = cq.Workplane("XZ", origin=(0, 60, 0)).center(x, 15).circle(4).extrude(-6).faces(">Y").chamfer(1.2)
        part = part.union(pin)
    spec = _spec(part)
    pins = [f for f in spec.features if f.type == "boss"]
    assert len(pins) == 2, spec.warnings
    assert all(f.diameter_mm == pytest.approx(8, abs=0.6) for f in pins)
    assert not [f for f in spec.features if f.type == "hole"]
