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
