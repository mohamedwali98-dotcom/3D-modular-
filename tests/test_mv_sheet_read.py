"""A drawing sheet read end to end: views named (projection chosen by the drawing), dimensions read, part built
with no typed size (sheet-reading spec 2026-10-01)."""
from pathlib import Path

import cv2
import numpy as np
import pytest

from s2c.multiview import spec as S
from s2c.multiview.build import build, volume
from s2c.multiview.pipeline import MvPipeline
from s2c.multiview.sheet_read import link_diameters, read_sheet
from tests.builders import add_pins, cut_box, solid_block, word_service
from tests.line_views import draw_text, drawing_sheet

THIRD = ("front", "top", "right")
FIRST = ("front", "top", "left")


def _part():
    return add_pins(cut_box(cut_box(solid_block(), 0, 40, 50, 15, 60, 70), 65, 40, 50, 80, 60, 70), z=20)


def _faces(read):
    return sorted(c.face for c in read.crops)


def test_a_third_angle_sheet_is_read_as_third_angle_with_its_picture_left_out():
    img, _, words = drawing_sheet(_part(), faces=THIRD, layout="third", iso=True)
    read = read_sheet(img, "auto", service=word_service(words))
    assert read is not None
    assert _faces(read) == sorted(THIRD)
    assert (read.naming.projection, read.naming.projection_source) == ("third", "drawing")
    assert read.scale.mm_per_px == pytest.approx(0.25, rel=0.01)
    assert any("left out" in w for w in read.warnings)
    assert not any("pick the face" in w for w in read.warnings)  # the web app has no face picker


def test_a_first_angle_sheet_stays_first_angle():
    img, _, words = drawing_sheet(_part(), faces=FIRST, layout="first", iso=True)
    read = read_sheet(img, "auto", service=word_service(words))
    assert _faces(read) == sorted(FIRST)
    assert read.naming.projection == "first"


def test_a_symmetric_part_keeps_the_iso_convention():
    """A plain block reads the same either way: the view above the front is the bottom view (ISO first-angle)."""
    img, _, words = drawing_sheet(solid_block(), faces=THIRD, layout="third")
    read = read_sheet(img, "auto", service=word_service(words))
    assert read.naming.projection == "first" and read.naming.projection_source == "setting"
    assert _faces(read) == ["bottom", "front", "left"]


def test_the_sheet_builds_with_no_typed_size():
    part = _part()
    img, _, words = drawing_sheet(part, faces=THIRD, layout="third", iso=True)
    read = read_sheet(img, "auto", service=word_service(words))
    pipe = MvPipeline()
    observed = pipe.observe(read.inputs())
    link_diameters(read, observed)
    spec = pipe.fuse(observed)
    assert isinstance(spec, S.MultiViewSpec), spec
    env = spec.envelope
    assert (env.x_mm, env.y_mm, env.z_mm) == pytest.approx((80, 66, 70), rel=0.02)
    assert volume(build(spec)) == pytest.approx(part.val().Volume(), rel=0.03)


def test_a_diameter_written_by_a_hole_is_its_size():
    part = solid_block().faces(">Z").workplane(centerOption="CenterOfBoundBox").hole(12)
    img, boxes, words = drawing_sheet(part, faces=THIRD, layout="third", gap=220)
    ink = 255 - cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    fx, fy, fw, fh = boxes["front"]
    cx, cy = fx + fw // 2, fy + fh // 2  # the hole, Ø12 at 4 px/mm: radius 24 px
    box = draw_text(ink, "D12", fx + fw + 20, cy - 30, 22, False)   # written outside the view, a leader to the hole
    cv2.line(ink, (cx + 24, cy), (fx + fw + 18, cy - 8), 255, 1)
    img = cv2.cvtColor(255 - ink, cv2.COLOR_GRAY2BGR)
    read = read_sheet(img, "third", service=word_service([*words, (box, "D12")]))
    pipe = MvPipeline()
    observed = pipe.observe(read.inputs())
    link_diameters(read, observed)
    spec = pipe.fuse(observed)
    holes = [f for f in spec.features if f.type == "hole"]
    assert holes and all(f.diameter_mm == pytest.approx(12) for f in holes if f.face == "front")
    k = next(i for i, f in enumerate(spec.features) if f.type == "hole" and f.face == "front")
    assert spec.provenance[f"features[{k}].diameter_mm"] == "user_written"


def test_without_a_reader_the_sheet_still_splits_and_names():
    img, _, _ = drawing_sheet(_part(), faces=THIRD, layout="third", iso=True)
    read = read_sheet(img, "auto")
    assert read.scale.mm_per_px is None and _faces(read) == sorted(THIRD)
    assert np.all([c.png for c in read.crops])


def _hole_sheet(callout: str):
    part = solid_block().faces(">Z").workplane(centerOption="CenterOfBoundBox").hole(12)
    img, boxes, words = drawing_sheet(part, faces=THIRD, layout="third", gap=220)
    ink = 255 - cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    fx, fy, fw, fh = boxes["front"]
    cx, cy = fx + fw // 2, fy + fh // 2
    box = draw_text(ink, callout, fx + fw + 20, cy - 30, 22, False)
    cv2.line(ink, (cx + 24, cy), (fx + fw + 18, cy - 8), 255, 1)
    return cv2.cvtColor(255 - ink, cv2.COLOR_GRAY2BGR), [*words, (box, callout)]


def _front_hole(callout: str):
    img, words = _hole_sheet(callout)
    read = read_sheet(img, "third", service=word_service(words))
    pipe = MvPipeline()
    observed = pipe.observe(read.inputs())
    link_diameters(read, observed)
    spec = pipe.fuse(observed)
    k = next(i for i, f in enumerate(spec.features) if f.type == "hole" and f.face == "front")
    return spec, k


def test_a_radius_is_never_a_hole_size():
    spec, k = _front_hole("R5")
    assert spec.features[k].diameter_mm == pytest.approx(12, abs=0.6)
    assert spec.provenance[f"features[{k}].diameter_mm"] == "measured"


def test_a_written_diameter_far_from_the_drawn_hole_is_flagged():
    spec, k = _front_hole("D20")
    assert spec.provenance[f"features[{k}].diameter_mm"] != "user_written"
    assert any("D20" in w for w in spec.warnings)


def test_has_views_finds_a_sheet_and_refuses_a_single_outline():
    from s2c.multiview.sheet_read import has_views
    sheet = cv2.imread(str(Path(__file__).resolve().parents[1] / "examples" / "mv" / "sheet" / "sheet.png"))
    assert has_views(sheet)
    one = np.full((400, 400, 3), 255, np.uint8)
    cv2.rectangle(one, (100, 120), (300, 280), (0, 0, 0), 3)
    assert not has_views(one)
