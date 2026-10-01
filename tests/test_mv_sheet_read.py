"""A drawing sheet read end to end: views named (projection chosen by the drawing), dimensions read, part built
with no typed size (sheet-reading spec 2026-10-01)."""
import cv2
import numpy as np
import pytest

from s2c.multiview import spec as S
from s2c.multiview.build import build, volume
from s2c.multiview.pipeline import MvPipeline
from s2c.multiview.sheet_read import link_diameters, read_sheet
from s2c.reading import ReadingService
from tests.line_views import WordReader, _text, drawing_sheet
from tests.test_mv_relief import _block, _cut, _pins

THIRD = ("front", "top", "right")
FIRST = ("front", "top", "left")


def _part():
    return _pins(_cut(_cut(_block(), 0, 40, 50, 15, 60, 70), 65, 40, 50, 80, 60, 70), z=20)


def _faces(read):
    return sorted(c.face for c in read.crops)


def _service(words):
    return ReadingService([WordReader(words)], cache=None)


def test_a_third_angle_sheet_is_read_as_third_angle_with_its_picture_left_out():
    img, _, words = drawing_sheet(_part(), faces=THIRD, layout="third", iso=True)
    read = read_sheet(img, "auto", service=_service(words))
    assert read is not None
    assert _faces(read) == sorted(THIRD)
    assert (read.naming.projection, read.naming.projection_source) == ("third", "drawing")
    assert read.scale.mm_per_px == pytest.approx(0.25, rel=0.01)
    assert any("left out" in w for w in read.warnings)


def test_a_first_angle_sheet_stays_first_angle():
    img, _, words = drawing_sheet(_part(), faces=FIRST, layout="first", iso=True)
    read = read_sheet(img, "auto", service=_service(words))
    assert _faces(read) == sorted(FIRST)
    assert read.naming.projection == "first"


def test_a_symmetric_part_keeps_the_iso_convention():
    """A plain block reads the same either way: the view above the front is the bottom view (ISO first-angle)."""
    img, _, words = drawing_sheet(_block(), faces=THIRD, layout="third")
    read = read_sheet(img, "auto", service=_service(words))
    assert read.naming.projection == "first" and read.naming.projection_source == "setting"
    assert _faces(read) == ["bottom", "front", "left"]


def test_the_sheet_builds_with_no_typed_size():
    part = _part()
    img, _, words = drawing_sheet(part, faces=THIRD, layout="third", iso=True)
    read = read_sheet(img, "auto", service=_service(words))
    pipe = MvPipeline()
    observed = pipe.observe(read.inputs())
    link_diameters(read, observed)
    spec = pipe.fuse(observed)
    assert isinstance(spec, S.MultiViewSpec), spec
    env = spec.envelope
    assert (env.x_mm, env.y_mm, env.z_mm) == pytest.approx((80, 66, 70), rel=0.02)
    assert volume(build(spec)) == pytest.approx(part.val().Volume(), rel=0.03)


def test_a_diameter_written_by_a_hole_is_its_size():
    part = _block().faces(">Z").workplane(centerOption="CenterOfBoundBox").hole(12)
    img, boxes, words = drawing_sheet(part, faces=THIRD, layout="third", gap=220)
    ink = 255 - cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    fx, fy, fw, fh = boxes["front"]
    cx, cy = fx + fw // 2, fy + fh // 2  # the hole, Ø12 at 4 px/mm: radius 24 px
    box = _text(ink, "D12", fx + fw + 20, cy - 30, 22, False)   # written outside the view, a leader to the hole
    cv2.line(ink, (cx + 24, cy), (fx + fw + 18, cy - 8), 255, 1)
    img = cv2.cvtColor(255 - ink, cv2.COLOR_GRAY2BGR)
    read = read_sheet(img, "third", service=_service([*words, (box, "D12")]))
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
