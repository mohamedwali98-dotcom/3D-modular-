import math

import cadquery as cq
import pytest

from s2c.multiview.build import BuildError, build, export, volume
from tests.mv_helpers import circle, make_spec, outline, rect


def region_volume(solid, lo, hi):
    box = cq.Workplane("XY").box(hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2], centered=False).translate(lo)
    return volume(solid.intersect(box))


def hole(face, a, b, d, depth=None):
    return {"type": "hole", "face": face, "a_mm": a, "b_mm": b, "diameter_mm": d, "depth_mm": depth}


def test_three_rectangles_make_the_envelope_box():
    solid = build(make_spec((60.0, 40.0, 5.0)))
    assert math.isclose(volume(solid), 60 * 40 * 5, rel_tol=0.005)
    bb = solid.val().BoundingBox()
    assert [round(v, 6) for v in (bb.xmin, bb.ymin, bb.zmin, bb.xmax, bb.ymax, bb.zmax)] == [0, 0, 0, 60, 40, 5]


def test_front_outline_orientation():
    solid = build(make_spec((40.0, 30.0, 20.0), front=outline([(0, 0), (40, 0), (0, 30)])))
    assert math.isclose(volume(solid), 600 * 20, rel_tol=0.005)
    assert math.isclose(region_volume(solid, (0, 0, 0), (20, 30, 20)), 9000, rel_tol=0.01)


def test_top_outline_orientation():
    solid = build(make_spec((40.0, 30.0, 20.0), top=outline([(0, 0), (40, 0), (0, 20)])))
    assert math.isclose(volume(solid), 400 * 30, rel_tol=0.005)
    # b = 0 is the front edge (Z = z_mm), so the wide end of the triangle sits at the front
    assert math.isclose(region_volume(solid, (0, 0, 10), (40, 30, 20)), 9000, rel_tol=0.01)


def test_right_outline_orientation():
    solid = build(make_spec((40.0, 30.0, 20.0), right=outline([(0, 0), (20, 0), (0, 30)])))
    assert math.isclose(volume(solid), 300 * 40, rel_tol=0.005)
    # a = 0 is the front edge (Z = z_mm), so the tall end of the triangle sits at the front
    assert math.isclose(region_volume(solid, (0, 0, 10), (40, 30, 20)), 9000, rel_tol=0.01)


def test_through_holes():
    solid = build(make_spec((60.0, 40.0, 5.0), features=[hole("front", 10.0, 10.0, 6.0), hole("front", 50.0, 30.0, 6.0)]))
    assert math.isclose(volume(solid), 60 * 40 * 5 - 2 * math.pi * 9 * 5, rel_tol=0.005)


@pytest.mark.parametrize("prov", ["user_written", "user_edited"])
def test_a_typed_hole_that_breaks_the_part_still_fails(prov):
    """Only a hole read from a drawing is left out when its cut breaks the part: a size the user gave is kept."""
    spec = make_spec((60.0, 10.0, 5.0), features=[hole("front", 30.0, 5.0, 12.0)], prov=prov)
    with pytest.raises(BuildError):
        build(spec)


def test_a_read_hole_that_breaks_the_part_is_left_out_and_said():
    spec = make_spec((60.0, 10.0, 5.0), features=[hole("front", 30.0, 5.0, 12.0)], prov="measured")
    notes: list[str] = []
    solid = build(spec, notes)
    assert math.isclose(volume(solid), 60 * 10 * 5, rel_tol=0.005)
    assert notes == ["A hole read on the front view would break the part, so it was left out. Type its size to keep it."]


def test_blind_hole_on_the_back_is_cut_from_the_back():
    solid = build(make_spec((60.0, 40.0, 5.0), features=[hole("back", 10.0, 10.0, 6.0, 2.0)]))
    assert math.isclose(volume(solid), 60 * 40 * 5 - math.pi * 9 * 2, rel_tol=0.005)
    assert math.isclose(region_volume(solid, (0, 0, 0), (60, 40, 2)), 60 * 40 * 2 - math.pi * 9 * 2, rel_tol=0.005)


def test_hole_on_the_right_face_runs_along_x():
    solid = build(make_spec((60.0, 40.0, 20.0), features=[hole("right", 10.0, 20.0, 6.0)]))
    assert math.isclose(volume(solid), 60 * 40 * 20 - math.pi * 9 * 60, rel_tol=0.005)


def test_slot():
    slot = {"type": "slot", "face": "front", "a_mm": 30.0, "b_mm": 20.0, "width_mm": 6.0, "length_mm": 20.0}
    solid = build(make_spec((60.0, 40.0, 5.0), features=[slot]))
    assert math.isclose(volume(solid), 60 * 40 * 5 - ((20 - 6) * 6 + math.pi * 9) * 5, rel_tol=0.005)


def test_spacer_from_a_circle_and_two_rectangles():
    s = make_spec((20.0, 20.0, 30.0), front=outline(circle(10, 10, 10), inner=[circle(10, 10, 4)]))
    assert math.isclose(volume(build(s)), math.pi * (100 - 16) * 30, rel_tol=0.005)


def test_l_bracket_profile():
    l_shape = [(0, 0), (50, 0), (50, 30), (47, 30), (47, 3), (0, 3)]
    s = make_spec((50.0, 30.0, 20.0), front=outline(l_shape))
    assert math.isclose(volume(build(s)), (50 * 3 + 3 * 27) * 20, rel_tol=0.005)


def test_flange():
    feats = [hole("front", 40.0, 40.0, 30.0)] + [hole("front", float(a), float(b), 6.0)
                                                  for a, b in [(70, 40), (10, 40), (40, 70), (40, 10)]]
    s = make_spec((80.0, 80.0, 6.0), front=outline(circle(40, 40, 40, n=360)), features=feats)
    expected = math.pi * (40 ** 2 - 15 ** 2) * 6 - 4 * math.pi * 9 * 6
    assert math.isclose(volume(build(s)), expected, rel_tol=0.005)


def test_disjoint_views_are_rejected():
    s = make_spec((60.0, 40.0, 5.0), front=outline(rect(10.0, 40.0)), top=outline(rect(10.0, 5.0, x0=50.0)))
    with pytest.raises(BuildError) as e:
        build(s)
    assert e.value.reason == "intersection_empty"


def test_feature_outside_the_part_is_rejected():
    with pytest.raises(BuildError) as e:
        build(make_spec((60.0, 40.0, 5.0), features=[hole("front", 70.0, 10.0, 6.0)]))
    assert e.value.reason == "feature_outside_part"


def test_fillet_reduces_volume_and_a_huge_fillet_fails():
    v0 = volume(build(make_spec((60.0, 40.0, 5.0))))
    small = make_spec((60.0, 40.0, 5.0), finishes=[{"type": "fillet", "edges": "all_vertical", "radius_mm": 2.0}])
    assert volume(build(small)) < v0
    huge = make_spec((60.0, 40.0, 5.0), finishes=[{"type": "fillet", "edges": "all_vertical", "radius_mm": 30.0}])
    with pytest.raises(BuildError) as e:
        build(huge)
    assert e.value.reason == "fillet_failed"


def test_export_writes_step_and_stl(tmp_path):
    step, stl = export(build(make_spec((60.0, 40.0, 5.0))), tmp_path)
    assert step.read_text(errors="ignore").startswith("ISO-10303-21")
    assert stl.stat().st_size > 84


def test_build_retries_a_fuzzy_intersection_when_the_exact_one_comes_back_empty(monkeypatch):
    """OCC can return nothing when two faces almost coincide; a 1e-4 mm fuzzy boolean then gives the part."""
    exact = cq.Workplane.intersect
    calls = []

    def flaky(self, other, clean=True, tol=None):
        calls.append(tol)
        return cq.Workplane() if tol is None else exact(self, other, clean=clean, tol=tol)

    monkeypatch.setattr(cq.Workplane, "intersect", flaky)
    solid = build(make_spec((40.0, 30.0, 20.0), front=outline([(0, 0), (40, 0), (0, 30)])))
    assert math.isclose(volume(solid), 600 * 20, rel_tol=0.005)
    assert any(t is not None for t in calls)


def test_a_really_empty_intersection_still_abstains():
    with pytest.raises(BuildError) as e:
        build(make_spec((40.0, 30.0, 20.0), front=outline(rect(10, 10)), top=outline(rect(10, 10, 30, 10))))
    assert e.value.reason == "intersection_empty"
