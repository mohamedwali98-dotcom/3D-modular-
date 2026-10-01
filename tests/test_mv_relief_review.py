"""Regressions from the complex-parts review: thin plates, pockets with no hidden lines, a bore in a hub, AI-drawn
outlines, removing a misread feature, pin snapping."""
import cadquery as cq
import pytest

from s2c.multiview import spec as S
from s2c.multiview.build import build, volume
from tests.test_mv_relief import _block, _cut, _pockets, _spec


@pytest.mark.parametrize("size", [(100, 60, 5), (80, 60, 10)])
def test_a_thin_plate_keeps_its_thickness(size):
    part = _block(*size)
    spec = _spec(part)
    assert _pockets(spec) == []
    assert volume(build(spec)) == pytest.approx(part.val().Volume(), rel=0.03)


def test_a_pocket_with_no_hidden_lines_is_never_a_window():
    """No hidden line shows how deep the pockets go: they come out blind, their depth flagged for the user."""
    part = _cut(_cut(_block(100, 40, 80), 10, 25, 10, 40, 40, 70), 60, 25, 10, 90, 40, 70)
    spec = _spec(part, hidden=False)
    pockets = _pockets(spec)
    assert pockets and all(p.depth_mm is not None for p in pockets)
    for k, f in enumerate(spec.features):
        if f.type == "pocket":
            assert spec.provenance[f"features[{k}].depth_mm"] == "default"
    assert any("depth" in w for w in spec.warnings)
    assert volume(build(spec)) > 0.85 * part.val().Volume()


def test_a_bore_in_a_hub_stays_a_hole():
    base = cq.Workplane("XY").box(60, 10, 60, centered=False)
    hub = cq.Workplane("XZ", origin=(0, 10, 0)).center(30, 30).circle(14).extrude(-8)
    part = base.union(hub).cut(cq.Workplane("XZ", origin=(0, 20, 0)).center(30, 30).circle(10).extrude(20))
    spec = _spec(part, hidden=False)
    holes = [f for f in spec.features if f.type == "hole"]
    assert any(f.diameter_mm == pytest.approx(20, abs=1.5) for f in holes), spec.features
    assert not [f for f in spec.features if f.type == "boss" and f.diameter_mm < 24]
    assert volume(build(spec)) == pytest.approx(part.val().Volume(), rel=0.05)


def test_features_read_on_an_assumed_outline_are_inferred():
    """Front and top drawn, the right side assumed: a pocket's extent may come from the assumed outline."""
    part = _cut(_block(), 0, 40, 50, 15, 60, 70)
    spec = _spec(part, faces=("front", "top"))
    for k, f in enumerate(spec.features):
        if f.type in ("pocket", "boss"):
            assert spec.provenance[f"features[{k}].a_mm"] == "inferred"


def test_a_misread_feature_can_be_removed():
    from s2c.multiview.pipeline import ImageInput, MvPipeline
    from tests.line_views import draw_view
    from tests.test_mv_relief import _page
    part = _cut(_block(), 0, 40, 50, 15, 60, 70)
    pipe = MvPipeline()
    observed = pipe.observe([ImageInput(_page(draw_view(part, f, 4.0)), f, "drawing") for f in ("front", "top", "right")])
    env = {"envelope.x_mm": 80, "envelope.y_mm": 60, "envelope.z_mm": 70}
    spec = pipe.fuse(observed, env)
    k = next(i for i, f in enumerate(spec.features) if f.type == "pocket")
    kept = pipe.fuse(observed, {**env, f"features[{k}].keep": 0})
    assert len(kept.features) == len(spec.features) - 1
    assert not _pockets(kept)


def test_a_pin_diameter_snaps_to_the_grid_not_to_a_hole_size():
    from s2c.multiview.fuse import snap
    data = {"features": [{"type": "boss", "face": "top", "a_mm": 20, "b_mm": 15, "diameter_mm": 6.3, "height_mm": 6}],
            "provenance": {"features[0].diameter_mm": "scaled"}, "envelope": {"x_mm": 40, "y_mm": 20, "z_mm": 30},
            "views": {}}
    snap(data)
    assert data["features"][0]["diameter_mm"] == 6.5
    assert S.FaceBoss(**data["features"][0])
