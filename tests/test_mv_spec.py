import pytest
from pydantic import ValidationError

from s2c.multiview.spec import (
    CANONICAL_OF,
    FACES,
    Envelope,
    MultiViewSpec,
    face_size,
    numeric_field_paths,
    to_canonical,
    to_global,
)
from tests.mv_helpers import make_spec, outline, rect

ENV = Envelope(x_mm=60.0, y_mm=40.0, z_mm=20.0)
HOLE = {"type": "hole", "face": "front", "a_mm": 10.0, "b_mm": 10.0, "diameter_mm": 6.0}


def test_face_sizes_follow_the_face_table():
    assert face_size("front", ENV) == (60, 40) and face_size("back", ENV) == (60, 40)
    assert face_size("top", ENV) == (60, 20) and face_size("bottom", ENV) == (60, 20)
    assert face_size("right", ENV) == (20, 40) and face_size("left", ENV) == (20, 40)


def test_canonical_faces_map_to_global_axes():
    assert to_global("front", 10, 5, ENV) == {"x": 10, "y": 5}
    assert to_global("top", 10, 5, ENV) == {"x": 10, "z": 15}    # b runs from the front edge backwards
    assert to_global("right", 5, 7, ENV) == {"z": 15, "y": 7}    # a runs from the front edge backwards


@pytest.mark.parametrize("face", FACES)
def test_mirror_rule_keeps_the_same_global_point(face):
    for a, b in [(0.0, 0.0), (12.5, 3.0), (7.0, 19.0)]:
        (ca, cb), = to_canonical(face, [(a, b)], ENV)
        assert to_global(face, a, b, ENV) == pytest.approx(to_global(CANONICAL_OF[face], ca, cb, ENV))


def test_valid_spec_round_trips_through_json():
    s = make_spec((60.0, 40.0, 20.0), features=[HOLE])
    assert MultiViewSpec.model_validate_json(s.model_dump_json()) == s


def test_numeric_field_paths():
    s = make_spec((60.0, 40.0, 20.0), features=[HOLE])
    assert numeric_field_paths(s) == [
        "envelope.x_mm", "envelope.y_mm", "envelope.z_mm", "views.front.outer", "views.top.outer",
        "views.right.outer", "features[0].a_mm", "features[0].b_mm", "features[0].diameter_mm"]


def test_envelope_needs_a_trusted_source():
    with pytest.raises(ValidationError, match="trusted"):
        make_spec((60.0, 40.0, 20.0), prov="estimated")


def test_missing_provenance_is_rejected():
    data = make_spec((60.0, 40.0, 20.0)).model_dump()
    del data["provenance"]["envelope.z_mm"]
    with pytest.raises(ValidationError, match="envelope.z_mm"):
        MultiViewSpec.model_validate(data)


def test_outline_outside_the_envelope_is_rejected():
    with pytest.raises(ValidationError, match="outside"):
        make_spec((60.0, 40.0, 20.0), front=outline(rect(70.0, 40.0)))


def test_invented_fields_are_rejected():
    data = make_spec((60.0, 40.0, 20.0)).model_dump()
    data["envelope"]["w_mm"] = 3.0
    with pytest.raises(ValidationError):
        MultiViewSpec.model_validate(data)


@pytest.mark.parametrize("bad", [float("inf"), float("nan"), 1e300])
def test_sizes_are_finite_and_bounded(bad):
    with pytest.raises(ValidationError):
        Envelope(x_mm=bad, y_mm=10.0, z_mm=10.0)


def test_a_spec_cannot_carry_unbounded_lists():
    with pytest.raises(ValidationError):
        make_spec((60.0, 40.0, 20.0), features=[HOLE] * 201)
    many = [(60.0 * i / 6000, 0.0) for i in range(6000)] + [(30.0, 20.0)]
    with pytest.raises(ValidationError):
        make_spec((60.0, 40.0, 20.0), front=outline(many))
