"""The user's edits (audit M6): one parser says which paths exist; nothing else reaches a spec."""
import pytest

from s2c.multiview import edits
from s2c.multiview.spec import feature_path, finish_path


def test_known_paths_pass():
    values = {"envelope.x_mm": 60, "features[0].diameter_mm": 6, "features[2].keep": 0, "features[1].angle_deg": 30}
    assert edits.parse(values) == values


@pytest.mark.parametrize("path", ["features[0].type", "envelope.w_mm", "features[a].a_mm", "views.front.outer",
                                  "features[0].face", "finishes[0].radius_mm ", "envelope.x_mm.x"])
def test_any_other_path_is_refused(path):
    with pytest.raises(edits.EditError) as e:
        edits.parse({path: 1})
    assert e.value.path == path


def test_provenance_keys_have_one_spelling():
    assert feature_path(3, "diameter_mm") == "features[3].diameter_mm"
    assert finish_path(0, "radius_mm") == "finishes[0].radius_mm"
