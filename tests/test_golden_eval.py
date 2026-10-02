"""The golden-set evaluator (audit C3): what it counts, and the gate that fails when accuracy drops."""
import json
from pathlib import Path

import pytest

from s2c.multiview import golden

GOLDEN = Path(__file__).resolve().parent / "golden_sketch"


def test_values_are_matched_once_each_by_number_and_kind():
    wanted = [{"value": 60, "kind": "linear"}, {"value": 25, "kind": "linear"}, {"value": 25, "kind": "linear"},
              {"value": 6, "kind": "diameter"}]
    got = [(60.0, "linear"), (25.0, "linear"), (6.0, "linear"), (7.0, "linear")]
    assert golden.matched_values(wanted, got) == 2


def test_sizes_are_correct_silently_wrong_or_asked():
    """A size the reader wrote in for the user and got wrong is the worst outcome: it is counted on its own."""
    marks = golden.size_marks({"x": 60, "y": 40, "z": 25}, {"x": 60.0, "y": 40.0, "z": 30.0},
                              {"x": "user_written", "y": "user_edited", "z": "user_written"})
    assert marks == {"correct": 1, "wrong": 1, "asked": 1}


def test_the_summary_pools_every_sample():
    rows = [{"views": 3, "named": 3, "values": 10, "matched": 3, "sizes": 2, "correct": 2, "wrong": 0, "built": True},
            {"views": 2, "named": 1, "values": 4, "matched": 4, "sizes": 0, "correct": 0, "wrong": 1, "built": False}]
    assert golden.summarize(rows) == {"samples": 2, "faces": 0.8, "values": 0.5, "sizes": 1.0, "silent_errors": 1,
                                      "builds": 0.5}


def test_a_drop_past_the_tolerance_fails_the_gate():
    base = {"samples": 1, "faces": 1.0, "values": 0.3, "sizes": 1.0, "silent_errors": 0, "builds": 1.0}
    assert golden.regressions({**base, "values": 0.29}, base) == []
    assert golden.regressions({**base, "values": 0.2}, base) == ["values fell from 0.30 to 0.20"]
    assert golden.regressions({**base, "silent_errors": 1}, base) == ["silent errors rose from 0 to 1"]


def test_samples_are_folders_with_an_image_and_its_truth(tmp_path):
    for name, files in (("a", ("image.jpg", "expected.json")), ("b", ("expected.json",)), ("c", ("image.png",))):
        (tmp_path / name).mkdir()
        for f in files:
            (tmp_path / name / f).write_text("{}")
    assert [s.name for s in golden.samples(tmp_path)] == ["a"]  # b is drawn by its test, c has no truth


def test_the_golden_set_holds_its_baseline():
    """The real photos in tests/golden_sketch, read with TrOCR only, must not fall below the committed baseline.
    Skipped without the trocr extra."""
    pytest.importorskip("transformers")
    summary = golden.evaluate(GOLDEN)["summary"]
    baseline = json.loads((GOLDEN / "baseline.json").read_text(encoding="utf-8"))
    assert golden.regressions(summary, baseline) == [], summary
