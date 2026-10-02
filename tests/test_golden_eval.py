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


def test_views_are_scored_by_where_they_are():
    """Swapped names on the right views are wrong: each named view counts only when the view nearest its place in
    the truth's layout has that name."""
    layout = {"top": [0.3, 0.2], "front": [0.3, 0.7], "right": [0.8, 0.7]}
    assert golden.named_by_position([("top", (0.31, 0.22)), ("front", (0.3, 0.68)), ("right", (0.79, 0.7))],
                                    layout) == 3
    assert golden.named_by_position([("front", (0.31, 0.22)), ("top", (0.3, 0.68)), ("right", (0.79, 0.7))],
                                    layout) == 1


def test_a_read_that_matches_nothing_is_a_misread():
    wanted = [{"value": 60, "kind": "linear"}, {"value": 25, "kind": "linear"}]
    assert golden.misreads(wanted, [(60.0, "linear"), (7.0, "linear"), (None, "linear"), (25.0, "diameter")]) == 2


def test_the_summary_pools_every_sample():
    rows = [{"views": 3, "named": 3, "named_auto": 1, "values": 10, "matched": 3, "misread": 1, "sizes": 2,
             "correct": 2, "wrong": 0, "built": True},
            {"views": 2, "named": 1, "named_auto": 1, "values": 4, "matched": 4, "misread": 0, "sizes": 0,
             "correct": 0, "wrong": 1, "built": False}]
    assert golden.summarize(rows) == {"samples": 2, "faces": 0.8, "faces_auto": 0.4, "values": 0.5, "misreads": 1,
                                      "sizes": 1.0, "silent_errors": 1, "builds": 0.5}


BASE = {"samples": 1, "faces": 1.0, "faces_auto": 0.33, "values": 0.3, "misreads": 1, "sizes": 1.0,
        "silent_errors": 0, "builds": 1.0}


def test_a_drop_past_the_tolerance_fails_the_gate():
    assert golden.regressions({**BASE, "values": 0.29}, BASE) == []
    assert golden.regressions({**BASE, "values": 0.2}, BASE) == ["values fell from 0.30 to 0.20"]
    assert golden.regressions({**BASE, "silent_errors": 1}, BASE) == ["silent errors rose from 0 to 1"]
    assert golden.regressions({**BASE, "misreads": 2}, BASE) == ["misreads rose from 1 to 2"]


def test_a_changed_set_needs_a_new_baseline():
    """Pooled numbers over a different set of samples are not comparable: an easy new sample could hide a drop."""
    assert golden.regressions({**BASE, "samples": 2}, BASE) == [
        "the set has 2 samples and the baseline 1: record a new one with --write-baseline"]


def test_samples_are_folders_with_an_image_and_its_truth(tmp_path):
    for name, files in (("a", ("image.jpg", "expected.json")), ("b", ("expected.json",)), ("c", ("image.png",))):
        (tmp_path / name).mkdir()
        for f in files:
            (tmp_path / name / f).write_text("{}")
    assert [s.name for s in golden.samples(tmp_path)] == ["a"]  # b is drawn by its test, c has no truth


def test_the_gate_waits_for_the_reader_weights(tmp_path):
    """Without TrOCR's weights in the cache (tests run offline) the gate skips instead of scoring every value 0."""
    assert golden.reader_ready(cache_dir=tmp_path) is False


def test_the_golden_set_holds_its_baseline():
    """The real photos in tests/golden_sketch, read with TrOCR only, must not fall below the committed baseline.
    Skipped without the trocr extra or its downloaded weights."""
    pytest.importorskip("transformers")
    if not golden.reader_ready():
        pytest.skip("the TrOCR weights are not downloaded (uv run python scripts/golden_eval.py fetches them)")
    summary = golden.evaluate(GOLDEN)["summary"]
    baseline = json.loads((GOLDEN / "baseline.json").read_text(encoding="utf-8"))
    assert golden.regressions(summary, baseline) == [], summary
