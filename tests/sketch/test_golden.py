import json
import os
import sys
from pathlib import Path

import cv2
import pytest

from s2c.sketch.pipeline import analyse
from tests.sketch.synth import Sheet, TruthReader, bridge_block

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from sketch_accuracy import score

GOLDEN = Path(__file__).resolve().parents[1] / "golden_sketch"


@pytest.mark.xfail(strict=True, reason=(
    "Known classify limits end to end (13 of 18 values read at the right view): the SIDE '50' dimension is cut by "
    "the '12.5' text and its remnants widen the SIDE view; see tasks-10-13 report."))
def test_synthetic_bridge_block_scores_perfectly():
    sh = bridge_block(Sheet())
    expected = json.loads((GOLDEN / "bridge_block_synthetic" / "expected.json").read_text(encoding="utf-8"))
    reading, trace = analyse(cv2.imencode(".png", sh.bgr())[1].tobytes(),
                             [TruthReader(sh.texts, "a"), TruthReader(sh.texts, "b")])
    s = score(reading, trace, expected)
    assert s["read_ok"] == s["values"] == 18
    assert s["link_ok"] == 18
    assert s["envelope_ok"] == 1 and s["holes_ok"] == 4
    assert s["silent_errors"] == 0 and s["edits"] == 0


def test_misread_values_are_never_silent():
    sh = bridge_block(Sheet())
    expected = json.loads((GOLDEN / "bridge_block_synthetic" / "expected.json").read_text(encoding="utf-8"))
    readers = [TruthReader(sh.texts, "a"), TruthReader(sh.texts, "b", swap={"25": "35", "50": "5O"})]
    reading, trace = analyse(cv2.imencode(".png", sh.bgr())[1].tobytes(), readers)
    assert score(reading, trace, expected)["silent_errors"] == 0


REAL = [p for p in GOLDEN.iterdir() if (p / "expected.json").exists()
        and any(q.suffix.lower() in (".jpg", ".jpeg", ".png") for q in p.iterdir())]


@pytest.mark.skipif(os.environ.get("SKETCH_GOLDEN") != "1", reason="needs real readers")
@pytest.mark.parametrize("folder", REAL, ids=[p.name for p in REAL])
def test_real_golden_sheet_has_no_silent_errors(folder):
    image = next(q for q in folder.iterdir() if q.suffix.lower() in (".jpg", ".jpeg", ".png"))
    expected = json.loads((folder / "expected.json").read_text(encoding="utf-8"))
    reading, trace = analyse(image.read_bytes())
    assert score(reading, trace, expected)["silent_errors"] == 0
