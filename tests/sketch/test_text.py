import os

import numpy as np
import pytest

from s2c.sketch.models import SketchAbstain
from s2c.sketch.text import erase_mask, find_text_boxes, read_texts
from tests.sketch.synth import Sheet, TruthReader, boxes_overlap, bridge_block


def sheet():
    return bridge_block(Sheet())


def test_boxes_cover_every_written_text():
    sh = sheet()
    boxes = find_text_boxes(sh.ink(), 3.0)
    for s, b in sh.texts:
        assert any(boxes_overlap(box, b) > 0.6 for box in boxes), s
    for box in boxes:  # no box swallows two texts
        assert sum(boxes_overlap(box, b) > 0.6 for _, b in sh.texts) <= 1


def test_boxes_stay_tight_around_their_text():
    """A box must not swallow the short dimensions, arrowheads and extension lines next to its text:
    erasing it would wipe that geometry."""
    sh = sheet()
    boxes = find_text_boxes(sh.ink(), 3.0)
    for s, b in sh.texts:
        found = [box for box in boxes if boxes_overlap(box, b) > 0.6]
        assert found, s
        for box in found:
            assert box[2] * box[3] <= 2.5 * b[2] * b[3], (s, b, box)


def _ink_box(sh):
    """The last text's ink extent, measured before any line is drawn next to it: the synthetic tile
    box has margins that make one narrow glyph fail the 0.6 overlap on its own."""
    s, (x, y, w, h) = sh.texts[-1]
    ys, xs = np.nonzero(sh.ink()[y:y + h, x:x + w])
    box = (int(x + xs.min()), int(y + ys.min()), int(np.ptp(xs)) + 1, int(np.ptp(ys)) + 1)
    sh.texts[-1] = (s, box)
    return box


def _one_box_keeps_every_glyph(sh):
    """A thin glyph touching a drawn line must not be dropped as line debris."""
    boxes = find_text_boxes(sh.ink(), 3.0)
    for s, b in sh.texts:
        found = [box for box in boxes if boxes_overlap(box, b) > 0.6]
        assert len(found) == 1, (s, b, boxes)
        assert found[0][2] >= 6 and found[0][3] >= 6, (s, found[0])
        assert found[0][0] <= b[0] and found[0][0] + found[0][2] >= b[0] + b[2], (s, b, found[0])


def test_a_printed_one_standing_on_a_line_keeps_its_box():
    sh = Sheet()
    sh.text("1", (500, 480))
    _, top, _, height = _ink_box(sh)
    sh.line((300, top + height), (700, top + height), 1)  # touching the lowest ink row
    _one_box_keeps_every_glyph(sh)


def test_a_single_stroke_one_standing_on_a_line_keeps_its_box():
    """A hand-written "1" is one straight stroke, thin and long like a cut extension line."""
    sh = Sheet()
    sh.line((494, 463), (494, 479), 2)
    sh.text("5", (504, 473))
    sh.texts[-1] = ("15", (492, 463, 20, 17))
    _, top, _, height = _ink_box(sh)
    sh.line((300, top + height), (700, top + height), 1)  # touching the lowest ink row
    _one_box_keeps_every_glyph(sh)


def test_a_minus_sign_touching_a_line_keeps_its_box():
    sh = Sheet()
    sh.text("-5", (500, 480))
    left = _ink_box(sh)[0]
    sh.line((left - 1, 300), (left - 1, 700), 1)  # touching the minus
    _one_box_keeps_every_glyph(sh)


def test_a_decimal_point_on_a_line_keeps_its_box():
    sh = Sheet()
    sh.text("1.5", (500, 480))
    _, top, _, height = _ink_box(sh)
    sh.line((300, top + height), (700, top + height), 1)  # touching the lowest ink row
    _one_box_keeps_every_glyph(sh)


def test_a_hole_note_next_to_lines_keeps_its_box():
    """The "x" and the slash of the Ø sit next to the outline and to the leader."""
    sh = Sheet()
    sh.text("2xØ12.5", (500, 480))
    left, top, _, height = _ink_box(sh)
    bottom = top + height
    sh.line((300, bottom), (700, bottom), 1)  # touching the lowest ink row
    sh.line((left + 14, bottom), (left - 30, bottom + 44), 1)
    _one_box_keeps_every_glyph(sh)


def test_two_agreeing_readers_give_written_values_and_labels():
    sh = sheet()
    items = read_texts(sh.bgr(), find_text_boxes(sh.ink(), 3.0),
                       [TruthReader(sh.texts, "a"), TruthReader(sh.texts, "b")])
    dims = [t for t in items if t.role == "dimension"]
    labels = {t.label for t in items if t.role == "label"}
    assert labels == {"top", "front", "right"}
    assert len(dims) == 18
    assert all(t.badge == "written" for t in dims)
    assert sorted(t.parsed.value for t in dims).count(12.5) == 11


def test_disagreement_is_uncertain_with_both_candidates():
    sh = sheet()
    items = read_texts(sh.bgr(), find_text_boxes(sh.ink(), 3.0),
                       [TruthReader(sh.texts, "a"), TruthReader(sh.texts, "b", swap={"100": "700"})])
    t = next(t for t in items if t.role == "dimension" and 100.0 in t.candidates)
    assert t.badge == "uncertain" and t.candidates == [100.0, 700.0]
    assert [r.reader for r in t.readings] == ["a", "b"]


def test_a_single_reader_is_never_trusted_alone():
    sh = sheet()
    items = read_texts(sh.bgr(), find_text_boxes(sh.ink(), 3.0), [TruthReader(sh.texts)])
    assert all(t.badge == "uncertain" for t in items if t.role == "dimension")


def test_a_small_circle_read_as_O_stays_in_the_geometry():
    sh = Sheet()
    sh.circle((600, 500), 14)
    sh.texts.append(("O", (582, 482, 36, 36)))  # what a reader would say about the circle
    sh.text("40", (900, 500))
    items = read_texts(sh.bgr(), find_text_boxes(sh.ink(), 3.0),
                       [TruthReader(sh.texts, "a"), TruthReader(sh.texts, "b")])
    mask = erase_mask(items, sh.img.shape[:2])
    assert mask[500, 586] == 0 and mask[486, 600] == 0  # the circle is not erased
    assert mask[500, 900] == 255                         # the number is


def test_sideways_vertical_text_is_read():
    sh = Sheet()
    sh.line((400, 300), (400, 700), 1)
    sh.text("37.5", (380, 500), rotate=True)
    readers = [TruthReader(sh.texts, "a", upright_only=True),
               TruthReader(sh.texts, "b", upright_only=True)]
    items = read_texts(sh.bgr(), find_text_boxes(sh.ink(), 3.0), readers)
    dims = [t for t in items if t.role == "dimension"]
    assert len(dims) == 1 and dims[0].parsed.value == 37.5 and dims[0].badge == "written"


def test_readers_down_abstains():
    class Down:
        name = "down"

        def read(self, crops):
            return None
    sh = sheet()
    out = read_texts(sh.bgr(), find_text_boxes(sh.ink(), 3.0), [Down(), Down()])
    assert isinstance(out, SketchAbstain) and out.reason == "readers_unavailable"


def test_no_text_needs_no_reader():
    assert read_texts(np.full((100, 100, 3), 255, np.uint8), [], []) == []


def test_classical_detector_is_the_fallback(monkeypatch):
    from s2c.sketch.text import detect_text_boxes
    sh = sheet()
    monkeypatch.setenv("SKETCH_TEXT_DETECTOR", "classical")
    assert detect_text_boxes(sh.bgr(), sh.ink(), 3.0) == find_text_boxes(sh.ink(), 3.0)


def test_a_failing_second_reader_makes_values_uncertain():
    sh = Sheet()
    bridge_block(sh)

    class Down:
        name = "down"

        def read(self, crops):
            return None

    items = read_texts(sh.bgr(), find_text_boxes(sh.ink(), 3.0), [TruthReader(sh.texts, "a"), Down()])
    dims = [t for t in items if t.role == "dimension"]
    assert dims and all(t.badge == "uncertain" for t in dims)


@pytest.mark.skipif(os.environ.get("SKETCH_MODEL_TESTS") != "1", reason="downloads a model")
def test_paddle_detector_finds_the_written_values(monkeypatch):
    from s2c.sketch.text import detect_text_boxes
    sh = sheet()
    monkeypatch.setenv("SKETCH_TEXT_DETECTOR", "paddle")
    boxes = detect_text_boxes(sh.bgr(), sh.ink(), 3.0)
    found = sum(any(boxes_overlap(box, b) > 0.5 for box in boxes) for _, b in sh.texts)
    assert found >= 0.9 * len(sh.texts)
