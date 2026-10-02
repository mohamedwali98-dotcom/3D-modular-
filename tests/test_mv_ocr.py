import cv2
import numpy as np
import pytest

from s2c.multiview.ocr import Reading, decide, link, parse_value, read_values, text_regions
from s2c.multiview.outline import PixelCircle, PixelOutline, extract
from s2c.reading import ReaderResult, ReaderRun, ReadingService, as_reader


def test_parse_value():
    assert parse_value("60") == (60.0, "linear")
    assert parse_value(" 12,5 mm") == (12.5, "linear")
    assert parse_value("Ø6") == (6.0, "diameter")
    assert parse_value("⌀ 8") == (8.0, "diameter")
    assert parse_value("o6") == (6.0, "diameter")
    assert parse_value("R3") == (3.0, "radius")
    assert parse_value("D10") == (10.0, "diameter")
    assert parse_value("1O") == (10.0, "linear")
    assert parse_value("hello") is None
    assert parse_value("0") is None


def outline_with_holes(circular=False):
    return PixelOutline(outer=np.array([[400, 300], [1000, 300], [1000, 700], [400, 700]]),
                        circles=[PixelCircle(500, 400, 60), PixelCircle(900, 600, 60)],
                        bbox=(400, 300, 601, 401), circular=circular, shape=(1200, 1600))


def reading(value, kind, cx, cy):
    return Reading(float(value), kind, (cx - 20, cy - 15, 40, 30), 0.9, str(value))


def test_link_by_position():
    linked = link([reading(60, "linear", 700, 760), reading(40, "linear", 330, 500),
                   reading(6, "diameter", 560, 380), reading(99, "linear", 700, 500)], outline_with_holes())
    assert [(lv.axis, lv.hole_index) for lv in linked] == [("a", None), ("b", None), (None, 0), (None, None)]


def test_diameter_outside_a_round_outline_is_its_envelope():
    assert link([reading(80, "diameter", 700, 760)], outline_with_holes(circular=True))[0].axis == "ab"


def sketch_with_values():
    img = np.full((1200, 1600, 3), 255, np.uint8)
    cv2.rectangle(img, (400, 300), (1000, 700), (0, 0, 0), 4)
    cv2.putText(img, "60", (660, 790), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 4)
    cv2.putText(img, "40", (250, 520), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 4)
    return img


def test_text_regions_find_values_written_outside_the_outline():
    img = sketch_with_values()
    boxes = text_regions(img, extract(img))
    centres = sorted((x + w // 2, y + h // 2) for x, y, w, h in boxes)
    assert len(boxes) == 2
    assert abs(centres[0][0] - 290) < 40 and abs(centres[1][1] - 770) < 40


def svc(reader=None, batch=None):
    """The Studio's reader setup: the batch (Qwen-VL) reader first, then the per-crop (TrOCR) reader."""
    readers = [r for r in (as_reader(batch, "qwen", calibrated=False, batch=True),
                           as_reader(reader, "trocr", calibrated=True, batch=False)) if r]
    return ReadingService(readers, cache=None)


def test_read_values_keeps_numbers_and_drops_words():
    img = sketch_with_values()
    o = extract(img)
    assert [r.value_mm for r in read_values(img, o, svc(lambda crop: ("60", 0.9)))] == [60.0, 60.0]
    assert read_values(img, o, svc(lambda crop: ("sixty", 0.9))) == []
    assert read_values(img, o, None) == []


@pytest.mark.gpu
def test_trocr_reads_printed_digits():
    pytest.importorskip("transformers")
    from s2c.multiview.ocr import trocr_reader
    img = np.full((80, 160, 3), 255, np.uint8)
    cv2.putText(img, "60", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 4)
    text, confidence = trocr_reader()(img)
    assert parse_value(text) == (60.0, "linear") and confidence > 0.3


def test_parse_value_never_joins_digits_across_a_space():
    assert parse_value("12 48") is None
    assert parse_value("0 1") is None
    assert parse_value("⌀ 8") == (8.0, "diameter")
    assert parse_value("12 mm") == (12.0, "linear")


def test_a_diameter_far_from_every_hole_is_not_linked():
    far, near = reading(6, "diameter", 1400, 1100), reading(6, "diameter", 1000, 680)
    linked = link([far, near], outline_with_holes())
    assert [(lv.axis, lv.hole_index) for lv in linked] == [(None, None), (None, 1)]


def run(name, calibrated, *reads):
    return ReaderRun(name, calibrated, [ReaderResult(text=t, confidence=c) for t, c in reads], "ok", 1)


def failed(name, calibrated):
    return ReaderRun(name, calibrated, None, "timeout", 1)


def test_decide_confirms_when_two_readers_give_the_same_number():
    text, parsed, _, confirmed = decide([run("qwen", False, ("⌀6", 0.9)), run("trocr", True, ("6", 0.95))], 0)
    assert (text, parsed, confirmed) == ("⌀6", (6.0, "diameter"), True)


def test_decide_leaves_a_disagreement_unconfirmed_with_the_first_readers_text():
    assert decide([run("qwen", False, ("60", 0.9)), run("trocr", True, ("80", 0.95))], 0)[::3] == ("60", False)


def test_a_batch_read_alone_is_unconfirmed_when_the_local_reader_fails():
    assert decide([run("qwen", False, ("60", 0.9)), failed("trocr", True)], 0)[::3] == ("60", False)
    assert decide([run("qwen", False, ("60", 0.9)), run("trocr", True, ("60", 0.5))], 0)[::3] == ("60", False)


def test_a_lone_reader_is_trusted_only_when_it_is_calibrated():
    assert decide([run("trocr", True, ("60", 0.9))], 0)[3] is True
    assert decide([run("trocr", True, ("60", 0.5))], 0) is None
    assert decide([run("qwen", False, ("60", 0.9))], 0)[3] is False


def test_decide_drops_a_crop_nobody_could_read():
    assert decide([run("qwen", False, ("", 0.9)), failed("trocr", True)], 0) is None


def test_read_values_sends_at_most_16_crops_nearest_the_part():
    img = np.full((1200, 1600, 3), 255, np.uint8)
    cv2.rectangle(img, (500, 400), (1100, 800), (0, 0, 0), 4)
    for k in range(10):  # a row of values under the part and a row far above it
        cv2.putText(img, "8", (480 + 70 * k, 900), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 4)
        cv2.putText(img, "8", (480 + 70 * k, 120), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 4)
    o = extract(img)
    assert len(text_regions(img, o)) == 20
    seen = []
    readings = read_values(img, o, svc(batch=lambda crops: seen.append(len(crops)) or [("8", 0.9)] * len(crops)))
    assert seen == [16] and sum(r.bbox[1] > 800 for r in readings) == 10


def test_read_values_marks_which_values_the_readers_agree_on():
    img = sketch_with_values()
    o = extract(img)

    def batch(crops):  # crops come nearest the part first: the "60" under it, then the "40" to its left
        return [("60", 0.9), ("40 mm", 0.9)]

    got = {r.value_mm: r.confirmed for r in read_values(img, o, svc(lambda crop: ("40", 0.95), batch))}
    assert got == {60.0: False, 40.0: True}


def test_a_number_spelled_glyph_by_glyph_is_one_value():
    """A handwriting reader spells a written number out ("2 5", "Ø 6"); two numbers are never joined ("12 48")."""
    assert parse_value("2 5") == (25.0, "linear")
    assert parse_value("Ø 6") == (6.0, "diameter")
    assert parse_value("1 2 5") == (125.0, "linear")
    assert parse_value("12 48") is None
    assert parse_value("1 25") is None


def test_a_stray_stroke_beside_a_number_is_not_part_of_it():
    assert parse_value("5 -") == (5.0, "linear")
    assert parse_value("' 60") == (60.0, "linear")
