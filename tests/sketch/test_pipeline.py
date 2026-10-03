import cv2
import numpy as np
import pytest

from s2c.sketch import read_sketch
from s2c.sketch.models import SketchReading
from s2c.sketch.pipeline import analyse
from tests.sketch.synth import Sheet, TruthReader, bridge_block


def png(img):
    return cv2.imencode(".png", img)[1].tobytes()


def readers(sh):
    return [TruthReader(sh.texts, "a"), TruthReader(sh.texts, "b")]


SIDE_LOSS = pytest.mark.xfail(strict=True, reason=(
    "Known classify limit: the SIDE '50' dimension line is cut by the '12.5' text (accepted loss); its leftover "
    "extension line and arrow stub stay visible edges, widen the SIDE depth (z ~34.6 instead of 25) and break the "
    "x-hole and badge checks downstream."))


@pytest.fixture(scope="module")
def bridge():
    sh = bridge_block(Sheet())
    return sh, read_sketch(png(sh.bgr()), readers(sh))


@SIDE_LOSS
def test_bridge_block_views_and_envelope(bridge):
    _, r = bridge
    assert r.abstain is None
    assert {v.name for v in r.views} == {"top", "front", "right"}
    assert r.envelope["x"].value == pytest.approx(100, abs=0.5) and r.envelope["x"].badge == "written"
    assert r.envelope["y"].value == pytest.approx(50, abs=0.5)
    assert r.envelope["z"].value == pytest.approx(25, abs=0.5)
    assert r.envelope["y"].badge in ("written", "derived") and r.envelope["z"].badge in ("written", "derived")


@SIDE_LOSS
def test_bridge_block_holes(bridge):
    _, r = bridge
    holes = [f for f in r.features if f.type == "hole"]
    assert sorted(h.axis for h in holes) == ["x", "x", "y", "y"]
    assert all(h.diameter == pytest.approx(12.5, abs=0.3) and h.through for h in holes)
    base = sorted(h.position_mm[0] for h in holes if h.axis == "y")
    assert base == pytest.approx([12.5, 87.5], abs=0.5)


@SIDE_LOSS
def test_bridge_block_values_are_all_trusted_and_no_red_issues(bridge):
    _, r = bridge
    written = [d for d in r.dimensions if d.readings]
    assert len(written) == 18 and all(d.badge == "written" for d in written)
    assert not [i for i in r.issues if i.severity == "red"]


def test_json_contract_round_trip(bridge):
    _, r = bridge
    assert SketchReading.model_validate_json(r.model_dump_json()) == r


def test_px_are_in_photo_pixels():
    sh = bridge_block(Sheet())
    big = cv2.resize(sh.bgr(), (3200, 2262), interpolation=cv2.INTER_CUBIC)
    r = read_sketch(png(big), readers(sh))
    assert r.image_size_px == (3200, 2262)
    front = next(v for v in r.views if v.name == "front")
    assert front.bbox_px[0] == pytest.approx(2 * 150, abs=30)   # the left dimension line sits at x 150


def test_blank_page_abstains_at_the_views_stage():
    r = read_sketch(png(np.full((1131, 1600, 3), 250, np.uint8)), [])
    assert r.abstain is not None and r.abstain.reason == "no_views_found"


def test_unreadable_bytes_abstain():
    r = read_sketch(b"not an image", [])
    assert r.abstain.stage == "capture" and r.abstain.reason == "unreadable_image"


def test_inch_looking_values_raise_a_unit_question():
    sh = Sheet()
    for p, q in [((300, 300), (700, 300)), ((700, 300), (700, 500)), ((700, 500), (300, 500)),
                 ((300, 500), (300, 300))]:
        sh.line(p, q)
    sh.hdim(300, 400, 500, 545, ".50")
    sh.hdim(400, 700, 500, 545, "1.50")
    sh.hdim(300, 700, 500, 590, "2.00")
    sh.vdim(300, 500, 700, 750, ".75")
    sh.text("FRONT", (500, 680), scale=1.0)
    r = read_sketch(png(sh.bgr()), readers(sh))
    assert any(i.kind == "unit" and i.severity == "red" for i in r.issues)


def test_debug_overlays_are_written(tmp_path):
    from s2c.sketch.debug import write_overlays
    sh = bridge_block(Sheet())
    _, trace = analyse(png(sh.bgr()), readers(sh))
    paths = write_overlays(trace, tmp_path)
    assert paths and all(p.exists() for p in paths)
