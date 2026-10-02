"""One sketch to a 3D model in five steps: page, faces, labels, generate the rest, numbers, model
(sketch-to-model spec 2026-10-02)."""
from pathlib import Path

import cv2
import numpy as np

from s2c.multiview import spec as S
from s2c.multiview.pipeline import MvPipeline
from s2c.multiview.sheet import split_by_outlines
from s2c.multiview.sheet_read import link_sizes, page_of, read_drawing, read_sheet
from tests.hand_views import hand_photo
from tests.line_views import drawing_sheet
from tests.test_mv_relief import _block, _cut, _pins
from tests.test_mv_sheet_read import _service

THIRD = ("front", "top", "right")
GOLDEN = Path(__file__).resolve().parent / "golden_sketch"


def _part():
    return _pins(_cut(_cut(_block(), 0, 40, 50, 15, 60, 70), 65, 40, 50, 80, 60, 70), z=20)


def test_a_clean_drawing_is_used_as_drawn():
    img, _, _ = drawing_sheet(_part(), faces=THIRD, layout="third")
    page = page_of(img)
    assert page.kind == "drawing" and np.array_equal(page.image, img)


def test_a_hand_photo_becomes_a_clean_page():
    photo, _ = hand_photo(_part(), faces=THIRD, layout="third")
    page = page_of(photo)
    assert page.kind == "sketch" and max(page.image.shape[:2]) == 1600
    gray = cv2.cvtColor(page.image, cv2.COLOR_BGR2GRAY)
    assert set(np.unique(gray)) <= {0, 255}          # black ink on white
    corner = cv2.perspectiveTransform(np.float32([[[0, 0]]]), page.to_photo)[0, 0]
    assert 0 <= corner[0] < photo.shape[1] and 0 <= corner[1] < photo.shape[0]


def test_a_dark_photo_abstains_with_a_remedy():
    page = page_of(np.full((900, 1200, 3), 20, np.uint8))
    assert isinstance(page, S.MvAbstain) and page.remedy


def test_views_joined_by_dimension_and_miter_lines_are_three_faces():
    photo, _ = hand_photo(_part(), faces=THIRD, layout="third", miter=True)
    page = page_of(photo)
    sheet = split_by_outlines(page.image, page.stroke_px)
    big = [v for d in sheet.drawings for v in d.views if v.box[2] * v.box[3] > 0.02 * page.image.size / 3]
    assert len(big) == 3


def test_the_real_sketch_has_three_faces():
    page = page_of(cv2.imread(str(GOLDEN / "real_bracket_1" / "image.jpg")))
    sheet = split_by_outlines(page.image, page.stroke_px)
    area = page.image.shape[0] * page.image.shape[1]
    big = [v for d in sheet.drawings for v in d.views if v.box[2] * v.box[3] > 0.01 * area]
    assert len(big) == 3


def test_dimension_strips_are_not_part_of_a_face():
    """A dimension line whose extension lines meet the outline closes a strip onto the view; with its number beside
    it, the strip is taken off again, so views that share a size have the same box size."""
    for seed in range(3):
        photo, _ = hand_photo(_part(), faces=THIRD, layout="third", miter=True, seed=seed)
        page = page_of(photo)
        views = [v.box for d in split_by_outlines(page.image, page.stroke_px).drawings for v in d.views]
        top = min(views, key=lambda b: b[1])
        front, right = sorted((b for b in views if b != top), key=lambda b: b[0])
        assert abs(top[2] - front[2]) <= 0.03 * front[2]
        assert abs(front[3] - right[3]) <= 0.03 * front[3]


def test_a_hand_sketch_is_read_into_named_faces():
    photo, _ = hand_photo(_part(), faces=THIRD, layout="third", miter=True)
    read = read_drawing(photo, "third")
    assert read.kind == "sketch" and sorted(c.face for c in read.crops) == sorted(THIRD)


def test_a_clean_drawing_reads_exactly_as_before():
    img, _, words = drawing_sheet(_part(), faces=THIRD, layout="third", iso=True)
    a = read_sheet(img, "auto", service=_service(words))
    b = read_drawing(img, "auto", service=_service(words))
    assert b.kind == "drawing" and [c.face for c in a.crops] == [c.face for c in b.crops]
    assert a.scale.mm_per_px == b.scale.mm_per_px


def test_the_real_sketch_names_its_three_faces():
    """Third-angle set: top, front, right. On auto, the ISO default with the setting as the source is accepted
    when the drawing does not decide (spec 2)."""
    img = cv2.imread(str(GOLDEN / "real_bracket_1" / "image.jpg"))
    third = read_drawing(img, "third")
    assert sorted(f for f in third.naming.faces if f != "auto") == sorted(THIRD)
    auto = read_drawing(img, "auto")
    faces = sorted(auto.naming.faces)
    assert faces == sorted(THIRD) or (faces == sorted(("bottom", "front", "left"))
                                      and auto.naming.projection_source == "setting")


def test_a_wobbly_notched_block_reads_its_notch():
    part = _cut(_block(), 0, 40, 50, 15, 60, 70)
    for seed in range(3):  # three photos: wobble, slant and the dash pattern differ
        photo, _ = hand_photo(part, faces=THIRD, layout="third", seed=seed)
        read = read_drawing(photo, "third")
        pipe = MvPipeline()
        spec = pipe.fuse(pipe.observe(read.inputs()), {"envelope.x_mm": 80, "envelope.y_mm": 60, "envelope.z_mm": 70})
        pockets = [f for f in spec.features if f.type == "pocket"]
        assert len(pockets) == 1 and abs(pockets[0].depth_mm - 20) <= 2, seed


def test_straightening_keeps_a_hole_round():
    part = _block().faces(">Z").workplane(centerOption="CenterOfBoundBox").hole(12)
    photo, _ = hand_photo(part, faces=THIRD, layout="third")
    read = read_drawing(photo, "third")
    obs = MvPipeline().observe(read.inputs())
    front = next(o for o in obs.observations if o.face == "front")
    assert len(front.outline.circles) == 1


def test_a_sketch_builds_from_its_written_sizes():
    photo, words = hand_photo(_block(), faces=THIRD, layout="third")
    read = read_drawing(photo, "third", service=_service(words))
    pipe = MvPipeline()
    observed = pipe.observe(read.inputs())
    link_sizes(read, observed)
    spec = pipe.fuse(observed)
    assert (spec.envelope.x_mm, spec.envelope.y_mm, spec.envelope.z_mm) == (80, 60, 70)
    assert spec.provenance["envelope.x_mm"] == "user_written"


def test_a_sketch_without_numbers_suggests_and_never_measures():
    photo, _ = hand_photo(_block(), faces=THIRD, layout="third", dims=False)
    read = read_drawing(photo, "third")
    result = MvPipeline().fuse(MvPipeline().observe(read.inputs()))
    assert isinstance(result, S.MvAbstain) and result.stage == "dimensions"


def test_a_part_dimension_is_read_but_not_used_as_a_size():
    part = _cut(_block(), 0, 40, 50, 15, 60, 70)
    # one more dimension, over the notch's 15 mm width only (front view, along a, above the view)
    photo, words = hand_photo(part, faces=THIRD, layout="third", extra=[("front", 0, 15, "a", "15")])
    read = read_drawing(photo, "third", service=_service(words))
    pipe = MvPipeline()
    observed = pipe.observe(read.inputs())
    link_sizes(read, observed)
    spec = pipe.fuse(observed)
    assert spec.envelope.x_mm == 80
    assert any("Read 15" in w and "not used" in w for w in spec.warnings)


def test_a_sheets_views_are_not_read_again_for_numbers():
    """The numbers of a sheet are read once, on the sheet; a cropped view never reads its stubs and ticks as
    sizes (a "1" from a tick would be a silently wrong size)."""
    photo, words = hand_photo(_block(), faces=THIRD, layout="third")
    read = read_drawing(photo, "third", service=_service(words))
    assert all(not i.numbers for i in read.inputs())


def test_a_lone_straight_stroke_is_no_number():
    """A dash of the miter line reads as "1"; off any dimension line, a single straight stroke is not read."""
    import s2c.multiview.dimensions as dm
    ink = np.zeros((200, 300), np.uint8)
    cv2.line(ink, (100, 150), (130, 120), 255, 3)          # a slanted dash
    cv2.putText(ink, "80", (180, 120), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 1.0, 255, 2)
    assert dm._stroke_mark(ink, (95, 115, 40, 40), 3.0)
    assert not dm._stroke_mark(ink, (175, 95, 60, 35), 3.0)


def test_one_dimension_line_carries_one_value():
    """A second number near a dimension line's end (another callout) is read, but the line's value is the one
    written over its middle."""
    import s2c.multiview.dimensions as dm
    line = ("h", 100.0, 0, 400)
    dims = [dm.Dimension(60.0, "linear", "60", (180, 70, 40, 25), 401.0, True, line),
            dm.Dimension(6.0, "linear", "6", (405, 105, 20, 25), 401.0, True, line)]
    kept = dm._one_per_line(dims)
    assert [d.text for d in kept if d.line] == ["60"] and len(kept) == 2


def test_the_real_sketch_end_to_end(tmp_path):
    """The phone photo of a pen sketch (tests/golden_sketch/real_bracket_1) with the real handwriting reader:
    three faces named, the clear numbers read, each overall size the user's own, a misread number never used
    silently, and the part builds once the user confirms what was not read. Skipped without the trocr extra."""
    import json

    import pytest
    pytest.importorskip("transformers")
    from s2c.multiview.pipeline import default_pipeline
    from s2c.multiview.sheet_read import link_diameters

    pipe = default_pipeline()
    if pipe.reader is None and pipe.batch_reader is None:
        pytest.skip("no reader configured")
    truth = json.loads((GOLDEN / "real_bracket_1" / "expected.json").read_text())
    read = read_drawing(cv2.imread(str(GOLDEN / "real_bracket_1" / "image.jpg")), "auto", reader=pipe.reader,
                        service=pipe.reading())
    faces = sorted(f for f in read.naming.faces if f != "auto")
    assert faces == sorted(THIRD) or (faces == sorted(("bottom", "front", "left"))
                                      and read.naming.projection_source == "setting")
    wanted = [(d["value"], d["kind"]) for d in truth["dimensions"]]
    got = [(d.value_mm, d.kind) for d in read.scale.dimensions]
    matched = sum(min(wanted.count(v), got.count(v)) for v in set(wanted))
    assert matched >= 3, got  # 60, 25 and 3 today; the Ø callouts tangled with their leaders are not yet read
    observed = pipe.observe(read.inputs())
    link_sizes(read, observed)
    link_diameters(read, observed)
    draft = pipe.fuse(observed)
    known = draft.partial["known"] if isinstance(draft, S.MvAbstain) else draft.envelope.model_dump()
    assert known["envelope.x_mm" if isinstance(draft, S.MvAbstain) else "x_mm"] == 60
    assert known["envelope.z_mm" if isinstance(draft, S.MvAbstain) else "z_mm"] == 25
    spec = pipe.fuse(observed, {"envelope.y_mm": 25})
    assert spec.provenance["envelope.x_mm"] == "user_written"
    built = pipe.build(spec, tmp_path)
    assert not isinstance(built, S.MvAbstain), built


def test_a_busy_sketch_view_is_still_read_as_lines():
    """A sketch view dense with hidden lines is a line drawing all the same (it was drawn again from its strokes):
    its outline has no openings made of the gaps between lines."""
    read = read_drawing(cv2.imread(str(GOLDEN / "real_bracket_1" / "image.jpg")), "third")
    assert all(i.line_art for i in read.inputs())
    obs = MvPipeline().observe(read.inputs())
    assert all(o.outline.line_art and not o.outline.inner for o in obs.observations)
