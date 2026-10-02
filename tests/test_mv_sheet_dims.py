"""Dimensioned sheets: annotations stay out of the views; dimensions give the sheet's scale (sheet-reading spec)."""
import cv2
import numpy as np
import pytest

from s2c.multiview.sheet import crop_views, name_views, split_sheet, view_body
from tests.line_views import drawing_sheet
from tests.test_mv_relief import _block, _cut, _pins


def _part():
    return _pins(_cut(_cut(_block(), 0, 40, 50, 15, 60, 70), 65, 40, 50, 80, 60, 70), z=20)


@pytest.fixture(scope="module")
def third():
    return drawing_sheet(_part(), layout="third", faces=("front", "top", "right"), iso=True)


def _close(a, b, tol=6):
    return all(abs(p - q) <= tol for p, q in zip(a, b, strict=True))


def test_a_view_body_leaves_its_dimensions_out(third):
    img, boxes, _ = third
    sheet = split_sheet(img)
    ink = (cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) < 128).astype(np.uint8) * 255
    for view in sheet.drawings[0].views:
        body, _ = view_body(ink, view.box, max(img.shape[:2]))
        hits = [f for f, b in boxes.items() if _close(body, b)]
        assert hits or body[2] * body[3] < 2000, (view.box, body)


def test_named_views_of_a_dimensioned_sheet(third):
    img, _, _ = third
    sheet = split_sheet(img)
    naming = name_views(sheet, img, "third")
    faces = sorted(f for f in naming.faces if f not in ("auto", "skip"))
    assert faces == ["front", "right", "top"], naming.faces


def test_crops_hold_no_dimension_lines(third):
    img, boxes, _ = third
    sheet = split_sheet(img)
    naming = name_views(sheet, img, "third")
    for png, face in crop_views(sheet, img, naming):
        if face in ("auto", "skip"):
            continue
        crop = cv2.imdecode(np.frombuffer(png, np.uint8), cv2.IMREAD_GRAYSCALE)
        ys, xs = np.nonzero(crop < 128)
        w, h = xs.max() - xs.min() + 1, ys.max() - ys.min() + 1
        tw, th = boxes[face][2], boxes[face][3]
        assert abs(w - tw) <= 6 and abs(h - th) <= 6, (face, (w, h), (tw, th))


def _bodies(img, sheet):
    from s2c.multiview.sheet import ink_mask
    ink = ink_mask(img)
    long = max(img.shape[:2])
    return ink, [view_body(ink, v.box, long) for d in sheet.drawings for v in d.views]


def _service(words):
    from s2c.reading import ReadingService
    from tests.line_views import WordReader
    return ReadingService([WordReader(words)], cache=None)


def test_the_dimensions_give_the_sheet_scale(third):
    from s2c.multiview.dimensions import read_dimensions
    img, _, words = third
    ink, bodies = _bodies(img, split_sheet(img))
    scale = read_dimensions(img, ink, bodies, _service(words))
    assert scale.mm_per_px == pytest.approx(0.25, rel=0.01), scale
    assert len(scale.used) == 6 and not scale.rejected
    assert {d.value_mm for d in scale.used} == {80, 70, 66}


def test_a_misread_dimension_is_rejected_and_named(third):
    from s2c.multiview.dimensions import read_dimensions
    img, _, words = third
    wrong = [(box, "18" if k == 0 else t) for k, (box, t) in enumerate(words)]  # 80 read as 18
    ink, bodies = _bodies(img, split_sheet(img))
    scale = read_dimensions(img, ink, bodies, _service(wrong))
    assert scale.mm_per_px == pytest.approx(0.25, rel=0.01)
    assert [d.text for d in scale.rejected] == ["18"]
    assert any("18" in w for w in scale.warnings)
    assert scale.confirmed is False  # one disagrees: the sizes come out for the user to confirm


def test_six_agreeing_dimensions_confirm_the_scale(third):
    from s2c.multiview.dimensions import read_dimensions
    img, _, words = third
    ink, bodies = _bodies(img, split_sheet(img))
    assert read_dimensions(img, ink, bodies, _service(words)).confirmed is True


def test_one_dimension_gives_a_scale_to_confirm(third):
    from s2c.multiview.dimensions import read_dimensions
    img, _, words = third
    ink, bodies = _bodies(img, split_sheet(img))
    scale = read_dimensions(img, ink, bodies, _service(words[:1]))
    assert scale.mm_per_px == pytest.approx(0.25, rel=0.01) and scale.confirmed is False


def test_two_dimensions_that_disagree_give_no_scale(third):
    from s2c.multiview.dimensions import read_dimensions
    img, _, words = third
    ink, bodies = _bodies(img, split_sheet(img))
    scale = read_dimensions(img, ink, bodies, _service([words[0], (words[2][0], "18")]))
    assert scale.mm_per_px is None
    assert any("do not agree" in w for w in scale.warnings)


def test_an_unconfirmed_scale_suggests_the_sizes_instead_of_trusting_them():
    from s2c.multiview import spec as S
    from s2c.multiview.pipeline import ImageInput, MvPipeline
    from tests.line_views import draw_view
    pipe = MvPipeline()
    images = []
    for f in ("front", "top", "right"):
        ink = cv2.copyMakeBorder(draw_view(_block(), f, 4.0), 30, 30, 30, 30, cv2.BORDER_CONSTANT, value=0)
        images.append(ImageInput(cv2.imencode(".png", 255 - ink)[1].tobytes(), f, "drawing", mm_per_px=0.25,
                                 scale_confirmed=False))
    result = pipe.fuse(pipe.observe(images))
    assert isinstance(result, S.MvAbstain) and result.stage == "dimensions"
    suggested = result.partial["suggested"]
    assert suggested["envelope.x_mm"] == pytest.approx(80, rel=0.02)


def test_no_reader_no_scale(third):
    from s2c.multiview.dimensions import read_dimensions
    img, _, _ = third
    ink, bodies = _bodies(img, split_sheet(img))
    assert read_dimensions(img, ink, bodies, None).mm_per_px is None


def test_views_with_a_known_scale_build_with_no_typed_size():
    """ImageInput.mm_per_px (the image's own pixels): the envelope comes out measured, so nothing needs typing."""
    from s2c.multiview import spec as S
    from s2c.multiview.pipeline import ImageInput, MvPipeline
    from tests.line_views import draw_view
    part = _block()
    pipe = MvPipeline()
    images = []
    for f in ("front", "top", "right"):
        ink = cv2.copyMakeBorder(draw_view(part, f, 4.0), 30, 30, 30, 30, cv2.BORDER_CONSTANT, value=0)
        images.append(ImageInput(cv2.imencode(".png", 255 - ink)[1].tobytes(), f, "drawing", mm_per_px=0.25))
    spec = pipe.fuse(pipe.observe(images))
    assert isinstance(spec, S.MultiViewSpec), spec
    env = spec.envelope
    assert (env.x_mm, env.y_mm, env.z_mm) == pytest.approx((80, 60, 70), rel=0.02)
    assert spec.provenance["envelope.x_mm"] == "measured"


class _Counting:
    """WordReader that also counts the crops it is asked to read."""
    name, calibrated = "count", True

    def __init__(self, words):
        from tests.line_views import WordReader
        self.inner, self.crops = WordReader(words), 0

    def read(self, crops):
        self.crops += len(crops)
        return self.inner.read(crops)


def test_every_dimension_of_a_first_angle_sheet_is_read():
    """Digits with loops (0, 6, 8, 9) are text, never a view's body: both vertical 66s are read."""
    from s2c.multiview.dimensions import annotation_ink
    from s2c.multiview.sheet import name_views
    from s2c.multiview.sheet_read import part_bodies
    img, _, words = drawing_sheet(_part(), layout="first", faces=("front", "top", "left"), iso=True)
    sheet = split_sheet(img)
    ink, bodies = part_bodies(sheet, img, name_views(sheet, img, "first"))
    notes = annotation_ink(ink, bodies)
    for (x, y, w, h), text in words:  # every pixel of every written value stays annotation, readable
        assert np.count_nonzero(notes[y: y + h, x: x + w]) >= 0.95 * np.count_nonzero(ink[y: y + h, x: x + w]), text


def test_only_words_beside_a_dimension_line_are_read(third):
    """Arrowheads and stray marks are not sent to the reader: it is slow, and a mark read as "4" would be trouble."""
    from s2c.multiview.dimensions import read_dimensions
    from s2c.reading import ReadingService
    img, _, words = third
    ink, bodies = _bodies(img, split_sheet(img))
    counting = _Counting(words)
    read_dimensions(img, ink, bodies, ReadingService([counting], cache=None))
    assert counting.crops <= 3 * len(words), counting.crops


def test_extension_lines_touching_the_outline_stay_out_of_the_view():
    """Hand and some CAD drawings start extension lines on the outline: the dimension line closes a loop with it,
    which must not be filled into the view."""
    from s2c.multiview.sheet_read import read_sheet
    img, boxes, words = drawing_sheet(_block(), layout="third", faces=("front", "top", "right"), ext_gap=0)
    sheet = split_sheet(img)
    ink, _ = _bodies(img, sheet)
    long = max(img.shape[:2])
    for view in sheet.drawings[0].views:
        body, _ = view_body(ink, view.box, long)
        if body[2] * body[3] > 5000:
            assert any(_close(body, b) for b in boxes.values()), (view.box, body, boxes)
    read = read_sheet(img, "third", service=_service(words))
    assert read.scale.mm_per_px == pytest.approx(0.25, rel=0.01)


def test_a_thin_flange_stays_part_of_its_view():
    """A 1.5 mm flange drawn at 2 px/mm is about a line width thick, but it encloses area: it is the part."""
    import cadquery as cq
    part = cq.Workplane("XY").box(30, 1.5, 20, centered=False).union(
        cq.Workplane("XY").box(10, 20, 20, centered=False).translate((10, 1.5, 0)))
    img, boxes, _ = drawing_sheet(part, layout="first", faces=("front", "top", "left"), px_per_mm=2.0, dims=False)
    sheet = split_sheet(img)
    ink, _ = _bodies(img, sheet)
    long = max(img.shape[:2])
    front = next(v for v in sheet.drawings[0].views if _close(v.box, boxes["front"], 8))
    body, _ = view_body(ink, front.box, long)
    assert _close(body, boxes["front"], 6), (body, boxes["front"])
