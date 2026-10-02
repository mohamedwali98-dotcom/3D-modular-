"""The views of a split drawing sheet are named: labels first, then the layout, both held to the shared scale
(spec 3.2)."""
import cv2
import numpy as np
import pytest

from s2c.multiview.build import build
from s2c.multiview.sheet import (
    crop_views,
    find_symbol,
    ink_mask,
    label_face,
    name_views,
    named_count,
    split_sheet,
)
from s2c.multiview.spec import Envelope
from tests.mv_helpers import box_mesh, make_spec, outline
from tests.sheet_helpers import LABELS, draw_sheet, draw_symbol, part_views, relabel

ENV = Envelope(x_mm=80.0, y_mm=60.0, z_mm=40.0)
FIVE = ("front", "top", "right", "left", "back")
L_SIDE = [(0, 0), (40, 0), (40, 8), (8, 8), (8, 60), (0, 60)]


@pytest.fixture(scope="module")
def views():
    """All six views of an L-bracket: an upright plate at the front with a base leg running back."""
    return part_views(build(make_spec((ENV.x_mm, ENV.y_mm, ENV.z_mm), right=outline(L_SIDE))), ENV)


def pick(views, faces):
    return {f: views[f] for f in faces}


def off(a, b) -> int:
    return max(abs(p - q) for p, q in zip(a, b))


def index_of(views, box) -> int:
    i = min(range(len(views)), key=lambda i: off(views[i].box, box))
    assert off(views[i].box, box) <= 3, (views[i].box, box)
    return i


def named(sheet, naming, truth, faces) -> dict[str, str]:
    """Face given to the view drawn as each face."""
    views = sheet.drawings[naming.drawing].views
    return {f: naming.faces[index_of(views, truth[f])] for f in faces}


def fake_reader(img, truth, texts=LABELS):
    """Reads a label crop by finding it on the sheet and returning the text drawn there."""
    def read(crop):
        _, _, _, (x, y) = cv2.minMaxLoc(-cv2.matchTemplate(img, crop, cv2.TM_SQDIFF))
        cx, cy = x + crop.shape[1] / 2, y + crop.shape[0] / 2
        for key, (bx, by, bw, bh) in truth.items():
            if key.startswith("label:") and bx <= cx <= bx + bw and by <= cy <= by + bh:
                return texts[key[len("label:"):]], 0.9
        return "", 0.0
    return read


def test_layout_names_a_first_angle_sheet_without_labels(views):
    img, truth = draw_sheet(pick(views, FIVE), labels=False, layout="first")
    sheet = split_sheet(img)
    naming = name_views(sheet, img)
    assert named(sheet, naming, truth, FIVE) == {f: f for f in FIVE}
    assert (naming.projection, naming.projection_source) == ("first", "setting")
    assert named_count(naming) == 5


def test_layout_names_a_third_angle_sheet_with_the_switch(views):
    img, truth = draw_sheet(pick(views, FIVE), layout="third")
    sheet = split_sheet(img)
    naming = name_views(sheet, img, projection="third")
    assert named(sheet, naming, truth, FIVE) == {f: f for f in FIVE}
    assert naming.projection == "third"


def test_labels_name_the_views_when_read(views):
    """Drawn third-angle, set to first-angle: the layout alone would swap top and bottom, left and right."""
    img, truth = draw_sheet(pick(views, FIVE), layout="third")
    sheet = split_sheet(img)
    naming = name_views(sheet, img, projection="first", reader=fake_reader(img, truth))
    assert named(sheet, naming, truth, FIVE) == {f: f for f in FIVE}
    assert (naming.projection, naming.projection_source) == ("third", "labels")
    assert naming.warnings == ["The labels follow third-angle projection; the setting says first-angle. Used the labels."]


def test_a_label_that_does_not_match_its_size_is_overruled(views):
    """The user's sheet: the view above the front (first-angle bottom) is labelled "REAR VIEW" but has the top
    view's size, width by depth, not the rear's width by height."""
    faces = ("front", "top", "bottom", "right", "left")
    img, truth = draw_sheet(pick(views, faces))
    relabel(img, truth, "bottom", "REAR VIEW")
    sheet = split_sheet(img)
    texts = {**LABELS, "bottom": "REAR VIEW"}
    naming = name_views(sheet, img, reader=fake_reader(img, truth, texts))
    assert named(sheet, naming, truth, faces) == {f: f for f in faces}
    assert any("REAR VIEW does not match its size; used as bottom" in w for w in naming.warnings), naming.warnings


def test_two_views_labelled_alike_keep_the_consistent_one(views):
    faces = ("front", "top", "bottom", "right", "left")
    img, truth = draw_sheet(pick(views, faces))
    relabel(img, truth, "bottom", "TOP VIEW")
    sheet = split_sheet(img)
    naming = name_views(sheet, img, reader=fake_reader(img, truth, {**LABELS, "bottom": "TOP VIEW"}))
    assert named(sheet, naming, truth, faces) == {f: f for f in faces}
    assert any("TOP VIEW is also another view's name; used as bottom" in w for w in naming.warnings), naming.warnings


def test_labels_name_the_front_of_a_two_view_drawing(views):
    """Third-angle puts the top view above the front; set to first-angle, the layout alone takes the upper view
    for the front. The labels fit the scale with the lower view as the front, so they win."""
    faces = ("front", "top")
    img, truth = draw_sheet(pick(views, faces), layout="third")
    sheet = split_sheet(img)
    naming = name_views(sheet, img, projection="first", reader=fake_reader(img, truth))
    assert named(sheet, naming, truth, faces) == {f: f for f in faces}


def test_a_row_of_dimension_numbers_is_not_the_front(views):
    """Four equal chain dimensions line up like a row of views; they name more views than the part's two, but far
    less of the drawing."""
    faces = ("front", "top")
    img, truth = draw_sheet(pick(views, faces), labels=False)
    img = cv2.copyMakeBorder(img, 0, 60, 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255))
    x, y, _, h = truth["top"]
    for k in range(4):
        cv2.putText(img, "20", (x + 10 + 60 * k, y + h + 45), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    sheet = split_sheet(img)
    naming = name_views(sheet, img)
    assert len(sheet.drawings[naming.drawing].views) == 6
    assert named(sheet, naming, truth, faces) == {f: f for f in faces}
    assert sorted(naming.faces) == ["front", "skip", "skip", "skip", "skip", "top"]


def test_the_projection_symbol_sets_the_projection_and_is_not_the_part(views):
    img, truth = draw_sheet(pick(views, FIVE), layout="third", symbol="third")
    sheet = split_sheet(img)
    naming = name_views(sheet, img, projection="first")
    assert (naming.projection, naming.projection_source) == ("third", "symbol")
    assert len(sheet.drawings[naming.drawing].views) == 5
    assert named(sheet, naming, truth, FIVE) == {f: f for f in FIVE}
    symbol = next(i for i, d in enumerate(sheet.drawings) if len(d.views) == 2)
    assert find_symbol(sheet, img) == (symbol, "third")


def test_a_lone_symbol_is_the_part():
    img, truth = draw_sheet({}, symbol="first")
    sheet = split_sheet(img)
    naming = name_views(sheet, img, projection="third")
    assert len(sheet.drawings[naming.drawing].views) == 2
    assert (naming.projection, naming.projection_source) == ("first", "symbol")
    faces = named(sheet, naming, truth, ("symbol:side", "symbol:end"))
    assert faces == {"symbol:side": "front", "symbol:end": "left"}


def test_a_one_view_part_beside_a_symbol_is_the_part():
    """A plate drawn in one view with the projection symbol in the title block: the plate is the part, and one
    named view is not a sheet."""
    plate = part_views(box_mesh(80.0, 60.0, 4.0), Envelope(x_mm=80.0, y_mm=60.0, z_mm=4.0))
    img, truth = draw_sheet({"front": plate["front"]}, symbol="first")
    sheet = split_sheet(img)
    assert sorted(len(d.views) for d in sheet.drawings) == [1, 2]
    naming = name_views(sheet, img)
    assert (naming.projection, naming.projection_source) == ("first", "symbol")
    assert off(sheet.drawings[naming.drawing].views[0].box, truth["front"]) <= 3
    assert naming.faces == ["front"]
    assert named_count(naming) < 2


def test_a_symbol_drawn_with_centre_lines_is_still_the_symbol(views):
    img, truth = draw_sheet(pick(views, FIVE), layout="third", symbol="third", centre_lines=True)
    sheet = split_sheet(img)
    assert sorted(len(d.views) for d in sheet.drawings) == [2, 5]
    naming = name_views(sheet, img, projection="first")
    assert (naming.projection, naming.projection_source) == ("third", "symbol")
    assert len(sheet.drawings[naming.drawing].views) == 5
    assert named(sheet, naming, truth, FIVE) == {f: f for f in FIVE}

    img, _ = draw_sheet({}, symbol="first", centre_lines=True)
    assert find_symbol(split_sheet(img), img) == (0, "first")


@pytest.mark.parametrize("size, line, centre_lines", [(40, 2, True), (60, 3, True), (130, 1, False), (130, 2, True)])
def test_symbols_of_any_size_and_stroke_are_found(size, line, centre_lines):
    ink = np.zeros((size + 60, 3 * size + 60), np.uint8)
    draw_symbol(ink, "third", 30, 30, size, line, centre_lines)
    img = cv2.cvtColor(255 - ink, cv2.COLOR_GRAY2BGR)
    assert find_symbol(split_sheet(img), img) == (0, "third")


def test_a_symbol_whose_axis_runs_well_past_the_cone_is_found():
    """Drawn by hand, the cone's axis often runs a quarter of the symbol past its narrow end."""
    img, truth = draw_sheet({}, symbol="first")
    x, y, w, h = truth["symbol:side"]
    cv2.line(img, (x - h // 4, y + h // 2), (x + w // 2, y + h // 2), (0, 0, 0), 2)
    assert find_symbol(split_sheet(img), img) == (0, "first")


def test_a_flange_beside_a_cone_is_not_the_symbol():
    """Two concentric circles plus bolt holes: more than the symbol's end view."""
    img = np.full((220, 420, 3), 255, np.uint8)
    cv2.polylines(img, [np.array([(30, 80), (130, 30), (130, 190), (30, 140)], np.int32)], True, (0, 0, 0), 2)
    cv2.circle(img, (260, 110), 80, (0, 0, 0), 2)
    cv2.circle(img, (260, 110), 30, (0, 0, 0), 2)
    for dx, dy in ((0, -55), (55, 0), (0, 55), (-55, 0)):
        cv2.circle(img, (260 + dx, 110 + dy), 8, (0, 0, 0), 2)
    assert find_symbol(split_sheet(img), img) is None


@pytest.mark.parametrize("projection", ["first", "third"])
def test_a_symbol_with_its_narrow_end_on_the_right(projection):
    """Mirrored, the circles still sit beside the same end of the cone, so the projection is unchanged."""
    img, _ = draw_sheet({}, symbol=projection)
    img = np.ascontiguousarray(img[:, ::-1])
    sheet = split_sheet(img)
    assert find_symbol(sheet, img) == (0, projection)
    assert name_views(sheet, img, projection="third" if projection == "first" else "first").projection == projection


def test_a_label_that_fits_its_size_but_not_its_place_wins():
    """Depth equals height, so the view above the front has the rear's size too: the label is believed, and the
    warning says the layout disagrees."""
    faces = ("front", "top", "bottom", "right", "left")
    box = part_views(box_mesh(80.0, 60.0, 60.0), Envelope(x_mm=80.0, y_mm=60.0, z_mm=60.0))
    img, truth = draw_sheet(pick(box, faces))
    relabel(img, truth, "bottom", "REAR VIEW")
    sheet = split_sheet(img)
    naming = name_views(sheet, img, reader=fake_reader(img, truth, {**LABELS, "bottom": "REAR VIEW"}))
    got = named(sheet, naming, truth, faces)
    assert got == {"front": "front", "top": "top", "bottom": "back", "right": "right", "left": "left"}
    assert "REAR VIEW is where bottom belongs; used the label" in naming.warnings, naming.warnings


@pytest.mark.parametrize("face, text, why", [
    ("front", "TOP VIEW", "TOP VIEW is on the view read as the front; used as front"),
    ("back", "FRONT VIEW", "FRONT VIEW names the front, but another view is the front; used as back"),
])
def test_an_overruled_label_says_why(views, face, text, why):
    img, truth = draw_sheet(pick(views, FIVE))
    relabel(img, truth, face, text)
    sheet = split_sheet(img)
    naming = name_views(sheet, img, reader=fake_reader(img, truth, {**LABELS, face: text}))
    assert named(sheet, naming, truth, FIVE) == {f: f for f in FIVE}
    assert why in naming.warnings, naming.warnings
    assert not any("does not match its size" in w for w in naming.warnings)


@pytest.mark.parametrize("wrong", ["top", "left"])
def test_one_wrong_front_label_does_not_flip_the_naming(views, wrong):
    """The layout names all three views; a lone "FRONT VIEW" on another view is overruled, with a warning."""
    faces = ("front", "top", "left")
    img, truth = draw_sheet(pick(views, faces))
    texts = {f: "" for f in faces} | {wrong: "FRONT VIEW"}
    sheet = split_sheet(img)
    naming = name_views(sheet, img, reader=fake_reader(img, truth, texts))
    assert named(sheet, naming, truth, faces) == {f: f for f in faces}
    assert any(w.startswith("FRONT VIEW") for w in naming.warnings), naming.warnings


def test_an_unnamable_view_stays_auto(views):
    """A view as large as a real one but in line with nothing: its name would be a guess."""
    img, truth = draw_sheet(pick(views, FIVE), labels=False)
    img = cv2.copyMakeBorder(img, 0, 120, 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255))
    left, back, top = truth["left"], truth["back"], truth["top"]
    x = (left[0] + left[2] + back[0]) // 2 - 75
    y = top[1] + top[3] // 2 + 25
    cv2.rectangle(img, (x, y), (x + 150, y + 110), (0, 0, 0), 2)
    sheet = split_sheet(img)
    naming = name_views(sheet, img)
    views_ = sheet.drawings[naming.drawing].views
    assert len(views_) == 6
    assert naming.faces[index_of(views_, (x - 1, y - 1, 153, 113))] == "auto"
    assert named(sheet, naming, truth, FIVE) == {f: f for f in FIVE}
    assert named_count(naming) == 5


def test_small_marks_beside_the_front_are_skipped(views):
    """Split keeps dimension numbers as views; naming skips them, and the crops leave them out."""
    img, truth = draw_sheet(pick(views, FIVE), labels=False)
    fx, fy, fw, fh = truth["front"]
    lx = truth["left"][0]
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, "80", (fx + fw // 2 - 16, fy - 22), font, 0.8, (0, 0, 0), 2)  # above, in the front's column
    cv2.putText(img, "60", ((fx + fw + lx) // 2 - 14, fy + fh // 2 + 8), font, 0.6, (0, 0, 0), 2)  # in its row
    cv2.putText(img, "25", (lx + 20, truth["top"][1] + 40), font, 0.8, (0, 0, 0), 2)  # in line with nothing
    sheet = split_sheet(img)
    naming = name_views(sheet, img)
    views_ = sheet.drawings[naming.drawing].views
    assert len(views_) == 8
    assert named(sheet, naming, truth, FIVE) == {f: f for f in FIVE}
    marks = [i for i, v in enumerate(views_) if min(off(v.box, truth[f]) for f in FIVE) > 3]
    assert [naming.faces[i] for i in marks] == ["skip", "skip", "skip"]
    assert sum("3 small marks" in w for w in naming.warnings) == 1, naming.warnings
    crops = crop_views(sheet, img, naming)
    assert [face for _, face in crops] == [f for f in naming.faces if f != "skip"]
    assert named_count(naming) == 5


def test_a_thin_edge_view_is_still_named():
    """A plate's edge views are far under 15 % of the front's area, but they line up and pass the scale check."""
    env = Envelope(x_mm=80.0, y_mm=60.0, z_mm=4.0)
    faces = ("front", "top", "right")
    img, truth = draw_sheet(pick(part_views(box_mesh(80.0, 60.0, 4.0), env), faces), labels=False)
    sheet = split_sheet(img)
    naming = name_views(sheet, img)
    assert named(sheet, naming, truth, faces) == {f: f for f in faces}

    washer = np.full((360, 520, 3), 255, np.uint8)
    cv2.circle(washer, (160, 180), 100, (0, 0, 0), 2)
    cv2.circle(washer, (160, 180), 40, (0, 0, 0), 2)
    cv2.rectangle(washer, (340, 80), (356, 280), (0, 0, 0), 2)
    sheet = split_sheet(washer)
    naming = name_views(sheet, washer)
    assert naming.faces == ["front", "left"] or naming.faces == ["left", "front"]
    front = sheet.drawings[naming.drawing].views[naming.faces.index("front")]
    assert front.box[0] < 100


def test_centre_lines_do_not_change_a_views_size():
    """Chain lines run past the outline (spec 2); the end view's height is still the side view's height."""
    img = np.full((360, 640, 3), 255, np.uint8)
    cv2.rectangle(img, (60, 120), (300, 240), (0, 0, 0), 2)
    cv2.line(img, (44, 180), (316, 180), (0, 0, 0), 1)
    cv2.circle(img, (480, 180), 60, (0, 0, 0), 2)
    cv2.line(img, (404, 180), (556, 180), (0, 0, 0), 1)
    cv2.line(img, (480, 104), (480, 256), (0, 0, 0), 1)
    sheet = split_sheet(img)
    naming = name_views(sheet, img)
    assert len(naming.faces) == 2
    left_first = sorted(range(2), key=lambda i: sheet.drawings[naming.drawing].views[i].box[0])
    assert [naming.faces[i] for i in left_first] == ["front", "left"]


def test_crops_keep_the_view_and_its_scale(views):
    img, _ = draw_sheet(pick(views, FIVE), centre_lines=True)
    sheet = split_sheet(img)
    naming = name_views(sheet, img)
    crops = crop_views(sheet, img, naming)
    views_ = sheet.drawings[naming.drawing].views
    assert [face for _, face in crops] == naming.faces
    for (png, _), view in zip(crops, views_):
        crop = cv2.imdecode(np.frombuffer(png, np.uint8), cv2.IMREAD_COLOR)
        x, y, w, h = cv2.boundingRect(ink_mask(crop))
        assert abs(w - view.box[2]) <= 2 and abs(h - view.box[3]) <= 2, ((w, h), view.box)
        margin = round(0.04 * max(view.box[2], view.box[3]))
        assert abs(x - margin) <= 2 and abs(y - margin) <= 2
        assert (crop[:margin] == 255).all() and (crop[:, :margin] == 255).all()


@pytest.mark.parametrize("text, face", [
    ("Right Side View", "right"), ("REAR VIEW", "back"), ("Plan", "top"), ("Front Elevation", "front"),
    ("Frnt view", "front"), ("Section A-A", None),
    ("End Elevation", None), ("LEFT SIDE VIEW", "left"), ("BOTTOM VIEW", "bottom"), ("T0P VIEW", "top"),
    ("Tip view", None), ("Front and top", None), ("", None),
])
def test_label_face_keywords(text, face):
    assert label_face(text) == face
