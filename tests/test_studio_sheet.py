"""Drawing sheets in the Studio: an upload is split into its views, the projection switch renames them, and the
"Try a drawing sheet" example builds (drawing-sheet spec 3.6)."""
import os
import re
import time
from pathlib import Path

import cv2
import numpy as np
import pytest

from s2c.multiview import artifacts
from s2c.multiview.build import build
from s2c.multiview.pipeline import MvPipeline
from s2c.multiview.settings import AiSettings, GeometrySettings
from s2c.multiview.sheet import is_sheet, name_views, named_count, split_sheet
from s2c.multiview.spec import Envelope
from s2c.studio import handlers
from s2c.studio.handlers import AXIS_LABEL, FACE_CHOICES, Studio
from s2c.studio.theme import FACE_BADGES
from tests.mv_helpers import make_spec, outline
from tests.sheet_helpers import draw_sheet, part_views

ENV = Envelope(x_mm=80.0, y_mm=60.0, z_mm=40.0)
FIVE = ("front", "top", "right", "left", "back")
L_SIDE = [(0, 0), (40, 0), (40, 8), (8, 8), (8, 60), (0, 60)]
SHEET_DIR = Path(handlers.__file__).resolve().parents[2] / "examples" / "mv" / "sheet"


@pytest.fixture(scope="module")
def views():
    """All six views of an L-bracket: an upright plate at the front with a base leg running back."""
    return part_views(build(make_spec((ENV.x_mm, ENV.y_mm, ENV.z_mm), right=outline(L_SIDE))), ENV)


@pytest.fixture
def studio(tmp_path, monkeypatch):
    artifacts.clear_cache()
    monkeypatch.setattr("s2c.multiview.slice.find_slicer", lambda: None)
    return Studio(MvPipeline(), root=tmp_path / "files")


def pick(views, faces):
    return {f: views[f] for f in faces}


def off(a, b) -> int:
    return max(abs(p - q) for p, q in zip(a, b))


def save(tmp_path, img, name="sheet.png") -> str:
    path = tmp_path / name
    path.write_bytes(cv2.imencode(".png", img)[1].tobytes())
    return str(path)


def drawn_as(studio, sid, truth, faces) -> dict:
    """The item made from the view drawn as each face: its view's box is that face's truth box."""
    session = studio.store.get(sid)
    out = {}
    for item in session.items:
        _, sheet, naming = session.sheets[item.sheet_id]
        box = sheet.drawings[naming.drawing].views[item.view].box
        face = min(faces, key=lambda f: off(box, truth[f]))
        assert off(box, truth[face]) <= 3, (box, truth[face])
        out[face] = item
    assert len(out) == len(session.items)
    return out


def test_dropping_a_sheet_gives_one_item_per_view(studio, tmp_path, views):
    img, truth = draw_sheet(pick(views, FIVE))
    sid = studio.store.new()
    studio.add_images(sid, [save(tmp_path, img)])
    session = studio.store.get(sid)
    assert len(session.items) == 5
    items = drawn_as(studio, sid, truth, FIVE)
    assert {f: item.face for f, item in items.items()} == {f: f for f in FIVE}
    assert {item.kind for item in session.items} == {"drawing"}
    sheet_ids = {item.sheet_id for item in session.items}
    assert len(sheet_ids) == 1 and None not in sheet_ids
    for item in session.items:
        path = Path(item.path)
        assert path.parent == tmp_path / "files" / "sheets" / sid
        assert path.name == f"{item.sheet_id}_{item.view}.png"
        assert cv2.imread(str(path)) is not None
        assert not item.hand_face
    notes = " ".join(session.sheet_notes)
    assert "first-angle" in notes and "switch" in notes, notes
    assert "first-angle" in studio.sheet_html(sid)


def test_a_tagged_image_is_never_split(studio, tmp_path, views):
    img, _ = draw_sheet(pick(views, FIVE))
    sid = studio.store.new()
    studio.add_images(sid, [save(tmp_path, img)], face="front")
    session = studio.store.get(sid)
    assert [(i.face, i.sheet_id) for i in session.items] == [("front", None)]
    assert session.sheets == {} and session.sheet_notes == [] and studio.sheet_html(sid) == ""


def test_a_photo_is_not_split(studio, tmp_path):
    photo = np.full((600, 800, 3), 200, np.uint8)
    cv2.rectangle(photo, (200, 150), (600, 450), (60, 60, 60), -1)
    shadow = np.full((600, 900, 3), 200, np.uint8)  # a part beside its shadow: two filled blobs in one row
    cv2.rectangle(shadow, (100, 200), (380, 400), (60, 60, 60), -1)
    cv2.rectangle(shadow, (500, 200), (780, 400), (110, 110, 110), -1)
    sid = studio.store.new()
    studio.add_images(sid, [save(tmp_path, photo, "photo.png"), save(tmp_path, shadow, "shadow.png")])
    session = studio.store.get(sid)
    assert [(i.name, i.face, i.kind, i.sheet_id) for i in session.items] == [
        ("photo.png", "auto", "auto", None), ("shadow.png", "auto", "auto", None)]
    assert session.sheets == {}


def test_a_tight_crop_of_one_drawing_is_not_split(studio, tmp_path):
    """The plate's outline touches the image edge, so the split takes it for the sheet frame and its two holes for
    two views that line up and pass the scale check. Round views alone never make a sheet."""
    img = np.full((300, 600, 3), 255, np.uint8)
    cv2.rectangle(img, (3, 3), (596, 296), (0, 0, 0), 2)
    for cx in (150, 450):
        cv2.circle(img, (cx, 150), 40, (0, 0, 0), 2)
        cv2.line(img, (cx - 52, 150), (cx + 52, 150), (0, 0, 0), 1)  # centre lines run past the circle
        cv2.line(img, (cx, 98), (cx, 202), (0, 0, 0), 1)
    sheet = split_sheet(img)
    assert is_sheet(sheet) and named_count(name_views(sheet, img)) >= 2  # what the round-view rule guards against
    sid = studio.store.new()
    studio.add_images(sid, [save(tmp_path, img, "plate.png")])
    session = studio.store.get(sid)
    assert [(i.face, i.sheet_id) for i in session.items] == [("auto", None)]


def test_small_marks_are_never_items(studio, tmp_path, views):
    """A dimension number the split keeps as a view is "skip": no item, and the items keep their view index."""
    img, truth = draw_sheet(pick(views, FIVE), labels=False)
    lx = truth["left"][0]
    cv2.putText(img, "25", (lx + 20, truth["top"][1] + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    sid = studio.store.new()
    studio.add_images(sid, [save(tmp_path, img)])
    session = studio.store.get(sid)
    (_, sheet, naming), = session.sheets.values()
    assert len(sheet.drawings[naming.drawing].views) == 6 and naming.faces.count("skip") == 1
    assert len(session.items) == 5
    assert all(naming.faces[i.view] == i.face != "skip" for i in session.items)
    assert {f: i.face for f, i in drawn_as(studio, sid, truth, FIVE).items()} == {f: f for f in FIVE}
    assert "skip" not in FACE_CHOICES


def test_the_projection_switch_renames_but_keeps_hand_picked_faces(studio, tmp_path, views):
    img, truth = draw_sheet(pick(views, FIVE), labels=False)
    sid = studio.store.new()
    studio.add_images(sid, [save(tmp_path, img)])
    items = drawn_as(studio, sid, truth, FIVE)
    studio.set_face(sid, items["top"].id, "top")  # the user confirms it by hand
    assert items["top"].hand_face
    studio.set_projection(sid, "third")
    session = studio.store.get(sid)
    assert session.projection == "third"
    # third-angle: the view below the front is the bottom, the side views swap; the hand-picked face stays
    assert {f: i.face for f, i in items.items()} == {"front": "front", "top": "top", "right": "left",
                                                     "left": "right", "back": "back"}
    (_, _, naming), = session.sheets.values()
    assert (naming.projection, naming.projection_source) == ("third", "setting")
    assert "third-angle" in " ".join(session.sheet_notes)
    studio.set_projection(sid, "first")
    assert {f: i.face for f, i in items.items()} == {f: f for f in FIVE}


def test_a_hand_picked_face_is_never_given_to_a_second_view(studio, tmp_path, views):
    img, truth = draw_sheet(pick(views, FIVE), labels=False)
    sid = studio.store.new()
    studio.add_images(sid, [save(tmp_path, img)])
    items = drawn_as(studio, sid, truth, FIVE)
    studio.set_face(sid, items["left"].id, "left")
    studio.set_projection(sid, "third")
    assert items["left"].face == "left" and items["right"].face == "auto"
    assert any("left" in n and "by hand" in n for n in studio.store.get(sid).sheet_notes)


def test_the_symbol_overrides_the_switch_and_says_so(studio, tmp_path, views):
    img, truth = draw_sheet(pick(views, FIVE), labels=False, symbol="first")
    sid = studio.store.new()
    studio.add_images(sid, [save(tmp_path, img)])
    assert "symbol" in " ".join(studio.store.get(sid).sheet_notes)
    studio.set_projection(sid, "third")
    session = studio.store.get(sid)
    assert {f: i.face for f, i in drawn_as(studio, sid, truth, FIVE).items()} == {f: f for f in FIVE}
    (_, _, naming), = session.sheets.values()
    assert (naming.projection, naming.projection_source) == ("first", "symbol")
    notes = " ".join(session.sheet_notes)
    assert "symbol" in notes and "overrides the switch" in notes, notes


def test_removing_every_view_of_a_sheet_drops_its_notes(studio, tmp_path, views):
    img, _ = draw_sheet(pick(views, ("front", "top", "left")))
    sid = studio.store.new()
    studio.add_images(sid, [save(tmp_path, img)])
    for item in list(studio.store.get(sid).items):
        studio.remove(sid, item.id)
    session = studio.store.get(sid)
    assert session.sheets == {} and session.sheet_notes == []


def test_sheet_crops_are_swept_with_the_other_files(studio, tmp_path, views):
    img, _ = draw_sheet(pick(views, ("front", "top", "left")))
    path = save(tmp_path, img)
    old, new = studio.store.new(), studio.store.new()
    studio.add_images(old, [path])
    folder = tmp_path / "files" / "sheets" / old
    stale = time.time() - 2 * artifacts.TTL_S
    os.utime(folder, (stale, stale))
    studio.store.drop(old)  # the session has ended
    studio.add_images(new, [path])
    assert not folder.exists()
    assert all(Path(i.path).exists() for i in studio.store.get(new).items)
    artifacts.sweep(tmp_path / "files")  # the Studio's own sweep never takes the sheets folder while it is in use
    assert all(Path(i.path).exists() for i in studio.store.get(new).items)


def test_a_live_sessions_crops_outlive_an_hour(studio, tmp_path, views):
    """A user can spend over an hour in Review: the crops stay while the session lives."""
    img, _ = draw_sheet(pick(views, ("front", "top", "left")))
    path = save(tmp_path, img)
    live, other = studio.store.new(), studio.store.new()
    studio.add_images(live, [path])
    folder = tmp_path / "files" / "sheets" / live
    stale = time.time() - 2 * artifacts.TTL_S
    os.utime(folder, (stale, stale))
    os.utime(folder.parent, (stale, stale))
    studio.add_images(other, [path])  # sweeps the sheets folder
    artifacts.sweep(tmp_path / "files")
    assert all(Path(i.path).exists() for i in studio.store.get(live).items)
    review = studio.analyze(live, "none", AiSettings())
    assert review.stage == "review", review.message_html


def test_an_inferred_face_has_a_badge():
    assert FACE_BADGES["inferred"] == ("inferred from the other side view", "check")


def readme_sizes() -> dict[str, str]:
    text = (SHEET_DIR / "README.md").read_text(encoding="utf-8")
    sizes = {}
    for axis, label in AXIS_LABEL.items():
        m = re.search(re.escape(label) + r"[^0-9]*([0-9]+(?:\.[0-9]+)?) mm", text)
        assert m, label
        sizes[axis] = m.group(1)
    return sizes


def test_the_sheet_example_builds(studio):
    sid = studio.store.new()
    studio.load_sheet_example(sid)
    session = studio.store.get(sid)
    assert len(session.items) >= 3 and {i.kind for i in session.items} == {"drawing"}
    assert "auto" not in {i.face for i in session.items}
    assert "symbol" in " ".join(session.sheet_notes)
    review = studio.analyze(sid, "none", AiSettings())
    assert not review.ok and all(review.sizes[a]["required"] for a in "xyz")  # the sheet gives no millimetres
    review, model = studio.build(sid, readme_sizes(), review.rows, [], GeometrySettings())
    assert model.ok, model.message_html
    scores = {label.split(" · ")[0]: float(m.group(1)) for _, label in model.views
              if (m := re.search(r"match ([0-9.]+)", label))}
    assert set(scores) == {i.face for i in session.items}, scores
    assert min(scores.values()) >= 0.9, scores


def test_a_hand_sketch_photo_is_split_into_named_faces_and_renamed_on_its_page(studio, tmp_path):
    """A phone photo of a pen sketch: one item per face; the projection switch renames them on the sketch's page."""
    from tests.hand_views import hand_photo
    from tests.test_mv_relief import _block
    photo, _ = hand_photo(_block(), faces=("front", "top", "right"), layout="third")
    sid = studio.store.new()
    studio.set_projection(sid, "third")
    studio.add_images(sid, [save(tmp_path, photo, "sketch.jpg")])
    session = studio.store.get(sid)
    assert sorted(i.face for i in session.items) == ["front", "right", "top"]
    assert all(i.mm_per_px is None for i in session.items)  # a sketch is never to scale
    studio.set_projection(sid, "first")
    assert sorted(i.face for i in session.items) == ["bottom", "front", "left"]


def test_a_sketchs_faces_go_in_as_lines_with_their_numbers_read_once(tmp_path, monkeypatch):
    """The Studio feeds a sketch's faces as the web does: drawn again in lines, and never read again for numbers
    (a stub read as "1" would be a silently wrong size)."""
    artifacts.clear_cache()
    monkeypatch.setattr("s2c.multiview.slice.find_slicer", lambda: None)
    studio = Studio(MvPipeline(reader=lambda crop: ("1", 0.99)), root=tmp_path / "files")
    sid = studio.store.new()
    studio.set_projection(sid, "third")
    photo = Path(__file__).resolve().parent / "golden_sketch" / "real_bracket_1" / "image.jpg"
    studio.add_images(sid, [str(photo)])
    studio.analyze(sid, "none", AiSettings())
    observed = studio.store.get(sid).observed
    assert len(observed.observations) == 3
    assert all(not o.values for o in observed.observations)
    assert all(o.outline.line_art and not o.outline.inner for o in observed.observations)
