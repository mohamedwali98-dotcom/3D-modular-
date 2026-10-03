import math
import re

import cv2
import numpy as np
import pytest

from s2c.multiview import spec as S
from s2c.multiview.build import build, volume
from s2c.multiview.fuse import Observation, classify_drawn_circles, features_from
from s2c.multiview.ocr import Linked, Reading
from s2c.multiview.outline import extract
from s2c.multiview.pipeline import ImageInput, MvPipeline, input_mask
from s2c.multiview.turned import complete_turned
from tests.mv_helpers import circle, rect
from tests.test_mv_line_art import CHAIN, HIDDEN, centre_cross, page, pattern_line, png

INK = (0, 0, 0)
GREY = (128, 128, 128)
PX = 15                  # px per mm in every drawn view
R, r, L = 20, 10, 30     # the cone along x: large radius, small radius, length (mm)
CX, CY = 800, 600        # every view is drawn around the page centre


def drawn(face, img):
    return Observation(face=face, kind="drawing", outline=extract(img, drawing=True))


def envelope(x, y, z):
    return S.Envelope(x_mm=x, y_mm=y, z_mm=z)


def diameter(warning):
    return float(re.match(r"Circle Ø([\d.]+) on ", warning).group(1))


def end_view():
    """Two concentric circles with a centre cross: a cone seen from its small end, or a washer."""
    img = page()
    cv2.circle(img, (CX, CY), R * PX, INK, 2)
    cv2.circle(img, (CX, CY), r * PX, INK, 2)
    centre_cross(img, CX, CY, R * PX, beyond=40)
    return img


def cone_front():
    """The cone lying along x, small end on the left (it faces the left view), with its centre line."""
    img = page()
    x0, x1 = CX - L * PX // 2, CX + L * PX // 2
    pts = np.array([(x0, CY - r * PX), (x1, CY - R * PX), (x1, CY + R * PX), (x0, CY + r * PX)])
    cv2.polylines(img, [pts], True, INK, 2)
    pattern_line(img, (x0 - 40, CY), (x1 + 40, CY), CHAIN)
    return img


def washer_front(thick=6):
    img = page()
    x0, x1 = CX - thick * PX // 2, CX + thick * PX // 2
    cv2.rectangle(img, (x0, CY - R * PX), (x1, CY + R * PX), INK, 2)
    pattern_line(img, (x0 - 40, CY), (x1 + 40, CY), CHAIN)
    return img


def test_the_small_end_of_a_cone_is_an_edge():
    front, left = drawn("front", cone_front()), drawn("left", end_view())
    assert front.line_art and left.line_art
    assert left.outline.circular and len(left.outline.circles) == 1  # the outer circle is the outline
    env = envelope(L, 2 * R, 2 * R)
    edges, warnings = classify_drawn_circles([front, left], env)
    assert edges == {1: {0}}
    assert len(warnings) == 1 and warnings[0].endswith(" on left: read as an edge (step in front)")
    assert diameter(warnings[0]) == pytest.approx(2 * r, abs=0.5)
    assert features_from([front, left], env, edges)[0] == []


def test_a_washer_bore_is_a_hole():
    front, left = drawn("front", washer_front()), drawn("left", end_view())
    env = envelope(6, 2 * R, 2 * R)
    edges, warnings = classify_drawn_circles([front, left], env)
    assert edges == {}
    assert len(warnings) == 1 and warnings[0].endswith(" on left: read as a hole, no other view explains it")
    feats, _ = features_from([front, left], env, edges)
    assert len(feats) == 1 and feats[0]["face"] == "left"
    assert feats[0]["diameter_mm"] == pytest.approx(2 * r, rel=0.05)


def plate_front():
    """A 60 x 40 plate with a Ø8 circle at (20, 24) mm."""
    img = page()
    cv2.rectangle(img, (350, 300), (1250, 900), INK, 2)
    cv2.circle(img, (350 + 20 * PX, 900 - 24 * PX), 4 * PX, INK, 2)
    centre_cross(img, 350 + 20 * PX, 900 - 24 * PX, 4 * PX)
    return img


def plate_top():
    """The plate from above, 60 x 10, with the hole's sides dashed at x = 20 -/+ 4 mm."""
    img = page()
    cv2.rectangle(img, (350, 525), (1250, 675), INK, 2)
    for x in (16, 24):
        pattern_line(img, (350 + x * PX, 525), (350 + x * PX, 675), HIDDEN)
    return img


def plate_right():
    """The plate from the right, 10 x 40, with the hole's sides dashed at y = 24 -/+ 4 mm: lines across the view,
    placed off-centre so a flipped height would miss them."""
    img = page()
    cv2.rectangle(img, (725, 300), (875, 900), INK, 2)
    for y in (20, 28):
        pattern_line(img, (725, 900 - y * PX), (875, 900 - y * PX), HIDDEN)
    return img


@pytest.mark.parametrize("face, view", [("top", plate_top), ("right", plate_right)])
def test_hidden_lines_make_a_hole(face, view):
    front, side = drawn("front", plate_front()), drawn(face, view())
    assert len(front.outline.circles) == 1 and len(side.hidden) == 2
    env = envelope(60, 40, 10)
    assert classify_drawn_circles([front, side], env) == ({}, [])
    feats, _ = features_from([front, side], env, {})
    assert len(feats) == 1 and feats[0]["diameter_mm"] == pytest.approx(8, rel=0.05)


def test_a_lone_plate_drawing_warns_once_for_all_its_holes():
    """Four holes no other view explains: one warning for the face, with the diameters smallest first."""
    img = page()
    cv2.rectangle(img, (350, 300), (1250, 900), INK, 2)
    for (x, y), d in zip([(10, 10), (50, 10), (10, 30), (50, 30)], (10, 6, 6, 10)):
        cv2.circle(img, (350 + x * PX, 900 - y * PX), d * PX // 2, INK, 2)
    front = drawn("front", img)
    assert len(front.outline.circles) == 4
    edges, warnings = classify_drawn_circles([front], envelope(60, 40, 10))
    assert edges == {} and len(warnings) == 1
    m = re.fullmatch(r"4 circles on front \((.+)\): read as holes, no other view explains them", warnings[0])
    assert m
    assert [float(d.removeprefix("Ø")) for d in m.group(1).split(", ")] == pytest.approx([6, 6, 10, 10], abs=0.5)


def bracket_front():
    """A 60 wide base, 8 high, and an upright lug 20 wide whose top is rounded about (30, 40) mm, with a Ø8 hole
    there. 50 mm high overall."""
    img = page()
    base, x0 = 975, 350

    def px(a, b):
        return (round(x0 + a * PX), round(base - b * PX))

    arc = [px(30 + 10 * math.cos(t), 40 + 10 * math.sin(t)) for t in np.linspace(0, math.pi, 60)]
    pts = [px(0, 0), px(60, 0), px(60, 8), px(40, 8), *arc, px(20, 8), px(0, 8)]
    cv2.polylines(img, [np.array(pts)], True, INK, 2)
    cv2.circle(img, px(30, 40), 4 * PX, INK, 2)
    centre_cross(img, *px(30, 40), 4 * PX)
    return img


def test_a_lug_hole_without_hidden_lines_is_a_hole():
    plain = page()
    cv2.rectangle(plain, (350, 375), (1250, 825), INK, 2)  # 60 x 30, nothing inside
    front, top = drawn("front", bracket_front()), drawn("top", plain)
    assert len(front.outline.circles) == 1 and top.hidden == []
    env = envelope(60, 50, 30)
    edges, warnings = classify_drawn_circles([front, top], env)
    assert edges == {}
    assert len(warnings) == 1 and warnings[0].endswith(" on front: read as a hole, no other view explains it")
    assert diameter(warnings[0]) == pytest.approx(8, abs=0.5)
    assert len(features_from([front, top], env, edges)[0]) == 1


def test_a_two_view_cone_builds_as_a_cone(tmp_path):
    pipe = MvPipeline()
    observed = pipe.observe([ImageInput(png(cone_front()), "front", "drawing"),
                             ImageInput(png(end_view()), "left", "drawing")])
    spec = pipe.fuse(observed, {"envelope.x_mm": L, "envelope.y_mm": 2 * R, "envelope.z_mm": 2 * R})
    assert isinstance(spec, S.MultiViewSpec)
    assert spec.features == []
    assert spec.views.top.source == "inferred" and spec.provenance["views.top.outer"] == "inferred"
    assert observed.filled_by["top"] == "inferred"  # the web app's FaceCard must know this value (web review C1)
    frustum = math.pi * L * (R ** 2 + R * r + r ** 2) / 3
    assert volume(build(spec)) == pytest.approx(frustum, rel=0.03)
    result = pipe.build(spec, tmp_path, observed.masks)
    assert result.step.exists() and result.iou["left"] > 0.85  # the small end's circle is no hole in the input


def filled_cone_front():
    img = page()
    x0, x1 = CX - L * PX // 2, CX + L * PX // 2
    pts = np.array([(x0, CY - r * PX), (x1, CY - R * PX), (x1, CY + R * PX), (x0, CY + r * PX)])
    cv2.fillPoly(img, [pts], GREY)
    return img


def filled_end_view():
    img = page()
    cv2.circle(img, (CX, CY), R * PX, GREY, -1)
    cv2.circle(img, (CX, CY), r * PX, (255, 255, 255), -1)
    return img


def test_filled_renders_keep_every_circle():
    """The same outlines as the cone, rendered filled: the see-through circle is a hole, as before line art."""
    front, left = drawn("front", filled_cone_front()), drawn("left", filled_end_view())
    assert not front.line_art and not left.line_art and len(left.outline.circles) == 1
    env = envelope(L, 2 * R, 2 * R)
    assert classify_drawn_circles([front, left], env) == ({}, [])
    assert len(features_from([front, left], env, {})[0]) == 1


# ---- fix round 1 ------------------------------------------------------------------


def cross_drilled_bar(px, d, section):
    """A bar along x, 40 long and 20 high, drilled across (along z) through its middle with a Ø d hole. Seen from
    the left it is round, or a flat bar 50 deep with fully rounded edges."""
    front = page()
    cv2.rectangle(front, (CX - 20 * px, CY - 10 * px), (CX + 20 * px, CY + 10 * px), INK, 2)
    cv2.circle(front, (CX, CY), round(d * px / 2), INK, 2)
    left = page()
    if section == "round":
        cv2.circle(left, (CX, CY), 10 * px, INK, 2)
    else:
        for x, start in ((CX - 15 * px, 90), (CX + 15 * px, -90)):
            cv2.ellipse(left, (x, CY), (10 * px, 10 * px), 0, start, start + 180, INK, 2)
        for y in (CY - 10 * px, CY + 10 * px):
            cv2.line(left, (CX - 15 * px, y), (CX + 15 * px, y), INK, 2)
    return front, left


@pytest.mark.parametrize("px", [8, 10, 12, 16, 20])
@pytest.mark.parametrize("d", [6, 8, 10, 12])
@pytest.mark.parametrize("section", ["round", "rounded flat"])
def test_a_cross_hole_is_not_explained_by_a_curved_silhouette(px, d, section):
    """Every chord of a round silhouette is some width, centred on the axis: only a corner is a step or an end."""
    front, left = (drawn(f, img) for f, img in zip(("front", "left"), cross_drilled_bar(px, d, section)))
    assert len(front.outline.circles) == 1
    edges, warnings = classify_drawn_circles([front, left], envelope(40, 20, 20 if section == "round" else 50))
    assert edges == {}
    assert warnings[0].endswith(" on front: read as a hole, no other view explains it")


def small_cone(line):
    """The cone at 4 px/mm, a 160 px view on a full page, drawn with thick lines."""
    front, end = page(), page()
    x0, x1 = CX - L * 2, CX + L * 2
    pts = np.array([(x0, CY - r * 4), (x1, CY - R * 4), (x1, CY + R * 4), (x0, CY + r * 4)])
    cv2.polylines(front, [pts], True, INK, line)
    cv2.circle(end, (CX, CY), R * 4, INK, line)
    cv2.circle(end, (CX, CY), r * 4, INK, line)
    return front, end


@pytest.mark.parametrize("line", [2, 3])
def test_a_thick_line_does_not_widen_the_silhouette(line):
    """A silhouette is measured to the outside of its line, a circle to the middle of its line."""
    front, end = small_cone(line)
    observed = MvPipeline().observe([ImageInput(png(front), "front", "drawing"), ImageInput(png(end), "left", "drawing")])
    assert all(o.line_art for o in observed.observations)
    edges, _ = classify_drawn_circles(observed.observations, envelope(L, 2 * R, 2 * R))
    assert edges == {1: {0}}


def test_the_input_mask_follows_the_verdict_on_every_fuse():
    """The verdict does not move with the sizes, which scale both views of the shared axis alike, but it does
    with the views: without the side view the small end's circle is a hole again, and the mask shows it."""
    pipe = MvPipeline()
    observed = pipe.observe([ImageInput(png(cone_front()), "front", "drawing"),
                             ImageInput(png(end_view()), "left", "drawing")])
    sizes = {"envelope.x_mm": L, "envelope.y_mm": 2 * R, "envelope.z_mm": 2 * R}
    left = observed.observations[1]
    assert pipe.fuse(observed, sizes).features == []
    assert (observed.masks["left"] == input_mask(left.outline, {0})).all()
    observed.observations = [left]
    assert len(pipe.fuse(observed, sizes).features) == 1
    assert (observed.masks["left"] == input_mask(left.outline)).all()


def test_the_warning_gives_the_diameter_the_hole_gets():
    front = drawn("front", bracket_front())
    front.values = [Linked(Reading(8.5, "diameter", (0, 0, 1, 1), 0.9, "Ø8.5"), None, 0)]
    env = envelope(60, 50, 30)
    _, warnings = classify_drawn_circles([front], env)
    feats, prov = features_from([front], env)
    assert feats[0]["diameter_mm"] == 8.5 and prov["features[0].diameter_mm"] == "user_written"
    assert warnings == ["Circle Ø8.5 on front: read as a hole, no other view explains it"]


def test_complete_turned_leaves_two_given_side_views_alone():
    env = envelope(40, 10, 40)
    given = {"top": S.Outline(outer=circle(20, 20, 20), source="observed", confidence=0.9),
             "front": S.Outline(outer=rect(40, 10), source="observed", confidence=0.9),
             "right": S.Outline(outer=rect(40, 10), source="observed", confidence=0.9)}
    assert complete_turned(given, env) == (given, [])


def test_complete_turned_needs_a_round_view():
    env = envelope(40, 10, 40)
    given = {"top": S.Outline(outer=rect(40, 40), source="observed", confidence=0.9),
             "front": S.Outline(outer=rect(40, 10), source="observed", confidence=0.9)}
    assert complete_turned(given, env) == (given, [])


def test_a_drawing_with_a_filled_view_gets_no_copied_side_view():
    filled = page()
    cv2.circle(filled, (CX, CY), R * PX, GREY, -1)
    pipe = MvPipeline()
    observed = pipe.observe([ImageInput(png(cone_front()), "front", "drawing"),
                             ImageInput(png(filled), "left", "drawing")])
    assert [o.line_art for o in observed.observations] == [True, False]
    spec = pipe.fuse(observed, {"envelope.x_mm": L, "envelope.y_mm": 2 * R, "envelope.z_mm": 2 * R})
    assert spec.views.top.source == "assumed"
    assert not [w for w in spec.warnings if "copied" in w]
