"""Builders shared by several test modules: parts made from boxes and pins, a part's drawn views read into a spec,
and a reading service that reads known words. Test modules import these instead of each other's private helpers."""
import cadquery as cq
import cv2

from s2c.multiview import spec as S
from s2c.multiview.pipeline import ImageInput, MvPipeline
from s2c.reading import ReadingService
from tests.line_views import WordReader, draw_view

RELIEF_FACES = ("front", "top", "right")


def page_png(ink, margin=40) -> bytes:
    """White-ink-on-black lines as a PNG page, black on white, with a margin."""
    ink = cv2.copyMakeBorder(ink, margin, margin, margin, margin, cv2.BORDER_CONSTANT, value=0)
    return cv2.imencode(".png", 255 - ink)[1].tobytes()


def solid_block(x=80, y=60, z=70) -> cq.Workplane:
    return cq.Workplane("XY").box(x, y, z, centered=False)


def cut_box(part, x0, y0, z0, x1, y1, z1) -> cq.Workplane:
    return part.cut(cq.Workplane("XY").box(x1 - x0, y1 - y0, z1 - z0, centered=False).translate((x0, y0, z0)))


def add_pins(part, xs=(20, 60), z=15, d=8, h=6, top=60):
    for x in xs:
        part = part.union(cq.Workplane("XZ", origin=(0, top, 0)).center(x, z).circle(d / 2).extrude(-h))
    return part


def relief_spec(part, faces=RELIEF_FACES, kind="drawing", hidden=True) -> S.MultiViewSpec:
    """The part's exact drawn views, observed and fused at its true size."""
    pipe = MvPipeline()
    observed = pipe.observe([ImageInput(page_png(draw_view(part, f, 4.0, hidden=hidden)), f, kind) for f in faces])
    bb = part.val().BoundingBox()
    spec = pipe.fuse(observed, {"envelope.x_mm": round(bb.xlen, 3), "envelope.y_mm": round(bb.ylen, 3),
                                "envelope.z_mm": round(bb.zlen, 3)})
    assert isinstance(spec, S.MultiViewSpec), spec
    return spec


def pockets_of(spec):
    return [f for f in spec.features if f.type == "pocket"]


def word_service(words):
    """A reading service that reads the words drawn on a test sheet: [(box, text)]."""
    return ReadingService([WordReader(words)], cache=None)
