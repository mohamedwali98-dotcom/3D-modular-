"""Three-view engineering drawing of the final solid: hidden-line projection (OCCT HLR) of the front, top and right
views in a third-angle layout, overall dimensions, hole diameters and a title block. DXF in millimetres; SVG and
PDF are rendered from the DXF. Spec 2026-09-23-studio section 6."""
from __future__ import annotations

import os
import tempfile
from datetime import UTC, datetime
from pathlib import Path

import cadquery as cq
import ezdxf
from cadquery.occ_impl.exporters.dxf import DxfDocument
from ezdxf.addons.drawing import Frontend, RenderContext, layout, svg
from ezdxf.addons.drawing.config import BackgroundPolicy, ColorPolicy, Configuration
from matplotlib.figure import Figure
from OCP.BRepLib import BRepLib
from OCP.gp import gp_Ax2, gp_Dir, gp_Pnt
from OCP.HLRAlgo import HLRAlgo_Projector
from OCP.HLRBRep import HLRBRep_Algo, HLRBRep_HLRToShape

from s2c.multiview.settings import EDGE_LABELS
from s2c.multiview.spec import CANONICAL_OF, FaceHole, Fillet, MultiViewSpec, feature_path, to_canonical

DRAWING_FORMATS = ("dxf", "svg", "pdf")
DRAWING_FILES = {"dxf": "drawing.dxf", "svg": "drawing.svg", "pdf": "drawing.pdf"}
# face -> (view direction, drawing x direction). Face frames of the multi-view spec section 3: after shifting by the
# envelope, a point (a, b) of that face lands at (a, b) in its view.
VIEWS = {"front": ((0, 0, 1), (1, 0, 0)), "top": ((0, 1, 0), (1, 0, 0)), "right": ((1, 0, 0), (0, 0, -1))}
DIM_STYLE = {"dimtxt": 3.5, "dimlfac": 1, "dimasz": 2.5, "dimexo": 1.5}  # the EZDXF style scales by 100 otherwise
RENDER = Configuration(background_policy=BackgroundPolicy.WHITE, color_policy=ColorPolicy.BLACK)



def feature_note(f, depth_prov: str | None = None) -> str:
    """A feature the views do not dimension, in words: "pocket 15 x 20, 20 deep, on the front face". A depth no
    view of the source drawing showed says so."""
    if f.type == "pocket":
        depth = "through" if f.depth_mm is None else f"{f.depth_mm:g} deep"
        if depth_prov == "default":
            depth += " (depth not drawn, assumed)"
        return f"pocket {f.width_mm:g} x {f.height_mm:g}, {depth}, on the {f.face} face"
    if f.type == "boss":
        return f"pin Ø{f.diameter_mm:g}, {f.height_mm:g} high, on the {f.face} face"
    if f.type == "slot":
        return f"slot {f.length_mm:g} x {f.width_mm:g} on the {f.face} face"
    return f"{f.type} on the {f.face} face"

def _collect(*compounds) -> list[cq.Shape]:
    # HLR returns each part (VCompound, OutLineVCompound, ...) as a TopoDS_Shape whose direct children are free
    # Edges. DxfDocument.add_shape wraps a WorkplaneLike's stack items as the immediate children of a fresh
    # compound, so pushing the individual Edge objects on the Workplane's stack (rather than one Compound of
    # them) is what keeps them at that top level and lets add_shape's plane-aware spline fallback see a plane.
    edges = []
    for c in compounds:
        if c is None or c.IsNull():
            continue
        BRepLib.BuildCurves3d_s(c, 1e-6)  # HLR edges carry only 2D curves
        edges.extend(cq.Shape.cast(c).Edges())
    return edges


def _project(shape: cq.Shape, direction, xdir) -> tuple[list[cq.Shape], list[cq.Shape]]:
    algo = HLRBRep_Algo()
    algo.Add(shape.wrapped)
    algo.Projector(HLRAlgo_Projector(gp_Ax2(gp_Pnt(0, 0, 0), gp_Dir(*direction), gp_Dir(*xdir))))
    algo.Update()
    algo.Hide()
    hlr = HLRBRep_HLRToShape(algo)
    return _collect(hlr.VCompound(), hlr.OutLineVCompound()), _collect(hlr.HCompound(), hlr.OutLineHCompound())


def _linear(msp, p1, p2, base, angle=0.0) -> None:
    msp.add_linear_dim(base=base, p1=p1, p2=p2, angle=angle, dimstyle="EZDXF", override=DIM_STYLE,
                       dxfattribs={"layer": "DIMENSIONS"}).render()


def drawing_document(solid, spec: MultiViewSpec) -> ezdxf.document.Drawing:
    shape = solid.val() if isinstance(solid, cq.Workplane) else solid
    env = spec.envelope
    w, h, d = env.x_mm, env.y_mm, env.z_mm
    gap = max(15.0, 0.3 * max(w, h, d))
    shift = {"front": (0.0, 0.0), "top": (0.0, d), "right": (d, 0.0)}      # envelope corner to (0, 0)
    place = {"front": (0.0, 0.0), "top": (0.0, h + gap), "right": (w + gap, 0.0)}  # third-angle layout
    dxf = DxfDocument(setup=True)  # linetypes and the EZDXF dimension style
    dxf.add_layer("VISIBLE", color=7)
    dxf.add_layer("HIDDEN", color=8, linetype="DASHED")
    dxf.add_layer("DIMENSIONS", color=1)
    dxf.add_layer("TEXT", color=7)
    for face, (direction, xdir) in VIEWS.items():
        dx, dy = shift[face][0] + place[face][0], shift[face][1] + place[face][1]
        visible, hidden = _project(shape, direction, xdir)
        for part, layer in ((visible, "VISIBLE"), (hidden, "HIDDEN")):
            if part:
                moved = [e.translate(cq.Vector(dx, dy, 0)) for e in part]
                dxf.add_shape(cq.Workplane("XY").add(moved), layer)
    doc = dxf.document
    doc.header["$INSUNITS"] = 4
    msp = doc.modelspace()
    _linear(msp, (0, 0), (w, 0), (0, -10))                               # width under the front view
    _linear(msp, (0, 0), (0, h), (-10, 0), angle=90)                     # height left of the front view
    _linear(msp, (w + gap, 0), (w + gap + d, 0), (w + gap, -10))         # depth under the right view
    notes = []
    for k, f in enumerate(spec.features):
        if isinstance(f, FaceHole) and f.face in place:
            cx, cy = place[f.face][0] + f.a_mm, place[f.face][1] + f.b_mm
            msp.add_diameter_dim(center=(cx, cy), radius=f.diameter_mm / 2, angle=45, dimstyle="EZDXF",
                                 override=DIM_STYLE, dxfattribs={"layer": "DIMENSIONS"}).render()
        elif isinstance(f, FaceHole):  # far side: dimension it, mirrored, in the opposite face's view
            canonical = CANONICAL_OF[f.face]
            a2, b2 = to_canonical(f.face, [(f.a_mm, f.b_mm)], spec.envelope)[0]
            cx, cy = place[canonical][0] + a2, place[canonical][1] + b2
            msp.add_diameter_dim(center=(cx, cy), radius=f.diameter_mm / 2, angle=45, dimstyle="EZDXF",
                                 override=DIM_STYLE, text=f"<> ({f.face})",  # ezdxf prepends its own diameter
                                 dxfattribs={"layer": "DIMENSIONS"}).render()   # sign to "<>"; don't add a second
        else:
            notes.append(f"Feature {k + 1}: {feature_note(f, spec.provenance.get(feature_path(k, 'depth_mm')))}")
    for finish in spec.finishes:
        if isinstance(finish, Fillet):
            notes.append(f"Fillet R{finish.radius_mm:g} on {EDGE_LABELS[finish.edges]}")
        else:  # a chamfer's size is a leg length, not a radius
            notes.append(f"Chamfer C{finish.radius_mm:g} on {EDGE_LABELS[finish.edges]}")
    lines = ["Sketch-to-CAD", f"Envelope {w:g} x {h:g} x {d:g} mm", "Third-angle projection, units mm",
             f"Date {datetime.now(tz=UTC).date().isoformat()}", *notes]
    x0, y0 = w + gap + d + gap, h + gap + d  # past the right view (whose own right edge is w + gap + d), any envelope
    for i, text in enumerate(lines):
        msp.add_text(text, height=3.5 if i else 5.0, dxfattribs={"layer": "TEXT"}).set_placement((x0, y0 - 7 * i))
    return doc


def _svg(doc) -> str:
    backend = svg.SVGBackend()
    Frontend(RenderContext(doc), backend, config=RENDER).draw_layout(doc.modelspace())
    return backend.get_string(layout.Page(297, 210, layout.Units.mm, margins=layout.Margins.all(10)))


def _pdf(doc, path: Path) -> None:
    from ezdxf.addons.drawing.matplotlib import MatplotlibBackend
    fig = Figure(figsize=(11.69, 8.27))  # A4 landscape; no pyplot, so no GUI backend is involved
    ax = fig.add_axes((0.03, 0.03, 0.94, 0.94))
    ax.set_axis_off()
    Frontend(RenderContext(doc), MatplotlibBackend(ax), config=RENDER).draw_layout(doc.modelspace(), finalize=True)
    fig.savefig(str(path), format="pdf")


def _atomic_write(path: Path, write) -> None:
    """write(tmp) fills a temp file next to `path`; only a fully-written file ever appears at `path`, so a
    concurrent reader checking existence (artifacts.export_part) never sees a half-written drawing."""
    fd, name = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    os.close(fd)
    tmp = Path(name)
    try:
        write(tmp)
        os.replace(tmp, path)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise


def write_drawings(solid, spec: MultiViewSpec, out_dir: Path, formats) -> dict[str, Path]:
    wanted = [f for f in DRAWING_FORMATS if f in formats]
    if not wanted:
        return {}
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    doc = drawing_document(solid, spec)
    files = {f: out_dir / DRAWING_FILES[f] for f in wanted}
    writers = {"dxf": lambda tmp: doc.saveas(tmp), "svg": lambda tmp: tmp.write_text(_svg(doc), encoding="utf-8"),
               "pdf": lambda tmp: _pdf(doc, tmp)}
    for f in wanted:
        _atomic_write(files[f], writers[f])
    return files
