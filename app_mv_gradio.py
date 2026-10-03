"""Lab app for the multi-view path: upload face images, review, edit the numbers, rebuild, download.
Spec 2026-09-23 section 11. Run: uv run python app_mv_gradio.py, then open http://localhost:7860"""
from __future__ import annotations

import math
import re
import shutil
import time
import uuid
from pathlib import Path

import gradio as gr
import numpy as np
from dotenv import load_dotenv

from s2c.config import data_path
from s2c.multiview.pipeline import ImageInput, MvPipeline, Observed, default_pipeline
from s2c.multiview.raster import outline_mask
from s2c.multiview.spec import CANONICAL_FACES, CANONICAL_OF, FACES, MultiViewSpec, MvAbstain, face_size

OUT_ROOT = data_path("tmp", "mv_gradio")
TTL_S = 3600  # built files and Gradio's upload cache live one hour, as docs/disclosure.md says
FACE_CHOICES = ("auto", *FACES)
KIND_CHOICES = ("auto", "sketch", "photo", "drawing")
REFERENCES = ["none", "1 TND", "1 EUR", "2 EUR", "card", "a4"]
AMBER = frozenset({"scaled", "inferred", "estimated", "default"})
MISSING = "missing: confirm or type a value"
BADGES = {"observed": "observed", "mirrored": "mirrored from the opposite face", "qwen-image": "drawn by Qwen-Image",
          "triposr": "predicted by TripoSR", "assumed": "assumed rectangle"}
NOTICE = ("Photos of real parts: shoot top-down, with the part lying flat on a plain surface. "
          "Images are sent to DashScope and Hugging Face Spaces to read the handwriting and predict missing faces.")
OUTPUTS = ("faces", "reads", "values", "warnings", "message", "model", "views", "files", "stats", "state", "rejected")
_FEATURE = re.compile(r"(\w+)\[(\d+)\]\.(\w+)")
# the raw face tag that mirrors into each canonical face, e.g. a "back" photo mirrors into "front"
OPPOSITE_OF = {canon: face for face, canon in CANONICAL_OF.items() if face != canon}


def sweep_outputs(root: Path = OUT_ROOT, ttl_s: float = TTL_S) -> None:
    """Delete build folders older than the time-to-live."""
    if not root.exists():
        return
    now = time.time()
    for d in root.iterdir():
        if d.is_dir() and now - d.stat().st_mtime > ttl_s:
            shutil.rmtree(d, ignore_errors=True)


def new_state() -> dict:
    return {"observed": None, "shown": {}, "edits": {}}


def tag_rows(paths) -> list[list[str]]:
    return [[Path(p).name, "auto", "auto"] for p in paths or []]


def _pack(state: dict, **parts) -> tuple:
    out = {"faces": [], "reads": [], "values": [], "warnings": "", "message": "", "model": None, "views": [],
           "files": [], "stats": "", "state": state, "rejected": gr.update()}
    out.update(parts)
    return tuple(out[k] for k in OUTPUTS)


def _abstain(a: MvAbstain) -> str:
    return f"**Stopped at {a.stage}: {a.reason}.** {a.remedy}"


def _bullets(lines) -> str:
    return "\n".join(f"- {line}" for line in lines)


def _value(spec: MultiViewSpec, path: str) -> float:
    data = spec.model_dump()
    m = _FEATURE.fullmatch(path)
    if m:
        return data[m.group(1)][int(m.group(2))][m.group(3)]
    head, name = path.split(".", 1)
    return data[head][name]


def value_rows(spec: MultiViewSpec | None, abstain: MvAbstain | None, edits: dict) -> list[list]:
    """Envelope first, then every numeric feature value, with its source. Outlines are not edited here.
    A missing axis shows no value in its box: a suggestion is only a hint appended to the source column text
    (e.g. "missing: ... - suggested 60 mm"), never a pre-filled box the user could leave untouched and have it
    count as theirs (CLAUDE.md rule 2)."""
    if spec is not None:
        return [[p, _value(spec, p), f"{s} (check)" if s in AMBER else s]
                for p, s in spec.provenance.items() if not p.startswith("views.")]
    partial = (abstain.partial if abstain else None) or {}
    known, suggested = partial.get("known", {}), partial.get("suggested", {})
    rows = []
    for axis in "xyz":
        path = f"envelope.{axis}_mm"
        if path in edits:
            rows.append([path, edits[path], "user_edited"])
        elif path in known:
            rows.append([path, known[path], "found"])
        elif path in suggested:
            rows.append([path, "", f"{MISSING} — suggested {suggested[path]:g} mm"])
        else:
            rows.append([path, "", MISSING])
    return rows


def parse_edits(rows, shown: dict) -> tuple[dict[str, float | None], list[str]]:
    """Values the user changed, keyed by path. A cleared box maps to None, telling the caller to drop any earlier
    edit and revert to the measured or suggested value. Every bad entry is reported; none raises."""
    edits: dict[str, float | None] = {}
    errors: list[str] = []
    for row in rows or []:
        if len(row) < 2 or str(row[0]).strip() not in shown:
            continue
        path, text = str(row[0]).strip(), str(row[1]).strip()
        old, _source = shown[path]
        if text in ("", "None", "nan"):
            if old not in ("", None):
                edits[path] = None
            continue
        try:
            value = float(text.replace(",", "."))
        except ValueError:
            errors.append(f"{path}: '{text}' is not a number")
            continue
        if not (value > 0 and math.isfinite(value)):
            errors.append(f"{path}: must be more than 0")
            continue
        if old in ("", None) or abs(value - float(old)) > 1e-9:
            edits[path] = value
    return edits, errors


def face_gallery(observed: Observed, spec: MultiViewSpec | None) -> list[tuple[np.ndarray, str]]:
    if spec is None:
        return [(255 - mask, f"{face}: input") for face, mask in observed.masks.items()]
    items = []
    for face in CANONICAL_FACES:
        ol = getattr(spec.views, face)
        mask = outline_mask(ol.outer, ol.inner, *face_size(face, spec.envelope), px=256)
        badge = BADGES.get(observed.filled_by.get(face, ol.source), ol.source)
        # the merge warning is tagged with whichever raw face was photographed: this face, or its opposite
        tags = (f"{face}: merged", f"{OPPOSITE_OF[face]}: merged")
        merged = next((w.split(": ", 1)[1] for w in spec.warnings if w.startswith(tags)), "")
        items.append((255 - mask, f"{face}: {badge}" + (f", {merged}" if merged else "")))
    return items


def ai_faces(observed: Observed) -> list[str]:
    """Canonical faces nobody photographed and that were drawn by Qwen-Image or predicted by TripoSR: the only
    ones worth rejecting. An observed, mirrored or already-assumed face has nothing left to fall back to."""
    return [f for f in CANONICAL_FACES if observed.filled_by.get(f) in ("qwen-image", "triposr")]


def read_rows(observed: Observed) -> list[list]:
    rows = []
    for o in observed.observations:
        for lv in o.values:
            where = (f"hole {lv.hole_index + 1}" if lv.hole_index is not None
                     else f"{lv.axis} axis" if lv.axis else "not linked")
            rows.append([o.face, lv.reading.text, lv.reading.value_mm, where])
    return rows


class Handlers:
    """The two buttons, as plain methods so tests call them without a browser."""

    def __init__(self, pipe: MvPipeline):
        self.pipe = pipe

    def analyze(self, paths, tags, reference) -> tuple:
        state = new_state()
        if not paths:
            return _pack(state, message="Add at least one image.")
        images = []
        for i, path in enumerate(paths):
            row = list(tags[i]) if tags is not None and i < len(tags) else []
            face = str(row[1]).strip().lower() if len(row) > 1 and row[1] else "auto"
            kind = str(row[2]).strip().lower() if len(row) > 2 and row[2] else "auto"
            if face not in FACE_CHOICES or kind not in KIND_CHOICES:
                return _pack(state, message=f"Row {i + 1}: face must be one of {', '.join(FACE_CHOICES)}; "
                                            f"kind one of {', '.join(KIND_CHOICES)}.")
            images.append(ImageInput(Path(path).read_bytes(), None if face == "auto" else face,
                                     None if kind == "auto" else kind))
        observed = self.pipe.observe(images, None if reference in (None, "", "none") else reference)
        if isinstance(observed, MvAbstain):
            return _pack(state, message=_abstain(observed))
        state["observed"] = observed
        return _pack(state, **self._review(observed, self.pipe.fuse(observed), state))

    def rebuild(self, rows, rejected, state) -> tuple:
        state = state or new_state()
        observed = state.get("observed")
        if observed is None:
            return _pack(state, message="Analyze images first.")
        rej = tuple(rejected or ())
        edits, errors = parse_edits(rows, state["shown"])
        if errors:
            res = self.pipe.fuse(observed, dict(state["edits"]), rejected=rej)
            review = self._review(observed, res, state, rej)
            # keep whatever the model/views/files/stats components already show, and the user's own typed rows
            # (not the recomputed ones) so they can see and fix the exact bad cell
            shown = state.get("warnings", review.get("warnings"))  # the last build's warnings, slicer notes included
            return _pack(state, **{**review, "values": rows, "warnings": shown, "message": _bullets(errors)},
                         model=gr.update(), views=gr.update(), files=gr.update(), stats=gr.update())
        for path, value in edits.items():
            if value is None:
                state["edits"].pop(path, None)
            else:
                state["edits"][path] = value
        res = self.pipe.fuse(observed, dict(state["edits"]), rejected=rej)
        review = self._review(observed, res, state, rej)
        if isinstance(res, MvAbstain):
            return _pack(state, **review)
        sweep_outputs(OUT_ROOT)
        built = self.pipe.build(res, OUT_ROOT / uuid.uuid4().hex, observed.masks)
        if isinstance(built, MvAbstain):
            return _pack(state, **{**review, "message": _abstain(built)})
        views = [(255 - mask, face + (f", IoU {built.iou[face]:.2f}" if face in built.iou else ""))
                 for face, mask in built.views.items()]
        stats = (f"Print time {built.print_time_s / 60:.0f} min, filament {built.filament_g or 0:.1f} g"
                 if built.print_time_s else "G-code unavailable: slicer not installed")
        files = [str(p) for p in (built.stl, built.step, built.gcode) if p]
        state["warnings"] = _bullets(built.warnings)
        return _pack(state, **{**review, "warnings": state["warnings"],
                               "message": "Built. Check every value marked (check)."},
                     model=str(built.stl), views=views, files=files, stats=stats)

    def _review(self, observed: Observed, res, state: dict, rejected=()) -> dict:
        spec = None if isinstance(res, MvAbstain) else res
        abstain = res if isinstance(res, MvAbstain) else None
        rows = value_rows(spec, abstain, state["edits"])
        state["shown"] = {r[0]: (r[1], r[2]) for r in rows}
        warnings = spec.warnings if spec is not None else observed.warnings
        message = _abstain(abstain) if abstain else "Review the faces and values, edit any number, then Rebuild."
        # a rejected face is refilled as "assumed", which drops it from ai_faces: keep it offered and ticked
        # anyway, so rejecting it does not quietly wear off on the next Rebuild
        keep = set(ai_faces(observed)) | set(rejected)
        choices = [f for f in CANONICAL_FACES if f in keep]
        return {"faces": face_gallery(observed, spec), "reads": read_rows(observed), "values": rows,
                "warnings": _bullets(warnings), "message": message,
                "rejected": gr.update(choices=choices, value=list(rejected))}


def build_app(pipe: MvPipeline) -> gr.Blocks:
    handlers = Handlers(pipe)
    with gr.Blocks(title="Sketch-to-CAD lab", delete_cache=(TTL_S, TTL_S)) as app:
        state = gr.State(new_state())
        gr.Markdown("# Sketch-to-CAD: multi-view lab\n\n" + NOTICE)
        with gr.Row():
            files = gr.File(label="Face images", file_count="multiple", file_types=["image"], type="filepath")
            with gr.Column():
                tags = gr.Dataframe(headers=["file", "face", "kind"], type="array", interactive=True,
                                    label=f"Face: {', '.join(FACE_CHOICES)}. Kind: {', '.join(KIND_CHOICES)}.")
                reference = gr.Dropdown(REFERENCES, value="none", label="Reference object in the photos")
                analyze = gr.Button("Analyze", variant="primary")
        message = gr.Markdown()
        with gr.Row():
            faces = gr.Gallery(label="Canonical faces", columns=3, height=280)
            reads = gr.Dataframe(headers=["face", "text", "value", "linked to"], type="array", interactive=False,
                                 label="Handwriting read")
        values = gr.Dataframe(headers=["field", "value (mm)", "source"], type="array", interactive=True,
                              label="Values: edit the value column")
        rejected = gr.CheckboxGroup([], label="Reject an AI-drawn face and use a rectangle")
        warnings = gr.Markdown()
        rebuild = gr.Button("Rebuild", variant="primary")
        with gr.Row():
            model = gr.Model3D(label="Part")
            views = gr.Gallery(label="Rendered views", columns=3, height=320)
        stats = gr.Markdown()
        downloads = gr.File(label="STL, STEP, G-code", file_count="multiple")
        outputs = [faces, reads, values, warnings, message, model, views, downloads, stats, state, rejected]
        files.change(tag_rows, files, tags)
        analyze.click(handlers.analyze, [files, tags, reference], outputs)
        rebuild.click(handlers.rebuild, [values, rejected, state], outputs)
    return app


if __name__ == "__main__":
    load_dotenv()
    build_app(default_pipeline()).launch()
