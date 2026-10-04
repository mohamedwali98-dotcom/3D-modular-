"""What the Studio's buttons do, as plain methods that tests call without a browser. Each returns a small view model
(Review, Model, Exported) that app.py maps onto components. Spec 2026-09-23-studio section 8."""
from __future__ import annotations

import json
import logging
import math
import os
import random
import re
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from s2c import MAX_PIXELS, obs
from s2c.multiview import spec as S
from s2c.multiview.artifacts import ROOT, build_part, bundle, export_part, sweep
from s2c.multiview.finish import largest_finish
from s2c.multiview.fuse import fuse_envelope
from s2c.multiview.pipeline import ImageInput, MvPipeline
from s2c.multiview.raster import iou, outline_mask
from s2c.multiview.settings import (
    DENSITIES,
    AiSettings,
    ExportSettings,
    GeometrySettings,
    MeshSettings,
    PrintSettings,
    filament_metres,
)
from s2c.multiview.sheet import SKIP
from s2c.multiview.sheet_read import choose_naming, read_drawing
from s2c.studio.session import Item, SessionStore
from s2c.studio.theme import FACE_BADGES, TRUSTED, bullet_html, card, chip, source_chip, stats_html

FACE_CHOICES = ["auto", *S.FACES]
KIND_CHOICES = ["auto", "sketch", "photo", "drawing"]
REFERENCES = ["none", "1 TND", "1 EUR", "2 EUR", "card", "a4"]
AXES = ("x", "y", "z")
AXIS_LABEL = {"x": "Width (X)", "y": "Height (Y)", "z": "Depth (Z)"}
EXAMPLES = Path(__file__).resolve().parents[2] / "examples" / "mv" / "sketches"
SHEET_EXAMPLE = Path(__file__).resolve().parents[2] / "examples" / "mv" / "sheet" / "sheet.png"
SHEETS = "sheets"  # crops of split drawing sheets: <root>/sheets/<session id>/<sheet id>_<view>.png
PROJECTIONS = {"auto": "read from the drawing", "first": "first-angle (ISO)", "third": "third-angle (US)"}
_PROJECTION_SOURCE = {"symbol": "set by the projection symbol on the sheet, which overrides the switch",
                      "labels": "set by the view labels, which override the switch",
                      "setting": "from the projection switch",
                      "drawing": "read from how the views agree with each other"}
_FEATURE = re.compile(r"(features|finishes)\[(\d+)\]\.(\w+)")
_FIELD_WORDS = {"a_mm": "position a", "b_mm": "position b", "diameter_mm": "diameter", "depth_mm": "depth",
                "width_mm": "width", "length_mm": "length", "angle_deg": "angle", "radius_mm": "size",
                "height_mm": "height"}
_FEATURE_WORDS = {"boss": "Pin"}
log = logging.getLogger(__name__)


@dataclass
class Review:
    ok: bool
    stage: str  # "capture" when the images themselves failed, else "review"
    message_html: str
    sizes: dict[str, dict] = field(default_factory=dict)  # axis -> value, placeholder, info, required
    rows: list[list] = field(default_factory=list)       # field, value, source chip
    faces: list[tuple[np.ndarray, str]] = field(default_factory=list)
    ai_faces: list[str] = field(default_factory=list)
    warnings_html: str = ""
    reads: list[list] = field(default_factory=list)
    seed: int = 7
    unchecked: int = 0
    rejected: list[str] = field(default_factory=list)         # currently-rejected faces, for the checkbox value
    reject_choices: list[str] = field(default_factory=list)   # ai_faces plus rejected, canonical order


@dataclass
class Model:
    ok: bool
    message_html: str
    preview: str | None = None
    views: list[tuple[np.ndarray, str]] = field(default_factory=list)
    stats_html: str = ""
    open_step: int | None = None  # the step whose controls can fix a failure the user cannot fix where they are


@dataclass
class Exported:
    message_html: str
    files: list[str] = field(default_factory=list)
    zip_path: str | None = None
    stats_html: str = ""


def parse_size(text) -> float | None:
    """'42,5', '42 mm' and ' 42 ' are 42.5, 42 and 42; anything that is not a positive number is None."""
    t = str(text or "").strip().lower().removesuffix("mm").strip().replace(",", ".")
    try:
        value = float(t)
    except ValueError:
        return None
    return value if math.isfinite(value) and value > 0 else None


def duration_text(seconds: float) -> str:
    """2692 s is '45 min' and 6300 s is '1 h 45 min': hours are floored, never rounded up."""
    hours, minutes = divmod(round(seconds / 60), 60)
    return f"{hours} h {minutes} min" if hours else f"{minutes} min"


def field_label(spec: S.MultiViewSpec, path: str) -> str:
    m = _FEATURE.fullmatch(path)
    if not m:
        return path
    group, k, name = m.group(1), int(m.group(2)), m.group(3)
    if group == "finishes":
        return f"{spec.finishes[k].type.capitalize()} {_FIELD_WORDS.get(name, name)}"
    f = spec.features[k]
    return f"{_FEATURE_WORDS.get(f.type, f.type.capitalize())} {k + 1} ({f.face}) · {_FIELD_WORDS.get(name, name)}"


def _value(spec: S.MultiViewSpec, path: str) -> float:
    data = spec.model_dump()
    m = _FEATURE.fullmatch(path)
    if m:
        return data[m.group(1)][int(m.group(2))][m.group(3)]
    head, name = path.split(".", 1)
    return data[head][name]


_REASON_WORDS = {"bad_image": "Image not readable"}


def _reason_words(reason: str) -> str:
    return _REASON_WORDS.get(reason, reason.replace("_", " ").capitalize())


def _decode(data: np.ndarray) -> np.ndarray | None:
    """The image, or None when it is not one or has more pixels than OpenCV may decode (s2c/__init__.py)."""
    try:
        return cv2.imdecode(data, cv2.IMREAD_COLOR) if data.size else None
    except cv2.error:
        return None


def _read_image(path) -> np.ndarray | None:
    try:
        data = np.fromfile(str(path), np.uint8)
    except OSError:
        return None
    return _decode(data)


def _sheet_notes(session) -> list[str]:
    """What the Capture card says about each sheet: its views, the projection used and what set it, and the
    split's and the naming's warnings."""
    notes = []
    for path, sheet, naming in session.sheets.values():
        views = sum(f != SKIP for f in naming.faces)
        notes.append(f"{Path(path).name}: {views} views, {PROJECTIONS[naming.projection]} projection, "
                     f"{_PROJECTION_SOURCE[naming.projection_source]}.")
        notes += sheet.warnings + naming.warnings
    return notes


def _rename(session, sheet_id: str, naming) -> None:
    """New names for a sheet's views, except those the user picked by hand. A name a hand-picked view already
    holds is never given to a second view of that sheet: that view is left for the user."""
    items = [i for i in session.items if i.sheet_id == sheet_id]
    picked = {i.face for i in items if i.hand_face and i.face != "auto"}
    for item in items:
        if item.hand_face:
            continue
        face = naming.faces[item.view]
        if face in picked:
            naming.warnings.append(f"View {item.view + 1} would be the {face}, which you gave another view by hand; "
                                   "pick its face.")
            face = "auto"
        item.face = "auto" if face == SKIP else face


def _abstain_card(a: S.MvAbstain) -> str:
    return card(f"Stopped at {a.stage}: {_reason_words(a.reason)}", a.remedy, "stop")


FINISH_FAILURES = frozenset({"fillet_failed", "chamfer_failed"})  # build.BuildError reasons of a finish


class Studio:
    def __init__(self, pipe: MvPipeline, store: SessionStore | None = None, root: Path = ROOT):
        self.pipe, self.store, self.root = pipe, store or SessionStore(), Path(root)

    # ---- capture -----------------------------------------------------------------------------------------
    def add_images(self, sid: str, paths, face: str = "auto", use_reader: bool | None = None) -> None:
        """An image whose face is "auto" and that is a drawing sheet becomes one item per view (drawing-sheet spec
        3.6); any other image is one item. `use_reader` is the Qwen-VL switch as it is now: a sheet is read once, at
        its drop, so it must not wait for Analyze to learn the switch is off."""
        session = self.store.get(sid)
        if use_reader is not None:
            session.ai = session.ai.model_copy(update={"use_reader": use_reader})
        face = face if face in FACE_CHOICES else "auto"
        for p in paths or []:
            views = None
            if face == "auto":
                try:
                    views = self._split(session, str(p))
                except Exception as e:  # noqa: BLE001 - the split is a refinement: an upload must never be lost to it
                    obs.fallback("sheet_split", e)
            session.items += views or [Item(uuid.uuid4().hex[:8], str(p), Path(p).name, face)]
        session.sheet_notes = _sheet_notes(session)

    def _split(self, session, path: str) -> list[Item] | None:
        """One "drawing" item per view of a sheet, cropped at the sheet's scale, or None when the image is not a
        sheet. A sheet needs two aligned line-drawn views (is_sheet), two of them named under the scale check (two
        unrelated sketches can line up) and one named view that is not round (holes left by a tight crop)."""
        image = _read_image(path)
        if image is None:
            return None
        pipe = self.pipe.configured(session.ai)  # the user's AI switches (Qwen-VL reader off) hold here too
        read = read_drawing(image, session.projection, reader=pipe.reader, service=pipe.reading(),
                            keep_unnamed=True)
        if read is None or isinstance(read, S.MvAbstain):  # no sheet of views, or a photo with no usable page
            return None
        sheet_id, folder = uuid.uuid4().hex[:8], self._sheet_folder(session.id)
        items = []
        for c in read.crops:
            crop = folder / f"{sheet_id}_{c.view}.png"
            crop.write_bytes(c.png)
            items.append(Item(uuid.uuid4().hex[:8], str(crop), f"{Path(path).name} · view {c.view + 1}", c.face,
                              "drawing", sheet_id, c.view, mm_per_px=read.scale.mm_per_px,
                              scale_confirmed=read.scale.confirmed, numbers=False,
                              line_art=read.kind == "sketch"))
        read.sheet.warnings += [w for w in read.warnings if w not in read.naming.warnings]
        if read.kind == "sketch":  # its views are on the rectified page, not on the photo: rename them there
            page = folder / sheet_id / f"{Path(path).stem}.png"
            page.parent.mkdir(exist_ok=True)
            cv2.imencode(".png", read.page.image)[1].tofile(str(page))
            path = str(page)
        session.sheets[sheet_id] = (path, read.sheet, read.naming)
        return items

    def _sheet_folder(self, sid: str) -> Path:
        """The session's folder of sheet crops, created after the sweep."""
        self._sweep_sheets()
        folder = self.root / SHEETS / sid
        folder.mkdir(parents=True, exist_ok=True)
        return folder

    def _sweep_sheets(self) -> None:
        """Crops live as long as their session: the folders of live sessions (and the sheets folder holding them)
        are touched first, so the sweep takes only the folders of sessions the store has dropped."""
        sheets = self.root / SHEETS
        now = time.time()
        for sid in self.store.ids():
            folder = sheets / sid
            if folder.is_dir():
                os.utime(folder, (now, now))
                os.utime(sheets, (now, now))
        sweep(sheets)

    def _sweep(self) -> None:
        self._sweep_sheets()
        sweep(self.root)

    def set_projection(self, sid: str, projection: str) -> None:
        """Rename the views of every sheet under `projection`, except the faces the user picked by hand. A sheet's
        projection symbol, or labels that all follow one projection, still decide, and the notes say so."""
        if projection not in PROJECTIONS:
            return
        session = self.store.get(sid)
        session.projection = projection
        for sheet_id, (path, sheet, _) in list(session.sheets.items()):
            image = _read_image(path)
            if image is None:  # the upload has expired; its views keep their names
                continue
            naming = choose_naming(sheet, image, projection, reader=self.pipe.reader)
            session.sheets[sheet_id] = (path, sheet, naming)
            _rename(session, sheet_id, naming)
        session.sheet_notes = _sheet_notes(session)

    def sheet_html(self, sid: str) -> str:
        session = self.store.get(sid)
        warned = any(sheet.warnings or naming.warnings for _, sheet, naming in session.sheets.values())
        return bullet_html("Drawing sheet", session.sheet_notes, "check" if warned else "info")

    def set_face(self, sid: str, item_id: str, face: str) -> None:
        for item in self.store.get(sid).items:
            if item.id == item_id and face in FACE_CHOICES:
                item.face, item.hand_face = face, True

    def set_kind(self, sid: str, item_id: str, kind: str) -> None:
        for item in self.store.get(sid).items:
            if item.id == item_id and kind in KIND_CHOICES:
                item.kind = kind

    def remove(self, sid: str, item_id: str) -> None:
        session = self.store.get(sid)
        session.items = [i for i in session.items if i.id != item_id]
        kept = {i.sheet_id for i in session.items}
        session.sheets = {k: v for k, v in session.sheets.items() if k in kept}
        session.sheet_notes = _sheet_notes(session)

    def load_examples(self, sid: str) -> None:
        session = self.store.get(sid)
        session.items, session.sheets, session.sheet_notes = [], {}, []
        for entry in json.loads((EXAMPLES / "examples.json").read_text()):
            session.items.append(Item(uuid.uuid4().hex[:8], str(EXAMPLES / entry["file"]), entry["file"],
                                      entry["face"], entry["kind"]))

    def load_sheet_example(self, sid: str, use_reader: bool | None = None) -> None:
        """A first-angle sheet drawn by our own code from a known part; its README gives the sizes to type."""
        session = self.store.get(sid)
        session.items, session.sheets = [], {}
        self.add_images(sid, [SHEET_EXAMPLE], use_reader=use_reader)

    def coverage_html(self, sid: str) -> str:
        items = self.store.get(sid).items
        if not items:
            return chip("Add at least one image", "info")
        counts = {face: sum(1 for i in items if i.face == face) for face in S.FACES}
        auto = sum(1 for i in items if i.face == "auto")
        chips = []
        for canon, opposite in (("front", "back"), ("top", "bottom"), ("right", "left")):
            n, m = counts[canon], counts[opposite]
            if n or m:
                chips.append(chip(f"{canon} ✓{n}" + (f" · {opposite} ✓{m}" if m else ""), "ok"))
            else:
                chips.append(chip(f"{canon}: AI will draw it", "ai"))
        if auto:
            chips.append(chip(f"{auto} image(s) face: auto", "check"))
        return "".join(chips)

    # ---- review --------------------------------------------------------------------------------------------
    def analyze(self, sid: str, reference: str, ai: AiSettings) -> Review:
        session = self.store.get(sid)
        if not session.items:
            return Review(False, "capture", card("No images yet", "Drop sketches or photos, then Analyze.", "check"))
        if ai.randomize_seed:
            ai = ai.model_copy(update={"seed": random.randint(0, 2**31 - 1)})
        session.ai, session.reference = ai, reference
        images = []
        for item in session.items:
            try:
                data = Path(item.path).read_bytes()
            except OSError:
                missing = card("Image missing", f"{item.name} is no longer available. Add it again.", "stop")
                return Review(False, "capture", missing)
            if _decode(np.frombuffer(data, np.uint8)) is None:
                bad = card("Image not readable", f"{item.name} is not an image we can read, or it has more than "
                           f"{MAX_PIXELS // 1_000_000} megapixels. Upload a JPEG or PNG, smaller if needed.", "stop")
                return Review(False, "capture", bad)
            images.append(ImageInput(data, None if item.face == "auto" else item.face,
                                     None if item.kind == "auto" else item.kind, mm_per_px=item.mm_per_px,
                                     scale_confirmed=item.scale_confirmed, numbers=item.numbers,
                                     line_art=item.line_art))
        pipe = self.pipe.configured(ai)
        # a new analysis replaces the last one even when it stops: Build, Redraw and Export never act on old images
        session.observed, session.spec, session.part, session.exported = None, None, None, None
        session.edits, session.rejected = {}, ()
        observed = pipe.observe(images, None if reference in (None, "", "none") else reference)
        if isinstance(observed, S.MvAbstain):
            return Review(False, "capture", _abstain_card(observed), seed=ai.seed)
        session.observed, session.edits, session.rejected, session.part = observed, {}, (), None
        return self._review(session, pipe.fuse(observed, geometry=session.geometry))

    def redraw(self, sid: str, ai: AiSettings | None = None) -> Review:
        """New drawings of the AI faces with the switches as they are now (`ai`): a helper switched off since Analyze
        is never called again, and a seed typed since is the one used."""
        session = self.store.get(sid)
        if session.observed is None:
            return Review(False, "capture", card("Nothing to redraw", "Analyze your images first.", "check"))
        ai = ai or session.ai
        if not (ai.use_qwen_image or ai.use_triposr):
            return Review(False, "review", card("Nothing to redraw with", "Turn on Qwen-Image or TripoSR to redraw "
                                                "the AI faces.", "check"), seed=ai.seed)
        if ai.randomize_seed:
            seed = random.randint(0, 2**31 - 1)
        else:  # the next seeds, or the one the user typed since
            seed = ai.seed + ai.attempts if ai.seed == session.ai.seed else ai.seed
        session.ai = ai.model_copy(update={"seed": seed % 2**31})
        return self._review(session, self._fuse(session))

    def suggested_sizes(self, sid: str) -> dict[str, str]:
        """The suggestion for each size the last review is missing, e.g. {"z": "10"}: a view that shows that axis
        next to a known one, scaled. The same fuse_envelope call the review made, so the same partial."""
        session = self.store.get(sid)
        if session.observed is None:
            return {}
        res = fuse_envelope(session.observed.observations, dict(session.edits))
        suggested = (res.partial or {}).get("suggested", {}) if isinstance(res, S.MvAbstain) else {}
        return {axis: f"{suggested[f'envelope.{axis}_mm']:g}" for axis in AXES if f"envelope.{axis}_mm" in suggested}

    def _fuse(self, session):
        return self.pipe.configured(session.ai).fuse(session.observed, dict(session.edits),
                                                     rejected=session.rejected, geometry=session.geometry)

    def _review(self, session, res) -> Review:
        spec = None if isinstance(res, S.MvAbstain) else res
        abstain = res if isinstance(res, S.MvAbstain) else None
        session.spec = spec
        observed = session.observed
        known = (abstain.partial or {}).get("known", {}) if abstain else {}
        suggested = ((abstain.partial or {}).get("suggested", {}) if abstain else {})
        sizes = {}
        for axis in AXES:
            path = f"envelope.{axis}_mm"
            if spec is not None:
                value, prov = getattr(spec.envelope, f"{axis}_mm"), spec.provenance[path]
            else:
                value = known.get(path, session.edits.get(path))
                read = (abstain.partial or {}).get("provenance", {}) if abstain else {}
                prov = "user_edited" if path in session.edits else read.get(path)  # where a known size came from
            hint = suggested.get(path)
            sizes[axis] = {"value": "" if value is None else f"{value:g}",
                           "placeholder": f"suggested {hint:g}" if hint else "mm",
                           "info": f"from: {prov.replace('_', ' ')}" if prov
                           else "Required: type it or use the suggestion",
                           "required": value is None}
            session.shown[path] = value
        rows, paths = [], []
        if spec is not None:
            for path, prov in spec.provenance.items():
                if path.startswith(("envelope.", "views.")):
                    continue
                paths.append(path)
                session.shown[path] = _value(spec, path)
                rows.append([field_label(spec, path), _value(spec, path), source_chip(prov)])
        session.row_paths = paths
        faces, ai_faces = [], []
        if spec is not None:
            for face in S.CANONICAL_FACES:
                ol = getattr(spec.views, face)
                a, b = S.face_size(face, spec.envelope)
                mask = 255 - outline_mask(ol.outer, ol.inner, a, b, px=256)
                who = observed.filled_by.get(face, ol.source)
                text, _ = FACE_BADGES.get(who, (who, "info"))
                merged = next((w.split(": ", 1)[1] for w in spec.warnings if w.startswith(f"{face}: merged")), "")
                faces.append((mask, f"{face}: {text}" + (f" · {merged}" if merged else "")))
                if who in ("qwen-image", "triposr"):
                    ai_faces.append(face)
        else:
            faces = [(255 - m, f"{face}: your image") for face, m in observed.masks.items()]
        rejected = [f for f in S.CANONICAL_FACES if f in session.rejected]
        reject_choices = [f for f in S.CANONICAL_FACES if f in ai_faces or f in rejected]
        warnings = spec.warnings if spec is not None else observed.warnings
        info = [w for w in warnings if ": merged " in w]
        check = [w for w in warnings if w not in info]
        reads = [[o.face, lv.reading.text, lv.reading.value_mm,
                  f"hole {lv.hole_index + 1}" if lv.hole_index is not None
                  else f"{lv.axis} axis" if lv.axis else "not linked"]
                 for o in observed.observations for lv in o.values]
        unchecked = sum(1 for p in paths if spec.provenance[p] not in TRUSTED)  # the amber rows of the table
        message = (_abstain_card(abstain) if abstain else
                   card("Ready to build", "Check the amber values, reject any AI face you do not trust, then Build.",
                        "ok"))
        return Review(spec is not None, "review", message, sizes, rows, faces, ai_faces,
                      bullet_html("Check", check, "check") + bullet_html("Info", info, "info"), reads,
                      session.ai.seed, unchecked, rejected, reject_choices)

    # ---- build ----------------------------------------------------------------------------------------------
    def build(self, sid: str, sizes: dict[str, str], rows, rejected,
              geometry: GeometrySettings) -> tuple[Review, Model]:
        session = self.store.get(sid)
        if session.observed is None:
            msg = card("Nothing to build", "Analyze your images first.", "check")
            return Review(False, "capture", msg), Model(False, msg)
        errors, edits = [], dict(session.edits)
        for axis in AXES:
            text = str(sizes.get(axis, "")).strip()
            path = f"envelope.{axis}_mm"
            if not text:
                edits.pop(path, None)
                continue
            value = parse_size(text)
            if value is None:
                errors.append(f"{AXIS_LABEL[axis]}: '{text}' is not a size in mm")
            elif session.shown.get(path) is None or abs(value - float(session.shown[path])) > 1e-9:
                edits[path] = value
        for path, row in zip(session.row_paths, rows or []):
            text = str(row[1]).strip() if len(row) > 1 else ""
            if not text:
                edits.pop(path, None)
                continue
            value = parse_size(text)
            shown = session.shown.get(path)
            if value is None:
                if text not in ("None", str(shown)):
                    errors.append(f"{row[0]}: '{row[1]}' is not a positive number")
            elif shown is None or abs(value - float(shown)) > 1e-9:
                edits[path] = value
        if errors:
            msg = card("Please fix these values", " · ".join(errors), "stop")
            review = self._review(session, self._fuse(session))
            review.ok, review.message_html = False, msg
            for axis in AXES:  # what the user sent stays on screen, the typo included, for them to fix
                if axis in sizes and axis in review.sizes:
                    review.sizes[axis]["value"] = str(sizes[axis])
            review.rows, review.rejected = list(rows or review.rows), list(rejected or [])
            return review, Model(False, msg)
        cleared = tuple(rejected or ()) != session.rejected and any(k.startswith("features[") for k in edits)
        if cleared:  # a reject change can renumber the pockets and pins: a value typed by index could land elsewhere
            edits = {k: v for k, v in edits.items() if not k.startswith("features[")}
        session.edits, session.rejected, session.geometry = edits, tuple(rejected or ()), geometry
        review = self._review(session, self._fuse(session))
        if cleared:
            note = "The feature values you typed were cleared because the rejected faces changed. Check them again."
            review.warnings_html = bullet_html("Check", [note], "check") + review.warnings_html
        if not review.ok:
            session.part = session.exported = None  # the part on screen no longer matches the review
            return review, Model(False, review.message_html)
        return review, self._model(session)

    def rebuild_geometry(self, sid: str, geometry: GeometrySettings) -> tuple[Review, Model]:
        """A finish/clearance/snap change re-fuses (so the Review table shows the new snapped values, not stale
        ones) and rebuilds; the caller maps the returned review onto the sizes and the values table."""
        session = self.store.get(sid)
        if session.observed is None:
            msg = card("Nothing to build", "Analyze your images first.", "check")
            return Review(False, "capture", msg), Model(False, msg)
        session.geometry = geometry
        review = self._review(session, self._fuse(session))
        if not review.ok:
            session.part = session.exported = None
            return review, Model(False, review.message_html)
        return review, self._model(session)

    def _model(self, session) -> Model:
        self._sweep()
        part = build_part(session.spec, session.geometry, self.root)
        if isinstance(part, S.MvAbstain):
            # A finish that cannot be built is fixed with the Geometry controls, which live in step 3 (index 2);
            # the part on screen is still good, so it is kept rather than blanking the viewer.
            if part.reason in FINISH_FAILURES and session.geometry.finish != "none":
                return self._finish_failure_model(session)
            session.part = None
            return Model(False, _abstain_card(part))
        session.part, session.exported = part, None
        return self._built_model(part, session)

    def _built_model(self, part, session) -> Model:
        masks = session.observed.masks
        views = []
        for face, mask in part.views.items():
            score = f" · match {iou(mask, masks[face]):.2f}" if face in masks else ""
            views.append((255 - mask, f"{face}{score}"))
        x, y, z = part.bbox_mm
        stats = stats_html([("Size", f"{x:.1f} × {y:.1f} × {z:.1f} mm"),
                            ("Volume", f"{part.volume_mm3 / 1000:.2f} cm³"),
                            ("Solid mass, PLA", f"{part.volume_mm3 * DENSITIES['PLA'] / 1000:.1f} g")])
        note = " ".join(part.warnings)
        return Model(True, card("Part built", note or "Rotate the part, then choose formats and export.", "ok"),
                     str(part.preview), views, stats)

    def _finish_failure_model(self, session) -> Model:
        """The size that failed, and the largest one that would actually build, found by real builds so the
        suggestion is never a guess."""
        kind, size = session.geometry.finish, session.geometry.finish_mm
        best = largest_finish(session.spec, session.geometry)
        body = f"No {kind} fits this part." if best is None else f"The largest that builds is {best:g} mm."
        msg = card(f"{kind.capitalize()} {size:g} mm does not fit.", body, "stop")
        if session.part is None:
            return Model(False, msg, open_step=2)
        prev = self._built_model(session.part, session)
        return Model(False, msg, prev.preview, prev.views, prev.stats_html, open_step=2)

    # ---- export ---------------------------------------------------------------------------------------------
    def export(self, sid: str, export: ExportSettings, mesh: MeshSettings, printing: PrintSettings) -> Exported:
        session = self.store.get(sid)
        if session.part is None:
            return Exported(card("Nothing to export", "Build the part first.", "check"))
        self._sweep()
        res = export_part(session.part, export.formats, mesh, printing, self.pipe.slicer, self.pipe.profile)
        session.exported = res
        settings = {"mesh": mesh.model_dump(), "printing": printing.model_dump(), "ai": session.ai.model_dump(),
                    "geometry": session.geometry.model_dump(), "formats": export.formats}
        zip_path = bundle(session.part, res, settings)
        x, y, z = session.part.bbox_mm
        items = [("Size", f"{x:.1f} × {y:.1f} × {z:.1f} mm"), ("Files", str(len(res.files))),
                 ("Download", f"{zip_path.stat().st_size / 1024:.0f} KB")]
        if res.print_time_s:
            grams = res.filament_g or 0
            items += [("Print time", duration_text(res.print_time_s)),
                      ("Filament", f"{grams:.1f} g · {filament_metres(grams, printing.material):.2f} m")]
        tone = "check" if res.warnings else "ok"
        body = " · ".join(res.warnings) or "Every file is in the zip, with a manifest of the values and their sources."
        return Exported(card("Files ready", body, tone), [str(p) for p in res.files.values()], str(zip_path),
                        stats_html(items))
