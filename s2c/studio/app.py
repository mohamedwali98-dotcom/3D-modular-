"""The Studio layout: one Walkthrough, Capture -> Review -> Model & export. Spec 2026-09-23-studio section 8.
Run: uv run python app_mv_studio.py"""
from __future__ import annotations

import gradio as gr

from s2c.multiview.pipeline import MvPipeline, default_pipeline
from s2c.multiview.settings import (
    EDGE_LABELS,
    FORMATS,
    MATERIALS,
    NOZZLES,
    AiSettings,
    ExportSettings,
    GeometrySettings,
    MeshSettings,
    PrintSettings,
)
from s2c.studio.handlers import AXES, AXIS_LABEL, FACE_CHOICES, KIND_CHOICES, REFERENCES, Studio
from s2c.studio.status import header_html, provider_status
from s2c.studio.theme import CSS, THEME, card

NOTE = ("**Photos of real parts:** shoot straight on, the part lying flat on a plain surface, with a coin, a card or "
        "an A4 sheet in frame. **Sketches:** dark pen on white paper, one face per sheet, sizes in mm. "
        "**Drawing sheets:** drop the whole sheet; its views are split and named, and you type the sizes.")
PROJECTION_CHOICES = [("Auto (read from the drawing)", "auto"), ("First-angle (ISO)", "first"),
                      ("Third-angle (US)", "third")]
TEMPERATURES = "PLA 210/60 · PETG 240/80 · ABS 250/100 · ASA 255/100 · TPU 225/50 °C"
PATTERNS = ["grid", "gyroid", "rectilinear", "honeycomb", "cubic", "lightning"]


def _size_update(size: dict) -> dict:
    return gr.update(value=size.get("value", ""), placeholder=size.get("placeholder", "mm"), info=size.get("info"),
                     elem_classes=["size", "required"] if size.get("required") else ["size"])


def _preview_update(m):
    """A failed build (e.g. a finish that does not fit) leaves preview unset; the 3D viewer then keeps whatever
    it is already showing instead of being cleared."""
    return m.preview if m.preview is not None else gr.update()


def build_app(pipe: MvPipeline | None = None, studio: Studio | None = None) -> gr.Blocks:
    studio = studio or Studio(pipe or default_pipeline())
    status = provider_status(studio.pipe)

    def new_session() -> str:  # gr.State deep-copies its value; a bound method would drag the store's lock along
        return studio.store.new()

    with gr.Blocks(title="Sketch-to-CAD Studio", delete_cache=(3600, 3600)) as app:
        sid = gr.State(new_session)  # session.py sweeps idle sessions lazily: gr.State never fires delete_callback
        version = gr.State(0)
        built = gr.State(None)  # the preview GLB, handed to the viewer once its step is on screen
        gr.HTML(header_html(status))
        with gr.Walkthrough(selected=0) as walk:
            # ---- 1 capture ----
            with gr.Step("Capture", id=0):
                capture_msg = gr.HTML()
                with gr.Row():
                    with gr.Column(scale=3):
                        drop = gr.File(label="Drop sketches or photos: one or more per face (JPEG or PNG)",
                                       file_count="multiple", file_types=["image"], height=150)

                        @gr.render(inputs=[sid, version], triggers=[version.change, app.load])
                        def cards(session_id, _v):
                            items = studio.store.get(session_id).items
                            if not items:
                                gr.Markdown("*No images yet. Drop them above, or press **Try an example**.*")
                                return
                            with gr.Row(equal_height=False):
                                for item in items:
                                    with gr.Column(min_width=170, variant="panel"):
                                        gr.Image(item.path, height=140, interactive=False, show_label=False,
                                                 key=f"img-{item.id}")
                                        # interactive is explicit: inside a render Gradio 6.28 draws them disabled
                                        face = gr.Dropdown(FACE_CHOICES, value=item.face, label="Face",
                                                           interactive=True, key=f"face-{item.id}")
                                        kind = gr.Radio(KIND_CHOICES, value=item.kind, label="Type",
                                                        interactive=True, key=f"kind-{item.id}")
                                        remove = gr.Button("Remove", size="sm", key=f"rm-{item.id}")
                                    face.input(lambda v, s, i=item.id: (studio.set_face(s, i, v),
                                                                        studio.coverage_html(s))[1],
                                               [face, sid], [coverage])
                                    kind.input(lambda v, s, i=item.id: studio.set_kind(s, i, v), [kind, sid], None)
                                    remove.click(lambda s, n, i=item.id: (studio.remove(s, i), n + 1,
                                                                          studio.coverage_html(s),
                                                                          studio.sheet_html(s))[1:],
                                                 [sid, version], [version, coverage, sheet_card])
                    with gr.Column(scale=2, min_width=320):
                        coverage = gr.HTML()
                        sheet_card = gr.HTML()
                        reference = gr.Dropdown(REFERENCES, value="none", label="Reference object in the photos",
                                                info="Gives real millimetres from a photo")
                        projection = gr.Radio(PROJECTION_CHOICES, value="auto", label="Projection of drawing sheets",
                                              info="A projection symbol on the sheet overrides this")
                        gr.Markdown(NOTE)
                        with gr.Row():
                            example = gr.Button("Try an example", variant="secondary")
                            sheet_example = gr.Button("Try a drawing sheet", variant="secondary")
                        with gr.Accordion("Reading & AI", open=False):
                            use_reader = gr.Checkbox(True, label="Read handwriting with Qwen-VL",
                                                     info="Off: TrOCR only", interactive=status["Qwen-VL"])
                            ai = AiSettings()  # hosted image services start off: the user opts in
                            use_qwen = gr.Checkbox(ai.use_qwen_image, label="Draw missing faces with Qwen-Image",
                                                   info="Sends your images to a hosted service", interactive=status["Qwen-Image"])
                            use_rescue = gr.Checkbox(ai.use_rescue, label="Rescue sketches with an open outline",
                                                     info="The redraw is marked 'cleaned'. Sends your images to a hosted service",
                                                     interactive=status["Qwen-Image"])
                            use_triposr = gr.Checkbox(ai.use_triposr, label="TripoSR fallback for missing faces",
                                                      info="Sends your images to a hosted service when no local GPU runs it",
                                                      interactive=status["TripoSR"])
                            use_solaria = gr.Checkbox(ai.use_solaria, label="Hole depth from photos (Solaria)",
                                                      info="Photos only; adds 60–180 s. Sends your images to a hosted service",
                                                      interactive=status["Solaria"])
                            seed = gr.Number(7, precision=0, minimum=0, maximum=2**31 - 1, label="Seed",
                                             info="Same seed, same drawing")
                            randomize = gr.Checkbox(False, label="Randomize seed")
                            attempts = gr.Slider(1, 4, value=2, step=1, label="Attempts per face",
                                                 info="Each attempt uses the next seed, about 30 s")
                        with gr.Row():
                            analyze = gr.Button("Analyze →", variant="primary", scale=3)
                            cancel = gr.Button("Cancel", scale=1)
            # ---- 2 review ----
            with gr.Step("Review", id=1):
                review_msg = gr.HTML()
                with gr.Row():
                    with gr.Column(scale=3):
                        with gr.Row():
                            sizes = {a: gr.Textbox(label=AXIS_LABEL[a], max_lines=1, elem_classes=["size"])
                                     for a in AXES}
                        suggest = gr.Button("Use suggested sizes", size="sm")
                        values = gr.Dataframe(headers=["Field", "Value (mm)", "Source"], type="array",
                                              datatype=["str", "number", "html"], interactive=True,
                                              static_columns=[0, 2], column_widths=["45%", "20%", "35%"],
                                              label="Values: edit any number")
                        with gr.Accordion("Handwriting read", open=False):
                            reads = gr.Dataframe(headers=["Face", "Text", "Value", "Linked to"], type="array",
                                                 interactive=False)
                    with gr.Column(scale=2, min_width=320):
                        faces = gr.Gallery(label="Faces the part is built from", columns=3, height=260,
                                           object_fit="contain", fit_columns=False)
                        rejected = gr.CheckboxGroup([], label="Reject an AI face (use a rectangle)")
                        redraw = gr.Button("Redraw AI faces (next seeds)", size="sm")
                        warnings = gr.HTML()
                        build = gr.Button("Build part →", variant="primary")
            # ---- 3 model & export ----
            with gr.Step("Model & export", id=2):
                model_msg = gr.HTML()
                with gr.Row():
                    with gr.Column(scale=3):
                        model = gr.Model3D(label="Part", height=520, clear_color=(0, 0, 0, 0))
                        views = gr.Gallery(label="Rendered views (match against your images)", columns=3,
                                           height=200, object_fit="contain", fit_columns=False)
                    with gr.Column(scale=2, min_width=320):
                        stats = gr.HTML()
                        with gr.Accordion("Geometry", open=True):
                            snap = gr.Checkbox(True, label="Snap estimated values to standard sizes",
                                               info="Your own numbers are never snapped")
                            clearance = gr.Radio(["fine", "medium", "coarse"], value="medium",
                                                 label="Hole clearance class (ISO 273)")
                            finish = gr.Radio(["none", "fillet", "chamfer"], value="none", label="Edge finish")
                            finish_mm = gr.Slider(0.2, 10, value=1.0, step=0.1, label="Finish size (mm)")
                            finish_edges = gr.Dropdown([(v, k) for k, v in EDGE_LABELS.items()],
                                                       value="all_vertical", label="Edges")
                        with gr.Accordion("Mesh & export", open=False):
                            quality = gr.Radio(["draft", "normal", "fine"], value="normal", label="Mesh quality",
                                               info="Chord error 0.1 / 0.02 / 0.005 mm")
                        with gr.Accordion("3D print", open=False):
                            material = gr.Dropdown(list(MATERIALS), value="PLA", label="Material",
                                                   info=TEMPERATURES)
                            nozzle = gr.Dropdown(list(NOZZLES), value=0.4, label="Nozzle (mm)")
                            layer = gr.Slider(0.05, 0.32, value=0.2, step=0.01, label="Layer height (mm)",
                                              info="At most 0.75 × nozzle")
                            infill = gr.Slider(0, 100, value=20, step=5, label="Infill (%)")
                            pattern = gr.Dropdown(PATTERNS, value="grid", label="Infill pattern")
                            perimeters = gr.Slider(1, 8, value=3, step=1, label="Perimeters")
                            supports = gr.Radio(["off", "buildplate", "everywhere"], value="buildplate",
                                                label="Supports")
                            brim = gr.Slider(0, 10, value=0, step=1, label="Brim (mm)")
                            scale = gr.Number(100, minimum=50, maximum=200, label="Print scale (%)",
                                              info="G-code only; the CAD files stay true size")
                        formats = gr.CheckboxGroup([(v, k) for k, v in FORMATS.items()],
                                                   value=ExportSettings().formats, label="Formats")
                        export = gr.Button("Export selected", variant="primary")
                        export_msg = gr.HTML()
                        download = gr.DownloadButton("Download all (.zip)", visible=False, variant="primary")
                        files = gr.File(label="Files", file_count="multiple", interactive=False)

        review_outputs = [review_msg, *sizes.values(), values, faces, rejected, warnings, reads, seed, build]

        def show_review(r) -> list:
            label = f"Build anyway ({r.unchecked} unchecked) →" if r.unchecked else "Build part →"
            return [r.message_html, *(_size_update(r.sizes.get(a, {})) for a in AXES), r.rows, r.faces,
                    gr.update(choices=r.reject_choices, value=r.rejected), r.warnings_html, r.reads, r.seed,
                    gr.update(value=label)]

        def ai_settings(*v) -> AiSettings:
            return AiSettings(use_reader=v[0], use_qwen_image=v[1], use_rescue=v[2], use_triposr=v[3],
                              use_solaria=v[4], seed=int(v[5] or 0), randomize_seed=v[6], attempts=int(v[7]))

        def geometry_settings(*v) -> GeometrySettings:
            return GeometrySettings(snap=v[0], clearance=v[1], finish=v[2], finish_mm=float(v[3]),
                                    finish_edges=v[4])

        def on_upload(paths, s, n):
            studio.add_images(s, paths)
            return None, n + 1, studio.coverage_html(s), studio.sheet_html(s)

        def on_example(s, n):
            studio.load_examples(s)
            return n + 1, studio.coverage_html(s), studio.sheet_html(s)

        def on_sheet_example(s, n):
            studio.load_sheet_example(s)
            return n + 1, studio.coverage_html(s), studio.sheet_html(s)

        def on_projection(p, s, n):
            studio.set_projection(s, p)
            return n + 1, studio.coverage_html(s), studio.sheet_html(s)

        def on_analyze(s, ref, *v, progress=gr.Progress()):  # noqa: B008 - how Gradio injects a progress bar
            progress(0.1, desc="Reading your images, then drawing any missing faces…")
            try:
                ai = ai_settings(*v)
            except ValueError as e:
                msg = card("Check the AI settings", str(e), "stop")
                return [gr.update(), msg, *[gr.update()] * len(review_outputs)]
            r = studio.analyze(s, ref, ai)
            if r.stage == "capture":
                return [gr.update(), r.message_html, *[gr.update()] * len(review_outputs)]
            return [gr.Walkthrough(selected=1), "", *show_review(r)]

        def on_redraw(s):
            return show_review(studio.redraw(s))

        def on_suggest(s, *boxes):  # fills only the empty boxes; Build records them as the user's own values
            hints = studio.suggested_sizes(s)
            return [hints[a] if a in hints and not str(v or "").strip() else gr.update() for a, v in zip(AXES, boxes)]

        def on_build(s, x, y, z, rows, rej, *g):
            try:
                geometry = geometry_settings(*g)
            except ValueError as e:
                msg = card("Check the geometry settings", str(e), "stop")
                return [gr.update(), msg, *[gr.update()] * (len(review_outputs) - 1), msg, gr.update(), [], ""]
            r, m = studio.build(s, {"x": x, "y": y, "z": z}, rows, rej, geometry)
            target = 2 if m.ok else m.open_step  # a finish that cannot be built opens step 3, where its controls are
            if target is None and r.ok:  # the part itself failed: say so on the step the user is looking at
                r.message_html = m.message_html
            step = gr.update() if target is None else gr.Walkthrough(selected=target)
            return [step, *show_review(r), m.message_html, _preview_update(m), m.views, m.stats_html]

        def on_geometry(s, *g):
            try:
                geometry = geometry_settings(*g)
            except ValueError as e:
                msg = card("Check the geometry settings", str(e), "stop")
                return msg, gr.update(), gr.update(), gr.update(), *[gr.update()] * len(sizes), gr.update()
            r, m = studio.rebuild_geometry(s, geometry)
            return (m.message_html, _preview_update(m), m.views, m.stats_html,
                    *(_size_update(r.sizes.get(a, {})) for a in AXES), r.rows)

        def on_export(s, fmts, q, mat, noz, lay, inf, pat, per, sup, br, sc):
            try:
                printing = PrintSettings(material=mat, nozzle_mm=float(noz), layer_mm=float(lay),
                                         infill_pct=int(inf), infill_pattern=pat, perimeters=int(per),
                                         supports=sup, brim_mm=float(br), scale_pct=float(sc or 100))
                export_settings = ExportSettings(formats=list(fmts or []))
            except ValueError as e:
                return card("Check the print settings", str(e).split("\n")[-1], "stop"), gr.update(), [], gr.update()
            if not export_settings.formats:
                return card("Choose at least one format", "Tick the files you want.", "check"), gr.update(), [], \
                    gr.update()
            e = studio.export(s, export_settings, MeshSettings(quality=q), printing)
            button = gr.DownloadButton(value=e.zip_path, visible=e.zip_path is not None)
            return e.message_html, button, e.files, e.stats_html or gr.update()

        ai_inputs = [use_reader, use_qwen, use_rescue, use_triposr, use_solaria, seed, randomize, attempts]
        geometry_inputs = [snap, clearance, finish, finish_mm, finish_edges]
        drop.upload(on_upload, [drop, sid, version], [drop, version, coverage, sheet_card])
        example.click(on_example, [sid, version], [version, coverage, sheet_card])
        sheet_example.click(on_sheet_example, [sid, version], [version, coverage, sheet_card])
        projection.input(on_projection, [projection, sid, version], [version, coverage, sheet_card])
        app.load(studio.coverage_html, [sid], [coverage])
        analyzing = analyze.click(on_analyze, [sid, reference, *ai_inputs], [walk, capture_msg, *review_outputs],
                                  concurrency_id="models", concurrency_limit=2)
        cancel.click(None, None, None, cancels=[analyzing])
        redraw.click(on_redraw, [sid], review_outputs, concurrency_id="models", concurrency_limit=2)
        suggest.click(on_suggest, [sid, *sizes.values()], list(sizes.values()))
        # The viewer is filled in a .then() after the step switch: a Model3D that gets its first file while its
        # step is still hidden never mounts its canvas (Gradio 6.28).
        build.click(on_build, [sid, *sizes.values(), values, rejected, *geometry_inputs],
                    [walk, *review_outputs, model_msg, built, views, stats], concurrency_id="cad").then(
            lambda path: path, [built], [model], api_visibility="private")
        gr.on([snap.input, clearance.input, finish.input, finish_mm.release, finish_edges.input], on_geometry,
              [sid, *geometry_inputs], [model_msg, model, views, stats, *sizes.values(), values],
              trigger_mode="always_last", concurrency_id="cad")
        export.click(on_export, [sid, formats, quality, material, nozzle, layer, infill, pattern, perimeters,
                                 supports, brim, scale], [export_msg, download, files, stats], concurrency_id="cad")
    return app


def launch() -> None:
    from dotenv import load_dotenv
    load_dotenv()
    build_app().queue(max_size=20, default_concurrency_limit=1).launch(theme=THEME, css=CSS, max_file_size="20mb")
