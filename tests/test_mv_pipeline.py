from pathlib import Path

import cv2
import numpy as np
import pytest

from s2c.multiview import pipeline
from s2c.multiview.ocr import Reading
from s2c.multiview.pipeline import ImageInput, MvPipeline
from s2c.multiview.spec import MvAbstain


def sketch(w_px, h_px, circles=()):
    img = np.full((1200, 1600, 3), 255, np.uint8)
    x0, y0 = (1600 - w_px) // 2, (1200 - h_px) // 2
    cv2.rectangle(img, (x0, y0), (x0 + w_px, y0 + h_px), (0, 0, 0), 4)
    for cx, cy, r in circles:
        cv2.circle(img, (x0 + cx, y0 + cy), r, (0, 0, 0), 3)
    return cv2.imencode(".png", img)[1].tobytes()


def fake_reads(per_image):
    calls = iter(per_image)

    def read_values(bgr, outline, reader, batch=None):
        x, y, w, h = outline.bbox
        out = []
        for value, where in next(calls):
            box = (x + w // 2 - 20, y + h + 30, 40, 30) if where == "below" else (x - 80, y + h // 2 - 15, 40, 30)
            out.append(Reading(float(value), "linear", box, 0.9, str(value)))
        return out

    return read_values


def test_front_and_top_sketches_to_a_built_part(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline, "read_values", fake_reads([[(60, "below"), (40, "left")], [(60, "below")]]))
    pipe = MvPipeline(reader=lambda crop: ("", 0.0))
    observed = pipe.observe([ImageInput(sketch(600, 400, circles=[(100, 300, 30)]), "front", "sketch"),
                             ImageInput(sketch(600, 100), "top", "sketch")])
    first = pipe.fuse(observed)
    assert isinstance(first, MvAbstain) and first.reason == "missing_z"
    assert abs(first.partial["suggested"]["envelope.z_mm"] - 10) <= 0.5
    spec = pipe.fuse(observed, {"envelope.z_mm": 10.0})
    assert spec.views.top.source == "observed" and spec.views.right.source == "assumed"
    assert spec.features[0].diameter_mm == pytest.approx(5.5)  # about 57 px -> 5.7 mm -> M5 clearance
    result = pipe.build(spec, tmp_path, observed.masks)
    assert result.stl.exists() and result.step.exists()
    assert result.iou["front"] > 0.85 and result.iou["top"] > 0.85


def test_an_untagged_image_without_a_model_abstains():
    res = MvPipeline().observe([ImageInput(sketch(600, 400))])
    assert isinstance(res, MvAbstain) and res.reason == "face_unknown"


def test_a_file_that_is_not_an_image_abstains():
    res = MvPipeline().observe([ImageInput(b"not an image", "front")])
    assert isinstance(res, MvAbstain) and res.reason == "bad_image"


def test_build_errors_become_abstentions(tmp_path):
    from s2c.multiview.spec import MultiViewSpec
    pipe = MvPipeline()
    observed = pipe.observe([ImageInput(sketch(600, 400), "front", "sketch")])
    spec = pipe.fuse(observed, {"envelope.x_mm": 60, "envelope.y_mm": 40, "envelope.z_mm": 5})
    data = spec.model_dump()
    data["features"] = [{"type": "hole", "face": "front", "a_mm": 70.0, "b_mm": 10.0, "diameter_mm": 6.0}]
    data["provenance"].update({"features[0].a_mm": "user_edited", "features[0].b_mm": "user_edited",
                               "features[0].diameter_mm": "user_edited"})
    res = pipe.build(MultiViewSpec.model_validate(data), tmp_path)
    assert isinstance(res, MvAbstain) and res.stage == "build" and res.reason == "feature_outside_part"


def test_the_batch_reader_feeds_ocr(monkeypatch):
    seen = []

    def fake(bgr, outline, service):
        seen.append([r.name for r in service.readers])
        return []

    def batch(crops):
        return []

    monkeypatch.setattr(pipeline, "read_values", fake)
    observed = MvPipeline(batch_reader=batch).observe([ImageInput(sketch(600, 400), "front", "sketch")])
    assert seen == [["qwen"]]
    assert "OCR unavailable: enter the dimensions by hand" not in observed.warnings


def test_the_pipeline_reads_with_qwen_first_then_trocr():
    pipe = MvPipeline(reader=lambda crop: ("", 0.0), batch_reader=lambda crops: [])
    assert [r.name for r in pipe.reading().readers] == ["qwen", "trocr"]
    assert [r.calibrated for r in pipe.reading().readers] == [False, True]
    assert MvPipeline().reading() is None


def test_the_qwen_cache_key_includes_the_base_url_so_switching_providers_never_reuses_a_stale_read(monkeypatch):
    monkeypatch.setenv("VLM_BASE_URL", "http://localhost:9/v1")
    monkeypatch.setenv("VLM_MODEL", "m")
    pipe = MvPipeline(batch_reader=lambda crops: [])
    qwen = next(r for r in pipe.reading().readers if r.name == "qwen")
    assert qwen.cache_key == "qwen:http://localhost:9/v1:m"


def test_the_qwen_cache_key_is_none_when_the_base_url_is_missing(monkeypatch):
    monkeypatch.delenv("VLM_BASE_URL", raising=False)
    monkeypatch.setenv("VLM_MODEL", "m")
    pipe = MvPipeline(batch_reader=lambda crops: [])
    qwen = next(r for r in pipe.reading().readers if r.name == "qwen")
    assert qwen.cache_key is None


def test_default_pipeline_reads_with_qwen_vl_when_configured(monkeypatch):
    for key, value in {"VLM_BASE_URL": "http://localhost:9/v1", "VLM_MODEL": "m", "VLM_API_KEY": "k"}.items():
        monkeypatch.setenv(key, value)
    assert pipeline.default_pipeline().batch_reader is not None


def test_default_pipeline_warms_trocr_then_qwen_in_one_daemon_thread(monkeypatch):
    import threading

    from s2c.multiview import qwen_reader
    from s2c.reading.trocr import TrocrReader

    for key, value in {"VLM_BASE_URL": "http://localhost:9/v1", "VLM_MODEL": "m", "VLM_API_KEY": "k"}.items():
        monkeypatch.setenv(key, value)
    order = []
    started, release = threading.Event(), threading.Event()

    def slow_warm(self):
        started.set()
        release.wait(timeout=2)
        order.append("trocr")

    monkeypatch.setattr(TrocrReader, "warm", slow_warm)
    monkeypatch.setattr(qwen_reader, "warm_chat", lambda chat: order.append("qwen"))
    pipeline.default_pipeline()
    assert started.wait(timeout=2)
    warmups = [t for t in threading.enumerate() if t.name == "reader-warmup"]
    assert warmups and warmups[0].daemon
    release.set()
    warmups[0].join(timeout=5)
    assert order == ["trocr", "qwen"]  # TrOCR must finish loading before Ollama sees the Qwen warm-up


def test_default_pipeline_does_not_warm_qwen_when_unconfigured(monkeypatch):
    import threading

    from s2c.multiview import qwen_reader
    from s2c.reading.trocr import TrocrReader

    monkeypatch.delenv("VLM_API_KEY", raising=False)
    monkeypatch.delenv("VLM_BASE_URL", raising=False)
    order = []
    monkeypatch.setattr(TrocrReader, "warm", lambda self: order.append("trocr"))
    monkeypatch.setattr(qwen_reader, "warm_chat", lambda chat: order.append("qwen"))
    pipeline.default_pipeline()
    warmups = [t for t in threading.enumerate() if t.name == "reader-warmup"]
    if warmups:
        warmups[0].join(timeout=5)
    assert order == ["trocr"]


def test_two_sketches_of_one_face_merge_into_one_observation(tmp_path):
    pipe = MvPipeline()
    observed = pipe.observe([ImageInput(sketch(600, 400), "front", "sketch"),
                             ImageInput(sketch(606, 404), "front", "sketch")])
    assert len(observed.observations) == 1 and len(observed.images) == 1
    assert any(w.startswith("front: merged 2 photos") for w in observed.warnings)
    spec = pipe.fuse(observed, {"envelope.x_mm": 60, "envelope.y_mm": 40, "envelope.z_mm": 5})
    assert pipe.build(spec, tmp_path, observed.masks).iou["front"] > 0.85


def test_qwen_draws_the_missing_faces(tmp_path):
    from tests.mv_helpers import rect
    from tests.test_mv_qwen_faces import fake_gen, silhouette
    gen = fake_gen(silhouette(rect(60, 10), 60, 10), silhouette(rect(10, 40), 10, 40))
    pipe = MvPipeline(image_gen=gen)
    observed = pipe.observe([ImageInput(sketch(600, 400), "front", "sketch")])
    spec = pipe.fuse(observed, {"envelope.x_mm": 60, "envelope.y_mm": 40, "envelope.z_mm": 10})
    assert observed.filled_by == {"front": "observed", "top": "qwen-image", "right": "qwen-image"}
    assert spec.views.top.source == "inferred" and spec.provenance["views.top.outer"] == "inferred"
    assert pipe.build(spec, tmp_path, observed.masks).iou["front"] > 0.85
    assert len(gen.calls) == 2 and gen.calls[0][0] == 1


def test_solaria_runs_once_per_face_and_its_ratio_becomes_a_depth():
    from tests.test_mv_depth import plate_photo, scene
    calls = []

    def depth(img):
        calls.append(img.shape)
        return scene(0.4)

    data = cv2.imencode(".png", plate_photo())[1].tobytes()
    pipe = MvPipeline(depth=depth)
    observed = pipe.observe([ImageInput(data, "front", "photo"), ImageInput(data, "front", "photo")])
    assert len(calls) == 1
    spec = pipe.fuse(observed, {"envelope.x_mm": 60, "envelope.y_mm": 40, "envelope.z_mm": 10})
    assert sorted(f.depth_mm for f in spec.features if f.depth_mm) == [4.0]


def test_a_failing_depth_provider_only_warns():
    from tests.test_mv_depth import plate_photo

    def down(img):
        raise RuntimeError("Space asleep")

    data = cv2.imencode(".png", plate_photo())[1].tobytes()
    observed = MvPipeline(depth=down).observe([ImageInput(data, "front", "photo")])
    assert "front: depth unavailable" in observed.warnings


def test_depth_failure_in_apply_is_a_warning():
    from tests.test_mv_depth import plate_photo

    def wrong_shape(img):
        return np.zeros((5, 5, 3), np.float32)

    data = cv2.imencode(".png", plate_photo())[1].tobytes()
    observed = MvPipeline(depth=wrong_shape).observe([ImageInput(data, "front", "photo")])
    assert not isinstance(observed, MvAbstain)
    assert "front: depth unavailable" in observed.warnings


def test_default_pipeline_warns_when_unconfigured(monkeypatch, caplog):
    for key in ("QWEN_IMAGE_BACKEND", "QWEN_IMAGE_SPACE", "QWEN_IMAGE_BASE_URL", "QWEN_IMAGE_MODEL",
               "VLM_API_KEY", "SOLARIA_SPACE"):
        monkeypatch.delenv(key, raising=False)
    with caplog.at_level("WARNING"):
        pipeline.default_pipeline()
    assert sum("Qwen-Image" in r.message for r in caplog.records) == 1
    assert sum("Solaria" in r.message for r in caplog.records) == 1


@pytest.mark.parametrize("kind", ["fillet", "chamfer"])
def test_the_bundled_sketch_example_takes_a_one_mm_finish(kind):
    """The demo's own sketches: squared outlines, so the default 1 mm finish builds."""
    from s2c.multiview.build import build
    from s2c.multiview.finish import apply_geometry
    from s2c.multiview.settings import GeometrySettings

    d = Path(__file__).parents[1] / "examples" / "mv" / "sketches"
    pipe = MvPipeline()
    observed = pipe.observe([ImageInput((d / "front.png").read_bytes(), "front", "sketch"),
                             ImageInput((d / "top.png").read_bytes(), "top", "sketch")])
    spec = pipe.fuse(observed, {"envelope.x_mm": 50, "envelope.y_mm": 30, "envelope.z_mm": 20})
    assert len(set(spec.views.front.outer)) <= 8
    finished, _ = apply_geometry(spec, GeometrySettings(finish=kind, finish_mm=1.0))
    assert build(finished).solids().vals()


def test_two_fuses_of_one_analysis_never_run_at_once(monkeypatch):
    """The web job and a merge (or two merges) may fuse the same Observed together: fuse locks it itself."""
    import threading
    import time
    pipe = MvPipeline()
    observed = pipe.observe([ImageInput(sketch(600, 400), "front", "sketch")])
    inside, most = [0], [0]
    real = pipeline.complete

    def slow(*a, **k):
        inside[0] += 1
        most[0] = max(most[0], inside[0])
        time.sleep(0.2)
        inside[0] -= 1
        return real(*a, **k)
    monkeypatch.setattr(pipeline, "complete", slow)
    sizes = {"envelope.x_mm": 60, "envelope.y_mm": 40, "envelope.z_mm": 5}
    threads = [threading.Thread(target=pipe.fuse, args=(observed, sizes)) for _ in range(3)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert most[0] == 1


def test_a_labeling_outage_falls_back_to_the_users_tags_or_stops_with_a_remedy():
    """The vision model timing out never crashes an analysis: a face the user picked is used as given; an untagged
    image stops with a sentence saying what to do."""
    def down(messages):
        raise TimeoutError("Request timed out.")

    pipe = MvPipeline(chat=down)
    tagged = pipe.observe([ImageInput(sketch(600, 400), "front", "sketch")])
    assert not isinstance(tagged, MvAbstain) and tagged.observations[0].face == "front"
    untagged = pipe.observe([ImageInput(sketch(600, 400), None, None)])
    assert isinstance(untagged, MvAbstain) and untagged.reason == "label_unavailable"
