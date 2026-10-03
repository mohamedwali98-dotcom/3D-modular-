# tests/test_web_api.py
import json
import sys
import time
from pathlib import Path

from fastapi.testclient import TestClient

from s2c.multiview.pipeline import MvPipeline
from s2c.sketch.models import Dimension, Size, SketchReading, View
from s2c.sketch.models import Reading as SketchDimReading
from s2c.web.api import get_pipeline
from s2c.web.server import app

SK = Path(__file__).resolve().parents[1] / "examples" / "mv" / "sketches"
app.dependency_overrides[get_pipeline] = lambda: MvPipeline()
c = TestClient(app, client=("127.0.0.1", 50000), raise_server_exceptions=False)


def _fake_sheet_reading() -> SketchReading:
    front = View(name="front", label_text="FRONT", bbox_px=(0, 0, 200, 100), size_mm=(50.0, 25.0))
    top = View(name="top", label_text="TOP", bbox_px=(0, 120, 200, 80), size_mm=(50.0, 20.0))
    width = Dimension(id="w", view="front", kind="linear", value=50.0, text_raw="50",
                      readings=[SketchDimReading(reader="qwen", text="50", confidence=0.9)], measures=[], axis="a",
                      badge="written", evidence="reader", bbox_px=(0, 0, 10, 10))
    return SketchReading(image_size_px=(200, 200), views=[front, top], entities=[], dimensions=[width],
                         features=[], envelope={a: Size(value=1.0, badge="derived", evidence="geometry")
                                                for a in "xyz"}, issues=[], timings_ms={})


def analyze(values=True):
    files = [("files", ("front.png", (SK / "front.png").read_bytes(), "image/png")),
             ("files", ("top.png", (SK / "top.png").read_bytes(), "image/png"))]
    r = c.post("/api/analyze", files=files, data={"faces": json.dumps(["front", "top"]),
                                                  "kinds": json.dumps(["sketch", "sketch"]), "reference": ""})
    assert r.status_code == 202, r.text
    jid = r.json()["job_id"]
    for _ in range(200):
        job = c.get(f"/api/jobs/{jid}").json()
        if job["status"] != "running":
            return job
        time.sleep(0.05)
    raise AssertionError("job never finished")


def test_status_and_examples():
    assert "providers" in c.get("/api/status").json()
    ex = c.get("/api/examples").json()
    assert {e["face"] for e in ex} == {"front", "top"}
    assert c.get(ex[0]["url"]).status_code == 200
    assert c.get("/api/examples/..%2Fpyproject.toml").status_code == 404


def test_analyze_job_reports_real_stages_and_images():
    job = analyze()
    assert job["status"] == "done", job
    assert [s["key"] for s in job["stages"]] == ["label", "outline", "read", "draw", "fuse"]
    assert job["images"][0]["outline"] and job["images"][0]["width"] > 0
    assert job["coverage"]["front"] == "observed"
    res = job["result"]
    assert res["request_id"] == job["job_id"]
    assert res["spec"] or res["abstain"]


def test_merge_with_sizes_gives_a_spec_then_model_and_export():
    job = analyze()
    m = c.post("/api/merge", json={"request_id": job["job_id"], "accepted": [], "rejected": [],
                                    "user_values": {"envelope.x_mm": 50, "envelope.y_mm": 30, "envelope.z_mm": 20}}).json()
    assert m["spec"], m
    assert m["spec"]["provenance"]["envelope.x_mm"] in ("user_written", "user_edited")
    geo = {"snap": True, "clearance": "medium", "finish": "none", "finish_mm": 1.0, "finish_edges": "all_vertical"}
    model = c.post("/api/model", json={"request_id": job["job_id"], "spec": m["spec"], "geometry": geo}).json()
    assert model["glb_url"] and c.get(model["glb_url"]).status_code == 200
    assert model["volume_cm3"] > 0 and len(model["bbox_mm"]) == 3
    ex = c.post("/api/export", json={"spec": m["spec"], "settings": {"export": {"formats": ["stl", "step"]}}}).json()
    assert set(ex["files"]) >= {"stl", "step"} and ex["files"]["stl"]["size_bytes"] > 0
    assert c.get(ex["files"]["stl"]["url"]).status_code == 200


def test_unknown_job_and_request_are_plain_404s():
    r = c.get("/api/jobs/" + "0" * 32)
    assert r.status_code == 404 and "error" in r.json()
    r = c.post("/api/merge", json={"request_id": "0" * 32, "user_values": {}, "accepted": [], "rejected": []})
    assert r.status_code == 404 and "Traceback" not in r.text


def test_upload_limits():
    files = [("files", ("x.png", b"#!/bin/sh\n", "image/png"))]
    assert c.post("/api/analyze", files=files, data={"faces": "[]", "kinds": "[]"}).status_code == 415
    assert c.post("/api/analyze", files=[], data={"faces": "[]", "kinds": "[]"}).status_code in (400, 422)


def test_artifact_traversal_is_404():
    assert c.get("/api/artifacts/" + "a" * 20 + "/..%2F..%2Fpyproject.toml").status_code == 404


def _flat_abstain(ab):
    assert ab["partial"] is None or all(isinstance(v, (int, float)) for v in ab["partial"].values()), ab
    assert isinstance(ab["missing"], list) and isinstance(ab["suggested"], dict)
    assert all(isinstance(v, (int, float)) for v in ab["suggested"].values())


def test_missing_x_abstain_has_a_flat_partial_and_the_missing_paths():
    job = analyze()
    ab = job["result"]["abstain"]
    assert ab and ab["reason"] == "missing_x"
    _flat_abstain(ab)
    assert "envelope.x_mm" in ab["missing"]
    m = c.post("/api/merge", json={"request_id": job["job_id"], "user_values": {}, "accepted": [], "rejected": []})
    ab = m.json()["abstain"]
    _flat_abstain(ab)
    assert "envelope.x_mm" in ab["missing"]


def test_read_stage_names_the_reader_that_runs():
    from s2c.web import jobs
    job = jobs.new_job(1, MvPipeline(reader=lambda *a, **k: None))
    read = job.stage("read")
    assert (read["tool"], read["ai"], read["state"]) == ("TrOCR", True, "pending")
    job = jobs.new_job(1, MvPipeline(batch_reader=lambda *a, **k: None))
    assert job.stage("read")["tool"] in ("Vision model", __import__("os").environ.get("VLM_MODEL"))
    assert jobs.new_job(1, MvPipeline()).stage("read")["state"] == "skipped"


def _rotated_jpeg() -> bytes:
    """A 400 x 200 JPEG whose EXIF says 'rotate 90 degrees clockwise to display' (orientation 6)."""
    import io

    from PIL import Image, ImageDraw

    im = Image.new("RGB", (400, 200), "white")
    ImageDraw.Draw(im).rectangle([60, 40, 340, 160], outline="black", width=6)
    exif = Image.Exif()
    exif[0x0112] = 6
    buf = io.BytesIO()
    im.save(buf, "JPEG", exif=exif)
    return buf.getvalue()


def test_upright_applies_the_exif_orientation():
    import io

    from PIL import Image

    from s2c.web.api import upright

    out = upright(_rotated_jpeg())
    im = Image.open(io.BytesIO(out))
    assert im.size == (200, 400)
    assert im.getexif().get(0x0112, 1) == 1
    plain = (SK / "front.png").read_bytes()
    assert upright(plain) is plain


def test_a_rotated_phone_jpeg_is_measured_the_way_the_browser_shows_it():
    files = [("files", ("phone.jpg", _rotated_jpeg(), "image/jpeg"))]
    r = c.post("/api/analyze", files=files, data={"faces": json.dumps(["front"]), "kinds": json.dumps(["sketch"])})
    assert r.status_code == 202, r.text
    jid = r.json()["job_id"]
    for _ in range(200):
        job = c.get(f"/api/jobs/{jid}").json()
        if job["status"] != "running":
            break
        time.sleep(0.05)
    img = job["images"][0]
    assert img["height"] == 2 * img["width"] > 0, img  # portrait, as the browser shows it


def _wait(jid):
    for _ in range(400):
        job = c.get(f"/api/jobs/{jid}").json()
        if job["status"] != "running":
            return job
        time.sleep(0.05)
    raise AssertionError("job never finished")


def _two_sketches():
    return [("files", ("front.png", (SK / "front.png").read_bytes(), "image/png")),
            ("files", ("top.png", (SK / "top.png").read_bytes(), "image/png"))]


def test_merge_uses_the_ai_settings_the_job_started_with():
    import numpy as np
    calls = []

    def gen(refs, prompt, seed, stage):
        calls.append(stage)
        return np.full((256, 256, 3), 255, np.uint8)

    app.dependency_overrides[get_pipeline] = lambda: MvPipeline(image_gen=gen)
    try:
        r = c.post("/api/analyze", files=_two_sketches(), data={
            "faces": json.dumps(["front", "top"]), "kinds": json.dumps(["sketch", "sketch"]), "reference": "",
            "ai": json.dumps({"use_qwen_image": False, "use_rescue": False})})
        assert r.status_code == 202, r.text
        job = _wait(r.json()["job_id"])
        before = len(calls)
        m = c.post("/api/merge", json={"request_id": job["job_id"], "accepted": [], "rejected": [],
                                        "user_values": {"envelope.x_mm": 50, "envelope.y_mm": 30, "envelope.z_mm": 20}})
        assert m.status_code == 200 and m.json()["spec"], m.text
        assert len(calls) == before, calls
    finally:
        app.dependency_overrides[get_pipeline] = lambda: MvPipeline()


def test_a_fourth_running_job_is_told_the_server_is_busy():
    import threading

    from s2c.multiview.spec import MvAbstain

    gate = threading.Event()

    class Slow(MvPipeline):
        def observe(self, images, reference=None, progress=None):
            gate.wait(10)
            return MvAbstain(stage="label", reason="face_unknown", remedy="Tell us which face this photo shows.")

    app.dependency_overrides[get_pipeline] = lambda: Slow()
    started = []
    try:
        data = {"faces": json.dumps(["front", "top"]), "kinds": json.dumps(["sketch", "sketch"])}
        for _ in range(3):
            r = c.post("/api/analyze", files=_two_sketches(), data=data)
            assert r.status_code == 202, r.text
            started.append(r.json()["job_id"])
        r = c.post("/api/analyze", files=_two_sketches(), data=data)
        assert r.status_code == 429 and r.json() == {"error": "The server is busy. Try again in a minute."}
    finally:
        gate.set()
        app.dependency_overrides[get_pipeline] = lambda: MvPipeline()
    for jid in started:
        _wait(jid)


def test_the_registry_keeps_at_most_max_jobs_dropping_the_least_used_finished():
    from s2c.web import jobs
    saved = dict(jobs.JOBS)
    jobs.JOBS.clear()
    try:
        first = jobs.new_job(1, MvPipeline())
        first.status = "done"
        for _ in range(jobs.MAX_JOBS + 10):
            jobs.new_job(1, MvPipeline()).status = "done"
        assert len(jobs.JOBS) <= jobs.MAX_JOBS and first.job_id not in jobs.JOBS
    finally:
        jobs.JOBS.clear()
        jobs.JOBS.update(saved)


def test_an_oversize_upload_is_refused_before_the_body_is_read():
    r = c.post("/api/analyze", content=b"x", headers={"Content-Length": str(62 * 1024 * 1024),
                                                     "Content-Type": "multipart/form-data; boundary=x"})
    assert r.status_code == 413 and "error" in r.json()


def test_an_unknown_scale_reference_is_a_400():
    r = c.post("/api/analyze", files=_two_sketches(), data={"faces": "[]", "kinds": "[]", "reference": "moon"})
    assert r.status_code == 400 and r.json() == {"error": "Unknown scale reference."}


def test_http_errors_without_a_plain_detail_get_a_sentence_for_their_status():
    from fastapi import FastAPI, HTTPException

    from s2c.web.api import install_error_handlers

    mini = FastAPI()
    install_error_handlers(mini)

    @mini.get("/api/odd")
    def odd():
        raise HTTPException(400, detail={"field": "x"})

    @mini.get("/api/teapot")
    def teapot():
        raise HTTPException(418)

    m = TestClient(mini, raise_server_exceptions=False)
    r = m.get("/api/odd")
    assert r.status_code == 400 and r.json()["error"] == "The request was not accepted."
    r = m.get("/api/teapot")
    assert r.status_code == 418 and r.json()["error"] != "Not found." and "Teapot" not in r.json()["error"]
    r = m.post("/api/odd")
    assert r.status_code == 405 and r.json()["error"] == "That action is not allowed here."
    assert m.get("/api/nothing").json()["error"] == "Not found."


def test_stage_tools_name_the_configured_model(monkeypatch):
    from s2c.web import jobs
    monkeypatch.setenv("VLM_MODEL", "gemma3:4b")
    job = jobs.new_job(1, MvPipeline(chat=lambda m: "", batch_reader=lambda *a, **k: None))
    assert job.stage("label")["tool"] == "gemma3:4b"
    assert job.stage("read")["tool"] == "gemma3:4b"
    monkeypatch.delenv("VLM_MODEL")
    job = jobs.new_job(1, MvPipeline(chat=lambda m: "", batch_reader=lambda *a, **k: None))
    assert (job.stage("label")["tool"], job.stage("read")["tool"]) == ("Vision model", "Vision model")


def test_an_abstain_carries_the_provenance_of_the_values_it_kept():
    job = analyze()
    m = c.post("/api/merge", json={"request_id": job["job_id"], "user_values": {"envelope.x_mm": 50},
                                    "accepted": [], "rejected": []}).json()
    ab = m["abstain"]
    assert ab and ab["partial"]["envelope.x_mm"] == 50
    assert ab["partial_provenance"] == {"envelope.x_mm": "user_edited"}


def _wait(jid: str) -> dict:
    for _ in range(200):
        job = c.get(f"/api/jobs/{jid}").json()
        if job["status"] != "running":
            return job
        time.sleep(0.05)
    raise AssertionError("job never finished")


def test_analyze_sheet_mode_reads_one_sheet_and_reaches_the_same_fuse_stage(monkeypatch):
    monkeypatch.setattr("s2c.sketch.read_sketch", lambda image_bytes: _fake_sheet_reading(), raising=False)
    r = c.post("/api/analyze", files=[("files", ("sheet.png", (SK / "front.png").read_bytes(), "image/png"))],
              data={"mode": "sheet"})
    assert r.status_code == 202, r.text
    job = _wait(r.json()["job_id"])
    assert [s["key"] for s in job["stages"]] == ["views", "lines", "values", "draw", "fuse"]
    assert job["stages"][0]["detail"].startswith("2 views found by the sketch reader")
    assert job["status"] == "done", job
    assert job["result"]["spec"] or job["result"]["abstain"]


def test_analyze_sheet_mode_rejects_more_than_one_file():
    files = [("files", ("a.png", (SK / "front.png").read_bytes(), "image/png")),
             ("files", ("b.png", (SK / "top.png").read_bytes(), "image/png"))]
    r = c.post("/api/analyze", files=files, data={"mode": "sheet"})
    assert r.status_code == 400


def test_analyze_sheet_mode_without_read_sketch_fails_with_a_clear_message(monkeypatch):
    monkeypatch.setitem(sys.modules, "s2c.sketch", None)  # the sketch package fails to import
    r = c.post("/api/analyze", files=[("files", ("sheet.png", (SK / "front.png").read_bytes(), "image/png"))],
              data={"mode": "sheet"})
    assert r.status_code == 202, r.text
    job = _wait(r.json()["job_id"])
    assert job["status"] == "failed"
    assert "per-face photos" in job["error"]


def test_analyze_sheet_mode_splits_a_drawn_sheet_and_builds(monkeypatch):
    """A clean orthographic sheet is split and named by s2c.multiview.sheet before the sketch reader is tried."""
    def no_sketch(image_bytes):
        raise AssertionError("the sketch reader must not run on a drawn sheet")
    monkeypatch.setattr("s2c.sketch.read_sketch", no_sketch, raising=False)
    sheet = Path(__file__).resolve().parents[1] / "examples" / "mv" / "sheet" / "sheet.png"
    r = c.post("/api/analyze", files=[("files", ("sheet.png", sheet.read_bytes(), "image/png"))],
               data={"mode": "sheet"})
    assert r.status_code == 202, r.text
    job = _wait(r.json()["job_id"])
    assert job["status"] == "done", job
    assert job["stages"][0]["detail"].startswith("3 views found")
    assert job["stages"][0]["tool"] == "Drawing reader"
    assert job["result"]["spec"] or job["result"]["abstain"]
    assert {i["face"] for i in job["images"]} >= {"front", "top"}
    assert job["coverage"]["front"] == job["coverage"]["top"] == "observed"


def test_analyze_sheet_mode_leaves_out_a_view_it_cannot_name(monkeypatch):
    """An isometric view beside the orthographic views is left out with a warning, not asked about."""
    import cv2
    import numpy as np
    monkeypatch.setattr("s2c.sketch.read_sketch", lambda image_bytes: _fake_sheet_reading(), raising=False)
    sheet = Path(__file__).resolve().parents[1] / "examples" / "mv" / "sheet" / "sheet.png"
    img = cv2.imdecode(np.frombuffer(sheet.read_bytes(), np.uint8), cv2.IMREAD_COLOR)
    h, w = img.shape[:2]
    img = cv2.copyMakeBorder(img, 0, 0, 0, w // 2, cv2.BORDER_CONSTANT, value=(255, 255, 255))
    cx, cy, r = w + w // 4, h // 4, min(w, h) // 8
    iso = np.array([[cx, cy - r], [cx + r, cy - r // 2], [cx + r, cy + r // 2], [cx, cy + r],
                    [cx - r, cy + r // 2], [cx - r, cy - r // 2]], np.int32)
    cv2.polylines(img, [iso], True, (0, 0, 0), 2)
    cv2.line(img, (cx, cy), (cx, cy + r), (0, 0, 0), 2)
    _, png = cv2.imencode(".png", img)
    r_ = c.post("/api/analyze", files=[("files", ("sheet.png", png.tobytes(), "image/png"))], data={"mode": "sheet"})
    job = _wait(r_.json()["job_id"])
    assert job["status"] == "done", job
    assert job["stages"][0]["tool"] == "Drawing reader"
    assert all(i["face"] for i in job["images"])
    assert job["result"]["abstain"] is None or job["result"]["abstain"]["reason"] != "face_unknown"

def test_analyze_sheet_mode_surfaces_the_abstain_reason_and_remedy(monkeypatch):
    from s2c.sketch.models import SketchAbstain
    empty = _fake_sheet_reading().model_copy(update={"views": [], "dimensions": [], "abstain": SketchAbstain(
        stage="text", reason="readers_unavailable", remedy="Reading service unavailable. Retry in a minute.")})
    monkeypatch.setattr("s2c.sketch.read_sketch", lambda image_bytes: empty, raising=False)
    r = c.post("/api/analyze", files=[("files", ("sheet.png", (SK / "front.png").read_bytes(), "image/png"))],
              data={"mode": "sheet"})
    job = _wait(r.json()["job_id"])
    assert job["status"] == "failed"
    assert job["error"] == "Reading service unavailable. Retry in a minute."  # the reason slug goes to the log
    assert job["stages"][0]["state"] == "failed"


def test_analyze_sheet_mode_with_no_views_fails_with_a_remedy(monkeypatch):
    empty = _fake_sheet_reading().model_copy(update={"views": [], "dimensions": []})
    monkeypatch.setattr("s2c.sketch.read_sketch", lambda image_bytes: empty, raising=False)
    r = c.post("/api/analyze", files=[("files", ("sheet.png", (SK / "front.png").read_bytes(), "image/png"))],
              data={"mode": "sheet"})
    job = _wait(r.json()["job_id"])
    assert job["status"] == "failed" and "no views" in job["error"].lower()


def test_analyze_sheet_mode_uses_the_projection_switch(monkeypatch):
    """An unlabelled third-angle sheet is named by the switch: the view above the front is the top."""
    import cv2

    from tests.sheet_helpers import draw_sheet
    from tests.test_mv_sheet_name import pick, views
    monkeypatch.setattr("s2c.sketch.read_sketch", lambda image_bytes: _fake_sheet_reading(), raising=False)
    img, _ = draw_sheet(pick(views.__wrapped__(), ("front", "top", "right")), layout="third", labels=False)
    png = cv2.imencode(".png", img)[1].tobytes()
    faces = {}
    for projection in ("first", "third"):
        r = c.post("/api/analyze", files=[("files", ("s.png", png, "image/png"))],
                   data={"mode": "sheet", "projection": projection})
        faces[projection] = {i["face"] for i in _wait(r.json()["job_id"])["images"]}
    assert faces["third"] == {"front", "top", "right"}
    assert faces["first"] != faces["third"]
    assert c.post("/api/analyze", files=[("files", ("s.png", png, "image/png"))],
                  data={"mode": "sheet", "projection": "sideways"}).status_code == 400


def test_a_dimensioned_sheet_builds_with_no_typed_size(monkeypatch):
    """Auto projection, the dimensions read: the analysis ends ready to build, every size measured."""
    import cv2

    from tests.builders import cut_box, solid_block
    from tests.line_views import WordReader, drawing_sheet
    monkeypatch.setattr("s2c.sketch.read_sketch", lambda image_bytes: _fake_sheet_reading(), raising=False)
    img, _, words = drawing_sheet(cut_box(solid_block(), 0, 40, 50, 15, 60, 70), faces=("front", "top", "right"),
                                  layout="third", iso=True)
    app.dependency_overrides[get_pipeline] = lambda: MvPipeline(reader=WordReader(words))
    try:
        r = c.post("/api/analyze", files=[("files", ("s.png", cv2.imencode(".png", img)[1].tobytes(), "image/png"))],
                   data={"mode": "sheet"})
        job = _wait(r.json()["job_id"])
    finally:
        app.dependency_overrides[get_pipeline] = lambda: MvPipeline()
    assert job["status"] == "done", job
    assert {i["face"] for i in job["images"]} == {"front", "top", "right"}
    assert "third-angle" in job["stages"][0]["detail"]
    spec = job["result"]["spec"]
    assert spec and job["result"]["abstain"] is None, job["result"]
    assert spec["provenance"]["envelope.x_mm"] == "measured"
    assert abs(spec["envelope"]["x_mm"] - 80) < 1.5


def test_a_hand_sketch_photo_builds_from_its_written_sizes(monkeypatch):
    """A phone photo of a pen sketch in sheet mode: page, faces, labels, numbers; the envelope is the user's own."""
    import cv2

    from tests.builders import solid_block
    from tests.hand_views import hand_photo
    from tests.line_views import WordReader
    def no_sketch(image_bytes):
        raise AssertionError("the team's sketch reader must not run when the five steps read the sketch")
    monkeypatch.setattr("s2c.sketch.read_sketch", no_sketch, raising=False)
    photo, words = hand_photo(solid_block(), faces=("front", "top", "right"), layout="third")
    app.dependency_overrides[get_pipeline] = lambda: MvPipeline(reader=WordReader(words))
    try:
        r = c.post("/api/analyze", files=[("files", ("p.jpg", cv2.imencode(".jpg", photo)[1].tobytes(), "image/jpeg"))],
                   data={"mode": "sheet", "projection": "third"})
        job = _wait(r.json()["job_id"])
    finally:
        app.dependency_overrides[get_pipeline] = lambda: MvPipeline()
    assert job["status"] == "done", job
    assert {i["face"] for i in job["images"]} == {"front", "top", "right"}
    assert job["stages"][0]["tool"] == "Sketch reader"
    spec = job["result"]["spec"]
    assert spec and spec["provenance"]["envelope.x_mm"] == "user_written", job["result"]
    assert (spec["envelope"]["x_mm"], spec["envelope"]["y_mm"], spec["envelope"]["z_mm"]) == (80, 60, 70)


def test_a_dark_photo_in_sheet_mode_fails_with_the_page_remedy(monkeypatch):
    import cv2
    import numpy as np
    monkeypatch.setattr("s2c.sketch.read_sketch", lambda image_bytes: _fake_sheet_reading(), raising=False)
    dark = cv2.imencode(".png", np.full((900, 1200, 3), 20, np.uint8))[1].tobytes()
    r = c.post("/api/analyze", files=[("files", ("d.png", dark, "image/png"))], data={"mode": "sheet"})
    job = _wait(r.json()["job_id"])
    assert job["status"] == "failed" and job["error"]


def test_the_legacy_mv_routes_are_not_served():
    """/mv had none of /api's guards (size limit, image check, concurrency cap): the server mounts /api only."""
    assert not [p for p in app.openapi()["paths"] if p.startswith("/mv")]
    files = [("files", ("front.png", (SK / "front.png").read_bytes(), "image/png"))]
    assert c.post("/mv/analyze", files=files).status_code in (404, 405)


SHEET = Path(__file__).resolve().parents[1] / "examples" / "mv" / "sheet" / "sheet.png"


def _sheet_job() -> dict:
    r = c.post("/api/analyze", files=[("files", ("sheet.png", SHEET.read_bytes(), "image/png"))],
               data={"mode": "sheet"})
    assert r.status_code == 202, r.text
    return _wait(r.json()["job_id"])


def test_a_library_error_in_a_sheet_job_never_reaches_the_browser(monkeypatch):
    """A CUDA out-of-memory error is a RuntimeError, like the job's own messages once were: its text (paths,
    sizes) must never be shown; the job fails with the fixed sentence."""
    from s2c.web import jobs

    def boom(*a, **k):
        raise RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB in C:/secret/path")
    monkeypatch.setattr("s2c.multiview.sheet_read.observe_drawing", boom)
    job = _sheet_job()
    assert job["status"] == "failed" and job["error"] == jobs.FAILED
    assert "CUDA" not in json.dumps(job)


def test_a_crash_in_the_drawing_reader_fails_the_job_instead_of_switching_readers(monkeypatch):
    """A bug in read_drawing must surface (a failed job, logged), never hand the user another algorithm's result."""
    from s2c.web import jobs
    called = []
    monkeypatch.setattr("s2c.multiview.sheet_read.read_drawing", lambda *a, **k: 1 / 0)
    monkeypatch.setattr(jobs, "_sheet_reading", lambda data: called.append(data))
    job = _sheet_job()
    assert job["status"] == "failed" and job["error"] == jobs.FAILED and not called


def test_without_ai_settings_no_image_goes_to_a_hosted_service():
    """Qwen-Image, its sketch rescue, TripoSR and Solaria send the user's images to hosted services: off unless the
    user turns them on (audit H5)."""
    from s2c.web import jobs
    provided = MvPipeline(image_gen=lambda *a, **k: None, mesh_provider=lambda img: None, depth=lambda img: None)
    app.dependency_overrides[get_pipeline] = lambda: provided  # every helper configured on the server
    try:
        job = analyze()
    finally:
        app.dependency_overrides[get_pipeline] = lambda: MvPipeline()
    pipe = jobs.get_job(job["job_id"]).pipe
    assert not pipe.draw_faces and not pipe.rescue_enabled and pipe.mesh_provider is None and pipe.depth is None


def test_building_the_model_counts_as_using_the_job():
    """A long stay on Model & Export keeps the analysis alive like a status poll does."""
    from s2c.web import jobs
    spec = json.loads((Path(__file__).resolve().parents[1] / "examples" / "mv" / "l_bracket.json").read_text())
    job = jobs.new_job(1, MvPipeline())
    job.status = "done"
    job.used = before = time.time() - 60  # used a minute ago: well inside its hour
    r = c.post("/api/model", json={"request_id": job.job_id, "spec": spec})
    assert r.status_code == 200 and job.used > before


def test_an_edit_to_a_field_that_does_not_exist_is_refused():
    """A path that names no editable value is refused with a plain sentence, never written into a feature or
    silently dropped."""
    job = analyze()
    for path in ("features[0].type", "envelope.w_mm"):
        r = c.post("/api/merge", json={"request_id": job["job_id"], "user_values": {path: 1}})
        assert r.status_code == 400 and path in r.json()["error"]


def test_an_edit_the_part_cannot_hold_is_left_out_with_a_warning():
    """A field the feature does not have, or a feature the part no longer has (features renumber when a rejected
    face changes the pockets), is left out and said so: never written into a feature, never silently dropped, and
    never a refusal that would leave the Review screen stuck on a value it no longer shows."""
    job = analyze()
    sizes = {"envelope.x_mm": 50, "envelope.y_mm": 30, "envelope.z_mm": 20}
    feats = c.post("/api/merge", json={"request_id": job["job_id"], "user_values": sizes}).json()["spec"]["features"]
    assert feats and feats[0]["type"] == "hole", feats
    for path in ("features[0].length_mm", "features[99].a_mm", "features[99].keep"):
        r = c.post("/api/merge", json={"request_id": job["job_id"], "user_values": {**sizes, path: 5}})
        spec = r.json()["spec"]
        assert r.status_code == 200 and spec, (path, r.text)
        assert any(path in w for w in spec["warnings"]) and "length_mm" not in spec["features"][0], path
    r = c.post("/api/merge", json={"request_id": job["job_id"], "user_values": {**sizes, "features[0].diameter_mm": 5}})
    assert r.status_code == 200 and r.json()["spec"]["features"][0]["diameter_mm"] == 5


def test_an_analysis_shows_in_the_metrics():
    analyze()
    text = c.get("/api/metrics").text
    assert 's2c_jobs_total{mode="photos",outcome=' in text and 's2c_stage_seconds_count{stage="outline"}' in text
