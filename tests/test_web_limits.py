"""Upload limits (audit H1): pixels before decode, and a body without Content-Length."""
import io
import os

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

import s2c
from s2c.multiview.pipeline import ImageInput, MvPipeline
from s2c.web import api
from s2c.web.api import get_pipeline
from s2c.web.server import app

app.dependency_overrides[get_pipeline] = lambda: MvPipeline()
c = TestClient(app, client=("127.0.0.1", 50000), raise_server_exceptions=False)


def _big_png() -> bytes:
    buf = io.BytesIO()
    Image.new("L", (8000, 6000)).save(buf, "PNG")  # 48 MP of black: a few kB on disk
    return buf.getvalue()


def test_an_image_with_too_many_pixels_is_refused_before_decoding():
    r = c.post("/api/analyze", files=[("files", ("big.png", _big_png(), "image/png"))],
               data={"faces": '["front"]', "kinds": '["sketch"]'})
    assert r.status_code == 413 and r.json()["error"] == api.TOO_MANY_PIXELS


def test_opencv_never_decodes_past_the_cap():
    assert os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] == str(s2c.MAX_PIXELS)
    with pytest.raises(cv2.error):
        cv2.imdecode(np.frombuffer(_big_png(), np.uint8), cv2.IMREAD_COLOR)


def test_the_pipeline_turns_an_oversized_image_into_its_bad_image_stop():
    """Paths that skip the upload check (the Studio, a script) get the usual stop, not a crash."""
    res = MvPipeline().observe([ImageInput(_big_png(), "front", "sketch")])
    assert res.reason == "bad_image" and res.remedy


def test_a_body_without_content_length_is_cut_at_the_limit(monkeypatch):
    monkeypatch.setattr(api, "MAX_BODY", 1000)
    r = c.post("/api/analyze", content=(b"x" * 500 for _ in range(4)),
               headers={"Content-Type": "multipart/form-data; boundary=zz"})
    assert r.status_code == 413
