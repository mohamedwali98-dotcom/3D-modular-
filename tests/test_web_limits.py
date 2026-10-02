"""Upload limits (audit H1): pixels before decode, and a body without Content-Length."""
import io
import os
import subprocess
import sys

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


def test_only_a_multipart_upload_gets_the_image_sized_limit():
    """FastAPI parses "Application/JSON" and "application/*+json" as JSON too: every body that is not a multipart
    upload is held to the JSON limit, however its type is spelled."""
    body = b'{"request_id": "' + b"0" * (3 * 1024 * 1024) + b'"}'
    for kind in ("Application/JSON", "application/merge-patch+json", "text/plain"):
        assert c.post("/api/merge", content=body, headers={"Content-Type": kind}).status_code == 413, kind


def _python(code: str, cwd=None, **env) -> subprocess.CompletedProcess:
    base = {k: v for k, v in os.environ.items() if k not in ("S2C_MAX_PIXELS", "OPENCV_IO_MAX_IMAGE_PIXELS")}
    return subprocess.run([sys.executable, "-c", code], cwd=cwd, env={**base, **env}, capture_output=True,
                          text=True, timeout=180, check=False)


def test_importing_cv2_before_s2c_is_warned_about():
    """OpenCV reads its pixel cap when cv2 is imported: s2c must come first, and says so when it did not."""
    assert "OPENCV_IO_MAX_IMAGE_PIXELS" in _python("import cv2, s2c").stderr
    assert "OPENCV_IO_MAX_IMAGE_PIXELS" not in _python("import s2c, cv2").stderr


def test_the_cap_is_s2c_max_pixels_from_the_environment_or_dot_env(tmp_path):
    (tmp_path / ".env").write_text("S2C_MAX_PIXELS=1000\n", encoding="utf-8")
    code = "import os, s2c; print(s2c.MAX_PIXELS, os.environ['OPENCV_IO_MAX_IMAGE_PIXELS'])"
    assert _python(code, cwd=tmp_path, OPENCV_IO_MAX_IMAGE_PIXELS="999999999999").stdout.split() == ["1000", "1000"]
