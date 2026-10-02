"""The /api perimeter (audit C1): who may call it, how often, and how many builds run at once."""
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from s2c.multiview.pipeline import MvPipeline
from s2c.web import chat
from s2c.web.api import get_pipeline
from s2c.web.server import app

SPEC = json.loads((Path(__file__).parents[1] / "examples" / "mv" / "l_bracket.json").read_text())
LOCAL, REMOTE = ("127.0.0.1", 50000), ("203.0.113.7", 50000)
UNKNOWN_JOB = {"request_id": "0" * 32}


@pytest.fixture(autouse=True)
def _fresh(monkeypatch):
    from s2c.web import guard
    monkeypatch.delenv("S2C_ACCESS_TOKEN", raising=False)
    app.dependency_overrides[get_pipeline] = lambda: MvPipeline()
    app.dependency_overrides[chat.get_chat_transport] = lambda: None
    guard.LIMITER.reset()
    yield
    app.dependency_overrides.pop(chat.get_chat_transport, None)


def client(addr=LOCAL, token=None) -> TestClient:
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    return TestClient(app, client=addr, headers=headers, raise_server_exceptions=False)


def test_without_a_token_only_this_computer_is_served():
    r = client(REMOTE).post("/api/merge", json=UNKNOWN_JOB)
    assert r.status_code == 401 and "S2C_ACCESS_TOKEN" in r.json()["error"]
    assert client(LOCAL).post("/api/merge", json=UNKNOWN_JOB).status_code == 404


def test_with_a_token_every_caller_must_send_it(monkeypatch):
    monkeypatch.setenv("S2C_ACCESS_TOKEN", "s3cret-token")
    assert client(REMOTE, "s3cret-token").post("/api/merge", json=UNKNOWN_JOB).status_code == 404
    assert client(REMOTE).post("/api/merge", json=UNKNOWN_JOB).status_code == 401
    assert client(REMOTE, "wrong").post("/api/merge", json=UNKNOWN_JOB).status_code == 401
    assert client(LOCAL).post("/api/merge", json=UNKNOWN_JOB).status_code == 401  # a local proxy is no proof


def test_status_examples_and_part_files_stay_open():
    """The health check, the example images and the content-keyed part files (loaded by <img>, the 3D viewer and
    download links, which send no header) need no token."""
    remote = client(REMOTE)
    assert remote.get("/api/status").status_code == 200
    assert remote.get("/api/examples").status_code == 200
    assert remote.get("/api/artifacts/" + "0" * 20 + "/part.stl").status_code == 404


def test_a_burst_of_chat_calls_is_slowed_down():
    from s2c.web import guard
    local = client()
    body = {"messages": [{"role": "user", "content": "a plate"}]}
    codes = [local.post("/api/chat", json=body).status_code for _ in range(guard.LIMITS[("POST", "/api/chat")] + 1)]
    assert 429 not in codes[:-1] and codes[-1] == 429


def test_builds_past_the_slots_are_turned_away():
    from s2c.web import guard
    taken = [guard.BUILD_SLOTS.acquire(blocking=False) for _ in range(guard.BUILD_SLOT_COUNT)]
    try:
        assert all(taken)
        assert client().post("/api/model", json={"spec": SPEC}).status_code == 429
    finally:
        for _ in taken:
            guard.BUILD_SLOTS.release()
    assert client().post("/api/model", json={"spec": SPEC}).status_code == 200


def test_a_json_body_over_the_limit_is_refused():
    body = b'{"spec": "' + b"x" * (3 * 1024 * 1024) + b'"}'
    r = client().post("/api/model", content=body, headers={"Content-Type": "application/json"})
    assert r.status_code == 413
