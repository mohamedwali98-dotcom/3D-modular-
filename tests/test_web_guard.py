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


def test_the_perimeters_own_refusals_carry_cors_headers():
    """A web app on an allowed origin can read why it was refused (401 from the guard, 413 from the body limit)."""
    origin = {"Origin": "http://localhost:5173"}
    r = client(REMOTE).post("/api/merge", json=UNKNOWN_JOB, headers=origin)
    assert r.status_code == 401 and r.headers.get("access-control-allow-origin") == origin["Origin"]
    big = b'{"spec": "' + b"x" * (3 * 1024 * 1024) + b'"}'
    r = client().post("/api/model", content=big, headers={**origin, "Content-Type": "application/json"})
    assert r.status_code == 413 and r.headers.get("access-control-allow-origin") == origin["Origin"]


def test_only_the_named_routes_are_open():
    """A future route that merely starts like an open one (/api/statusx) is guarded, and HEAD is open only where GET
    is."""
    remote = client(REMOTE)
    assert remote.get("/api/statusx").status_code == 401
    assert remote.get("/api/examplesx").status_code == 401
    assert remote.head("/api/jobs/" + "0" * 32).status_code == 401
    assert remote.head("/api/status").status_code != 401
    examples = remote.get("/api/examples").json()
    assert examples and remote.get(examples[0]["url"]).status_code == 200


def test_every_refusal_is_counted_by_reason():
    from s2c import obs
    from s2c.web import guard
    obs.reset()
    client(REMOTE).post("/api/merge", json=UNKNOWN_JOB)
    body = {"messages": [{"role": "user", "content": "a plate"}]}
    for _ in range(guard.LIMITS[("POST", "/api/chat")] + 1):
        client().post("/api/chat", json=body)
    taken = [guard.BUILD_SLOTS.acquire(blocking=False) for _ in range(guard.BUILD_SLOT_COUNT)]
    try:
        client().post("/api/model", json={"spec": SPEC})
    finally:
        for _ in taken:
            guard.BUILD_SLOTS.release()
    client().post("/api/model", content=b'{"spec": "' + b"x" * (3 * 1024 * 1024) + b'"}',
                  headers={"Content-Type": "application/json"})
    text = obs.render()
    for reason in ("token", "rate", "busy", "body"):
        assert f's2c_refused_total{{reason="{reason}"}} 1' in text, reason


def test_a_call_relayed_by_a_local_proxy_counts_as_the_device_it_came_from():
    """The Vite dev server (or nginx) on this computer relays a phone's call: without a token the phone is refused,
    as if it had called directly; this computer's own calls through the proxy still pass."""
    relayed = client(LOCAL).post("/api/merge", json=UNKNOWN_JOB, headers={"X-Forwarded-For": "192.168.1.23"})
    assert relayed.status_code == 401
    own = client(LOCAL).post("/api/merge", json=UNKNOWN_JOB, headers={"X-Forwarded-For": "127.0.0.1"})
    assert own.status_code == 404
    spoofed = client(LOCAL).post("/api/merge", json=UNKNOWN_JOB, headers={"X-Forwarded-For": "127.0.0.1, 192.168.1.23"})
    assert spoofed.status_code == 401  # the proxy appends the real peer last: an earlier "127.0.0.1" proves nothing


def test_a_refusal_says_whether_a_token_would_help(monkeypatch):
    assert client(REMOTE).post("/api/merge", json=UNKNOWN_JOB).json()["reason"] == "no_token"
    monkeypatch.setenv("S2C_ACCESS_TOKEN", "s3cret-token")
    assert client(REMOTE, "wrong").post("/api/merge", json=UNKNOWN_JOB).json()["reason"] == "bad_token"
