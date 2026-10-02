import json
import threading
import time

import pytest

from s2c.vision.client import VLMClient


def test_complete_json_sends_image_and_returns_text(tmp_path):
    seen = {}

    def fake_chat(messages):
        seen["messages"] = messages
        return '{"ok": true}'

    client = VLMClient(chat=fake_chat, log_path=tmp_path / "vlm.jsonl", model="fake-model")
    out = client.complete_json("SYS", "USER", b"\xff\xd8bytes")
    assert out == '{"ok": true}'
    assert seen["messages"][0] == {"role": "system", "content": "SYS"}
    parts = seen["messages"][1]["content"]
    assert parts[0] == {"type": "text", "text": "USER"}
    assert parts[1]["type"] == "image_url"
    assert parts[1]["image_url"]["url"].startswith("data:image/jpeg;base64,")


def test_every_call_is_logged(tmp_path):
    client = VLMClient(chat=lambda m: "{}", log_path=tmp_path / "vlm.jsonl", model="fake-model")
    client.complete_json("s", "u", b"x")
    client.complete_json("s", "u", b"x")
    lines = (tmp_path / "vlm.jsonl").read_text().splitlines()
    assert len(lines) == 2
    rec = json.loads(lines[0])
    assert rec["model"] == "fake-model"
    assert "latency_ms" in rec


def test_call_survives_when_log_write_fails(tmp_path):
    # Point the log path at a location that cannot be created: a file used as
    # a path segment, so mkdir(parents=True) on its "parent" fails on Windows
    # and POSIX alike.
    blocker = tmp_path / "blocker"
    blocker.write_text("not a directory")
    unwritable_log_path = blocker / "nested" / "vlm.jsonl"

    client = VLMClient(chat=lambda m: "{}", log_path=unwritable_log_path, model="fake-model")
    out = client.complete_json("s", "u", b"x")
    assert out == "{}"
    assert not unwritable_log_path.exists()


def test_usage_state_is_not_a_shared_class_attribute():
    # A plain attribute assignment always writes to an instance's own __dict__
    # regardless of whether the class defines an attribute of the same name, so
    # that alone can't tell a per-instance dict apart from a shared class-level
    # default. Assert directly that the class carries no such attribute, so a
    # future edit that reintroduces one (e.g. `_last_usage: dict = {}` on the
    # class body) fails this test immediately.
    assert "_last_usage" not in vars(VLMClient)
    client = VLMClient(chat=lambda m: "{}", model="model-a")
    assert isinstance(client._usage_local, threading.local)


def test_call_without_usage_does_not_inherit_previous_calls_usage(tmp_path):
    # A later call that records no usage must not pick up a previous call's
    # numbers: the recorded value has to be cleared after each read.
    calls = {"n": 0}

    def chat(messages):
        calls["n"] += 1
        if calls["n"] == 1:
            client.record_usage({"prompt_tokens": 999, "completion_tokens": 999})
        return "{}"

    client = VLMClient(chat=chat, log_path=tmp_path / "vlm.jsonl", model="fake-model")
    client.complete_json("s", "u", b"x")
    client.complete_json("s", "u", b"x")

    lines = [json.loads(line) for line in (tmp_path / "vlm.jsonl").read_text().splitlines()]
    assert lines[0]["prompt_tokens"] == 999
    assert "prompt_tokens" not in lines[1]
    assert "completion_tokens" not in lines[1]


def test_concurrent_calls_do_not_cross_contaminate_usage(tmp_path):
    # Two overlapping calls on ONE client, from two threads, each recording
    # different token counts. Both threads record their usage right after the
    # barrier releases them, then "first" sleeps far longer than "second"
    # needs to record its own usage and return. complete_json only reads
    # usage after chat() returns, so with a plain shared instance attribute
    # "second"'s write lands, and stays, in the gap between "first"'s write
    # and "first"'s own read: whichever thread wrote last at the start is the
    # value BOTH calls end up reading, since neither writes again afterwards.
    # That means one of the two logged lines is always wrong under a shared
    # attribute, deterministically, regardless of which thread happens to
    # write first in that opening instant.
    log_path = tmp_path / "vlm.jsonl"
    start_barrier = threading.Barrier(2)
    usage_by_call = {
        "first": {"prompt_tokens": 111, "completion_tokens": 11},
        "second": {"prompt_tokens": 222, "completion_tokens": 22},
    }
    delay_after_record = {"first": 0.3, "second": 0.05}

    def chat(messages):
        call_name = messages[1]["content"][0]["text"]
        start_barrier.wait()
        client.record_usage(usage_by_call[call_name])
        time.sleep(delay_after_record[call_name])
        return f"out-{call_name}"

    client = VLMClient(chat=chat, log_path=log_path, model="fake-model")
    results = {}

    def run(call_name):
        results[call_name] = client.complete_json("s", call_name, b"x")

    t1 = threading.Thread(target=run, args=("first",))
    t2 = threading.Thread(target=run, args=("second",))
    t1.start()
    t2.start()
    t1.join(timeout=5)
    t2.join(timeout=5)

    assert results["first"] == "out-first"
    assert results["second"] == "out-second"

    lines = [json.loads(line) for line in log_path.read_text().splitlines()]
    assert len(lines) == 2
    by_completion_tokens = {rec["completion_tokens"]: rec for rec in lines}
    assert by_completion_tokens[11]["prompt_tokens"] == 111
    assert by_completion_tokens[22]["prompt_tokens"] == 222


def test_model_and_base_url_come_from_environment(monkeypatch):
    monkeypatch.setenv("VLM_MODEL", "env-model")
    monkeypatch.setenv("VLM_BASE_URL", "http://env-host/v1")
    monkeypatch.setenv("VLM_API_KEY", "env-key")

    # No model/base_url/chat given: this exercises the real transport builder
    # (_openai_chat), but only builds the OpenAI client object, never sends a
    # request, so this makes no network call.
    client = VLMClient()
    assert client.model == "env-model"
    assert client.base_url == "http://env-host/v1"

    client_from_env = VLMClient.from_env()
    assert client_from_env.model == "env-model"
    assert client_from_env.base_url == "http://env-host/v1"


def test_from_env_reads_environment_at_call_time(monkeypatch):
    monkeypatch.setenv("VLM_MODEL", "model-one")
    monkeypatch.setenv("VLM_BASE_URL", "http://one/v1")
    monkeypatch.setenv("VLM_API_KEY", "key-one")
    client_one = VLMClient.from_env()
    assert client_one.model == "model-one"
    assert client_one.base_url == "http://one/v1"

    monkeypatch.setenv("VLM_MODEL", "model-two")
    monkeypatch.setenv("VLM_BASE_URL", "http://two/v1")
    client_two = VLMClient.from_env()
    assert client_two.model == "model-two"
    assert client_two.base_url == "http://two/v1"


def test_failed_call_is_logged_with_status_and_reraises(tmp_path):
    log_path = tmp_path / "vlm.jsonl"

    def failing_chat(messages):
        raise RuntimeError("SYSTEM_PROMPT_OR_IMAGE_CONTENT_MUST_NEVER_REACH_THE_LOG")

    client = VLMClient(chat=failing_chat, log_path=log_path, model="fake-model")

    with pytest.raises(RuntimeError):
        client.complete_json("s", "u", b"x")

    lines = log_path.read_text().splitlines()
    assert len(lines) == 1
    rec = json.loads(lines[0])
    assert rec["status"] == "error"
    assert rec["error_type"] == "RuntimeError"
    assert "latency_ms" in rec
    assert "SYSTEM_PROMPT_OR_IMAGE_CONTENT_MUST_NEVER_REACH_THE_LOG" not in lines[0]


def test_the_default_log_goes_to_the_log_directory(tmp_path, monkeypatch):
    """Call logs follow S2C_LOG_DIR (read when the client is made), so tests and deployments keep them out of git."""
    monkeypatch.setenv("S2C_LOG_DIR", str(tmp_path / "calls"))
    VLMClient(chat=lambda m: "{}", model="fake-model").complete_json("s", "u", b"x")
    assert (tmp_path / "calls" / "vlm.jsonl").is_file()
