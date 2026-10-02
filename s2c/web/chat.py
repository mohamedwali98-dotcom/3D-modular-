"""The Describe chat: one OpenAI-compatible chat model asks for what is missing and names the part; our code keeps
only the numbers the user wrote (rule 2) and builds the part with describe.spec_from_request. The model never
writes code or geometry. Provider only from env: CHAT_* falling back to VLM_*."""
from __future__ import annotations

import hashlib
import hmac
import json
import os
import re
import secrets
import time
from collections.abc import Callable
from dataclasses import dataclass

from pydantic import ValidationError

from s2c.logdir import log_file
from s2c.web.describe import REQUIRED, PartRequest, check, spec_from_request

MAX_MESSAGES = 20
MAX_CHARS = 2000
REPHRASE = "Sorry, could you rephrase that?"

SYSTEM_PROMPT = """You are the design assistant of Sketch-to-CAD. You help the user describe ONE simple mechanical part \
so our own software can build it. You never write code, CAD scripts or geometry.

Scope: only Sketch-to-CAD and designing a part in the grammar below. For anything else, reply in one polite sentence \
that you only help design parts here, with "part": null.

Grammar (all sizes in millimetres):
- plate: width_mm, height_mm, thickness_mm. Optional holes on the front face: a_mm from the left edge, b_mm from \
the bottom edge, diameter_mm.
- l_bracket: leg_a_mm (horizontal leg), leg_b_mm (vertical leg), width_mm, thickness_mm.
- spacer: outer_diameter_mm, inner_diameter_mm, length_mm.
- flange: outer_diameter_mm, inner_diameter_mm, thickness_mm, bolt_circle_diameter_mm, bolt_hole_diameter_mm, \
bolt_count (a count, not mm).
Any other shape: say it cannot be described here and suggest sketching it on the Capture screen.

Rules:
- Never invent, estimate or assume a size. Use only numbers the user wrote. If a size is missing, ask for it.
- Ask for at most two missing things at a time, briefly.
- When a choice helps (part type, hole count, a common size), offer 2 to 4 short options.
- Answer in the user's language.

Reply with one JSON object only, no other text:
{"reply": "<your message>", "options": ["<short option>", ...], \
"part": null or {"type": "plate|l_bracket|spacer|flange", "values": {"<key>": <number>}, \
"holes": [{"a_mm": <number>, "b_mm": <number>, "diameter_mm": <number>}]}}
"part" must be filled (not null) as soon as the part type is known, with every size the user gave so far in "values", repeated on every turn. Example: the user says "a plate 60 by 40, 5 thick" -> {"reply": "Got it. Any holes?", "options": ["No holes", "Two holes", "Four holes"], "part": {"type": "plate", "values": {"width_mm": 60, "height_mm": 40, "thickness_mm": 5}, "holes": []}}"""

_PROCESS_KEY = secrets.token_bytes(32)  # the server's alone: never derived from anything a client holds
_NUM = re.compile(r"\d+(?:\.\d+)?")
_COMMA_DECIMAL = re.compile(r"(\d+),(\d+)")


class ChatUnavailable(Exception):
    """The provider failed; the route answers 502 without the exception text.
    `reason` is a short safe slug: "model_not_found", "auth" or "error"."""

    def __init__(self, reason: str = "error"):
        super().__init__(reason)
        self.reason = reason


def _reason(exc: Exception) -> str:
    status = getattr(exc, "status_code", None)
    if status == 404:
        return "model_not_found"
    if status in (401, 403):
        return "auth"
    return "error"


@dataclass
class ChatTransport:
    model: str
    provider: str
    send: Callable[[list[dict]], tuple[str, dict]]  # messages -> (text, usage)


def get_chat_transport() -> ChatTransport | None:
    """A FastAPI dependency; tests override it. None when no provider is configured."""
    base_url = os.environ.get("CHAT_BASE_URL") or os.environ.get("VLM_BASE_URL") or ""
    model = os.environ.get("CHAT_MODEL") or os.environ.get("VLM_MODEL") or ""
    api_key = os.environ.get("CHAT_API_KEY") or os.environ.get("VLM_API_KEY") or ""
    # A local Ollama needs no key; in Docker the entrypoint rewrites localhost to host.docker.internal
    local = base_url.startswith(("http://localhost", "http://127.0.0.1", "http://host.docker.internal"))
    if not (base_url and model and (api_key or local)):
        return None
    from openai import OpenAI

    client = OpenAI(base_url=base_url, api_key=api_key or "local", timeout=90.0, max_retries=0)

    def send(messages: list[dict]) -> tuple[str, dict]:
        resp = client.chat.completions.create(model=model, messages=messages, temperature=0.2,
                                              response_format={"type": "json_object"})
        usage = getattr(resp, "usage", None)
        return resp.choices[0].message.content or "", {
            "prompt_tokens": getattr(usage, "prompt_tokens", None),
            "completion_tokens": getattr(usage, "completion_tokens", None)}

    return ChatTransport(model=model, provider=base_url, send=send)


def _log(record: dict) -> None:
    path = log_file("chat.jsonl")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
    except OSError:
        pass


def _call(transport: ChatTransport, messages: list[dict]) -> str:
    t0 = time.perf_counter()
    record = {"ts": time.time(), "provider": transport.provider, "model": transport.model}
    try:
        text, usage = transport.send(messages)
    except Exception as exc:
        _log({**record, "latency_ms": round((time.perf_counter() - t0) * 1000), "status": "error",
              "error_type": type(exc).__name__})
        raise ChatUnavailable(_reason(exc)) from exc
    _log({**record, "latency_ms": round((time.perf_counter() - t0) * 1000), "status": "ok",
          "chars_out": len(text), **(usage or {})})
    return text


def _parse(text: str) -> dict | None:
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`").removeprefix("json").strip()
    try:
        data = json.loads(text)
    except ValueError:
        start, end = text.find("{"), text.rfind("}")
        try:
            data = json.loads(text[start:end + 1]) if 0 <= start < end else None
        except ValueError:
            data = None
    if not isinstance(data, dict) or not isinstance(data.get("reply"), str):
        return None
    return data


def user_numbers(messages: list[dict]) -> set[float]:
    """Every number the user typed. "2,5" counts as both 2.5 and the pair 2 and 5."""
    nums: set[float] = set()
    for m in messages:
        if m["role"] != "user":
            continue
        nums.update(float(n) for n in _NUM.findall(m["content"]))
        nums.update(float(f"{a}.{b}") for a, b in _COMMA_DECIMAL.findall(m["content"]))
    return nums


def _written(value: float, nums: set[float]) -> bool:
    return any(abs(value - n) < 1e-9 for n in nums)


def filter_part(raw: object, nums: set[float]) -> tuple[PartRequest | None, list[str]]:
    """Keep only known keys whose number the user wrote; everything else goes to `missing`."""
    if not isinstance(raw, dict):
        return None, []
    try:
        req = PartRequest.model_validate(raw)
    except ValidationError:
        return None, []
    values = {k: v for k, v in req.values.items() if k in REQUIRED[req.type] and _written(v, nums)}
    holes, dropped = [], []
    if req.type == "plate":
        for i, h in enumerate(req.holes):
            if all(_written(x, nums) for x in (h.a_mm, h.b_mm, h.diameter_mm)):
                holes.append(h)
            else:
                dropped.append(f"holes[{i}]")
    part = PartRequest(type=req.type, values=values, holes=holes)
    return part, list(dict.fromkeys(check(part) + dropped))


def sign(prompt: str, reply: str) -> str:
    """The server's mark on a reply it wrote to the user turn `prompt`. The client sends the conversation back
    every turn, so an assistant turn it made up (to steer the model off its rules), or one of ours moved after
    another user turn, is told apart from ours. A restart starts conversations over."""
    return hmac.new(_PROCESS_KEY, json.dumps([prompt, reply]).encode(), hashlib.sha256).hexdigest()


def verified(turns: list[tuple[str, str, str | None]]) -> bool:
    """Every assistant turn (role, content, sig) carries our signature for the user turn just before it."""
    prompt = None
    for role, content, sig in turns:
        if role == "user":
            prompt = content
        elif prompt is None or sig is None or not hmac.compare_digest(sig.encode(), sign(prompt, content).encode()):
            return False
    return True


def _prompt(messages: list[dict]) -> str:
    return next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")


def run_chat(messages: list[dict], transport: ChatTransport) -> dict:
    convo = [{"role": "system", "content": SYSTEM_PROMPT}]
    convo += [{"role": m["role"], "content": m["content"]} for m in messages]
    data = None
    for _ in range(2):  # one retry on invalid JSON
        data = _parse(_call(transport, convo))
        if data is not None:
            break
    if data is None:
        return {"reply": REPHRASE, "sig": sign(_prompt(messages), REPHRASE), "options": [], "part": None, "missing": [], "spec": None,
                "model": transport.model}
    options = [str(o)[:80] for o in data.get("options") or [] if isinstance(o, (str, int, float))][:4]
    part, missing = filter_part(data.get("part"), user_numbers(messages))
    spec = spec_from_request(part) if part is not None and not missing else None
    return {"reply": data["reply"], "sig": sign(_prompt(messages), data["reply"]), "options": options,
            "part": part.model_dump() if part is not None else None, "missing": missing,
            "spec": spec.model_dump(mode="json") if spec is not None else None, "model": transport.model}
