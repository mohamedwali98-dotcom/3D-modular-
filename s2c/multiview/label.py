"""Vision model labels each image: which face it shows and the rough hole positions. Spec 6.1.
The model returns JSON only. It never sets the envelope, never returns code, and no millimetre it gives reaches geometry."""
from __future__ import annotations

import base64
import json
import logging
import os
import time
from collections.abc import Callable
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from s2c.logdir import log_file
from s2c.multiview.spec import MvAbstain

log = logging.getLogger(__name__)
Chat = Callable[[list[dict]], str]
CHAT_TIMEOUT_S = 60


class LabelHole(BaseModel):
    model_config = ConfigDict(extra="forbid")
    u: float = Field(ge=0, le=1)
    v: float = Field(ge=0, le=1)
    blind: bool = False


class MvLabel(BaseModel):
    model_config = ConfigDict(extra="forbid")
    face: Literal["front", "back", "left", "right", "top", "bottom", "unknown"]
    input_kind: Literal["sketch", "photo", "drawing"]
    holes: list[LabelHole] = []
    description: str = ""
    estimates: dict[str, float] = {}  # ignored on read; kept only so older model replies still validate
    confidence: float = Field(ge=0, le=1)


SYSTEM_PROMPT = """You label one image of a mechanical part for a CAD tool. Reply with JSON only, no prose, matching this schema:
{schema}
Rules:
- face: the side of the part the image shows (front, back, left, right, top, bottom), or unknown.
- input_kind: sketch (hand drawn), photo (real part) or drawing (clean printed drawing).
- holes: every round hole, as u, v fractions of the part's bounding box (u to the right, v upward).
- Never estimate any size in real-world units -- not the overall width, height, depth, nor any hole depth. Never output code."""


def prompt_schema() -> dict:
    """The label schema the model is shown: no estimates and no blind flags, because neither may reach
    geometry (rule 2, P0-3). The model still validates replies that carry them."""
    schema = MvLabel.model_json_schema()
    schema["properties"].pop("estimates", None)
    schema.get("$defs", {}).get("LabelHole", {}).get("properties", {}).pop("blind", None)
    return schema


def strip_fences(text: str) -> str:
    t = text.strip()
    if t.startswith("```"):
        t = t.split("\n", 1)[1] if "\n" in t else ""
        t = t.rsplit("```", 1)[0]
    return t.strip()


def label_image(image_bytes: bytes, chat: Chat, face_hint: str | None = None,
                kind_hint: str | None = None) -> MvLabel | MvAbstain:
    b64 = base64.b64encode(image_bytes).decode()
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT.format(schema=json.dumps(prompt_schema()))},
        {"role": "user", "content": [
            {"type": "text", "text": "Label this image."},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}]},
    ]
    label = None
    for _ in range(2):
        raw = chat(messages)
        try:
            label = MvLabel.model_validate_json(strip_fences(raw))
            break
        except (ValidationError, ValueError) as e:
            messages += [{"role": "assistant", "content": raw},
                         {"role": "user", "content": f"Your previous output failed validation: {e}. "
                                                     "Return corrected JSON only."}]
    if label is None:
        return MvAbstain(stage="label", reason="label_invalid",
                         remedy="The model could not describe this photo. Try a cleaner photo.")
    if label.estimates:
        log.warning("discarded model estimates %s: millimetre values are never model-estimated (rule 2)",
                   sorted(label.estimates))
    label = label.model_copy(update={
        "estimates": {}, "face": face_hint or label.face, "input_kind": kind_hint or label.input_kind})
    if label.face == "unknown":
        return MvAbstain(stage="label", reason="face_unknown", remedy="Tell us which face this photo shows.")
    return label


def hint_label(face: str, kind: str = "sketch") -> MvLabel:
    """A label from the user's tags alone, when no vision model is configured."""
    return MvLabel(face=face, input_kind=kind, confidence=0.9)


def env_chat(log_path: str | Path | None = None, stage: str = "mv_label") -> Chat | None:
    """OpenAI-compatible chat from VLM_BASE_URL, VLM_MODEL, VLM_API_KEY; None when not configured.
    Swap for the integrator's VLMClient when s2c/vision/ lands."""
    base, model, key = (os.environ.get(k) for k in ("VLM_BASE_URL", "VLM_MODEL", "VLM_API_KEY"))
    if not (base and model and key):
        return None
    from openai import OpenAI
    # The library default is 600 s with 2 retries; a hung provider must not hold an image for half an hour
    client, path = OpenAI(base_url=base, api_key=key, timeout=CHAT_TIMEOUT_S, max_retries=1), Path(log_path or log_file("vlm.jsonl"))

    def chat(messages: list[dict]) -> str:
        t0 = time.perf_counter()
        r = client.chat.completions.create(model=model, messages=messages, temperature=0)
        usage = r.usage
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"provider": base, "model": model, "stage": stage,
                                "latency_ms": round((time.perf_counter() - t0) * 1000),
                                "prompt_tokens": getattr(usage, "prompt_tokens", None),
                                "completion_tokens": getattr(usage, "completion_tokens", None)}) + "\n")
        return r.choices[0].message.content or ""

    return chat
