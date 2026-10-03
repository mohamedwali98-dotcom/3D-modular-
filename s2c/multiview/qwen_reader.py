"""Qwen-VL reads every handwritten crop of one image in a single call. Spec 2026-09-23 section 4.
The model only transcribes what the user wrote; parse_value and link in ocr.py decide what it means."""
from __future__ import annotations

import base64
import logging

import numpy as np
from pydantic import BaseModel, ConfigDict

from s2c import obs
from s2c.multiview.label import Chat, strip_fences
from s2c.multiview.ocr import BatchReader
from s2c.reading.tiles import tile_grid

log = logging.getLogger(__name__)
READ_CONFIDENCE = 0.9
PROMPT = ("The image holds numbered tiles #1..#n, one crop of handwriting from a mechanical sketch each. Return "
          'exactly what is written in each tile, as JSON only: {"reads": [{"i": 1, "text": "⌀6"}, ...]}, `i` the '
          'tile number. Use ⌀ for a diameter sign and R for a radius. Return "" when a tile is unreadable. '
          "Never guess a value that is not written.")


class _Read(BaseModel):
    model_config = ConfigDict(extra="forbid")
    i: int
    text: str


class _Reads(BaseModel):
    model_config = ConfigDict(extra="forbid")
    reads: list[_Read]


def _png_url(png: bytes) -> str:
    return "data:image/png;base64," + base64.b64encode(png).decode()


def qwen_batch_reader(chat: Chat) -> BatchReader:
    def read(crops: list[np.ndarray]) -> list[tuple[str, float]] | None:
        if not crops:
            return []
        content = [{"type": "text", "text": PROMPT},
                   {"type": "image_url", "image_url": {"url": _png_url(tile_grid(crops))}}]
        messages = [{"role": "user", "content": content}]
        for _ in range(2):
            try:
                raw = chat(messages)
            except Exception as e:  # noqa: BLE001 - a failed provider falls back, never breaks the request
                obs.fallback("qwen_vl", e)
                return None
            try:
                parsed = _Reads.model_validate_json(strip_fences(raw))
                bad = [r.i for r in parsed.reads if not 1 <= r.i <= len(crops)]
                if bad:
                    raise ValueError(f"crop numbers {bad} do not exist; use 1 to {len(crops)}")
            except ValueError as e:  # pydantic's ValidationError is a ValueError
                messages = messages + [{"role": "assistant", "content": raw},
                                       {"role": "user", "content": f"Your previous output failed validation: {e}. "
                                                                   "Return corrected JSON only."}]
                continue
            by_crop = {r.i: r.text for r in parsed.reads}
            return [(by_crop.get(k, ""), READ_CONFIDENCE) for k in range(1, len(crops) + 1)]
        return None

    return read


def warm_chat(chat: Chat) -> None:
    """One tiny text-only request to load the model before the first real read. A failed warm-up only
    means a slower first read, never a broken request."""
    try:
        chat([{"role": "user", "content": "Reply with OK."}])
    except Exception as e:  # noqa: BLE001 - a failed warm-up only means a slower first read
        obs.fallback("warmup_qwen_vl", e)
