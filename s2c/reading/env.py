"""Pick readers by name from SKETCH_READERS (default "trocr,paddle,vlm"); an unavailable reader is skipped.
TrOCR is in the default so a cold or slow VLM never leaves the sketch path with no reader at all."""
from __future__ import annotations

import logging
import os

from s2c import obs
from s2c.reading.base import Reader

log = logging.getLogger(__name__)


def readers_from_env(spec: str | None = None) -> list[Reader]:
    names = [n.strip() for n in (spec or os.environ.get("SKETCH_READERS", "trocr,paddle,vlm")).split(",")]
    out: list[Reader] = []
    for name in filter(None, names):
        try:
            if name == "vlm":
                from s2c.reading.vlm import VlmReader
                from s2c.vision.client import VLMClient
                out.append(VlmReader(VLMClient.from_env()))
            elif name == "paddle":
                from s2c.reading.paddle import PaddleReader
                out.append(PaddleReader())
            elif name == "trocr":
                from s2c.reading.trocr import TrocrReader
                out.append(TrocrReader())
            else:
                log.warning("unknown reader %r skipped", name)
        except Exception as exc:  # noqa: BLE001 - a missing optional dependency must not stop
            obs.fallback(f"reader_{name}_unavailable", exc)
    return out
