"""The VLM reader: transcribes handwriting by tiling crops into one numbered image."""
from __future__ import annotations

import logging

import cv2
import numpy as np
from pydantic import BaseModel, Field, ValidationError

from s2c import obs
from s2c.reading.base import Crop, ReaderResult, read_timeout_s
from s2c.vision.client import VLMClient

log = logging.getLogger(__name__)

SYSTEM = (
    "You transcribe handwriting cropped from an engineering sketch. Each numbered tile is one crop. "
    'Return JSON only: {"reads": [{"i": 1, "text": "...", "confidence": 0.0}]}, one entry per tile. '
    "Copy exactly the characters written. Use Ø for a diameter sign, R for a radius, ° for degrees. "
    "Keep a leading decimal point exactly as written, for example .50. "
    "Return an empty text when a tile is unreadable. "
    "Never estimate, measure or guess a value that is not written."
)


class _Read(BaseModel):
    i: int
    text: str
    confidence: float = Field(default=0.8, ge=0, le=1)


class _Reply(BaseModel):
    reads: list[_Read]


def _json_block(raw: str) -> str:
    start, end = raw.find("{"), raw.rfind("}")
    return raw[start: end + 1] if start >= 0 and end > start else raw


def tile_grid(crops: list[Crop], cols: int = 4, cell_h: int = 80, max_w: int = 320) -> bytes:
    """One image with every crop in a numbered cell; the number sits in a header, not on the ink."""
    header = 26
    tiles = []
    for c in crops:
        h, w = c.image.shape[:2]
        k = cell_h / max(h, 1)
        tile = cv2.resize(c.image, (min(max_w, max(1, round(w * k))), cell_h))
        tiles.append(tile)
    cell_w = max_w + 16
    rows = (len(tiles) + cols - 1) // cols
    grid = np.full((rows * (cell_h + header + 12), cols * cell_w, 3), 255, np.uint8)
    for i, tile in enumerate(tiles):
        r, col = divmod(i, cols)
        y, x = r * (cell_h + header + 12), col * cell_w
        cv2.putText(grid, f"#{i + 1}", (x + 4, y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (200, 0, 0), 2)
        grid[y + header: y + header + cell_h, x + 8: x + 8 + tile.shape[1]] = tile
        cv2.rectangle(grid, (x + 4, y + header - 2), (x + cell_w - 4, y + header + cell_h + 2),
                      (180, 180, 180), 1)
    _, buf = cv2.imencode(".png", grid)
    return buf.tobytes()


class VlmReader:
    name = "vlm"
    calibrated = False
    cache_key = None

    def __init__(self, client: VLMClient, batch: int = 24, timeout_s: float | None = None):
        self.client, self.batch = client, batch
        self.timeout_s = read_timeout_s() if timeout_s is None else timeout_s

    def read(self, crops: list[Crop]) -> list[ReaderResult] | None:
        out: list[ReaderResult] = []
        for start in range(0, len(crops), self.batch):
            part = self._read_batch(crops[start: start + self.batch])
            if part is None:
                return None
            out += part
        return out

    def _read_batch(self, crops: list[Crop]) -> list[ReaderResult] | None:
        n = len(crops)
        image = tile_grid(crops)
        user = f"There are {n} tiles, numbered 1 to {n}."
        reply = None
        for _ in range(2):
            try:
                raw = self.client.complete_json(SYSTEM, user, image, mime="image/png")
            except Exception as exc:  # noqa: BLE001 transport errors must not break the pipeline
                obs.fallback("reader_vlm", exc)
                return None
            try:
                reply = _Reply.model_validate_json(_json_block(raw))
                break
            except ValidationError as exc:
                user += f"\nYour previous reply was invalid ({exc.error_count()} errors). Return JSON only."
        if reply is None:
            return None
        by_i = {r.i: r for r in reply.reads}
        return [ReaderResult(text=by_i[i].text, confidence=by_i[i].confidence) if i in by_i
                else ReaderResult(text="", confidence=0.0) for i in range(1, n + 1)]
