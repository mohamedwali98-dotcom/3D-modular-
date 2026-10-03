"""PP-OCRv5 text recognition. Optional dependency: `uv sync --extra ocr`."""
from __future__ import annotations

import logging
import os

import numpy as np

from s2c import obs
from s2c.reading.base import Crop, ReaderResult

log = logging.getLogger(__name__)


class PaddleReader:
    """PP-OCRv5 text recognition. Optional dependency: `uv sync --extra ocr`.
    Checked against PaddleOCR 3.x (`TextRecognition(...).predict(...)` returning `rec_text`,
    `rec_score`)."""
    name = "paddle"
    calibrated = True
    timeout_s = 60.0

    def __init__(self, model_name: str | None = None):
        from paddleocr import TextRecognition

        name = model_name or os.environ.get("SKETCH_PADDLE_MODEL", "PP-OCRv5_server_rec")
        self._model = TextRecognition(model_name=name)
        self.cache_key = f"paddle:{name}"

    def read(self, crops: list[Crop]) -> list[ReaderResult] | None:
        try:
            out = []
            for c in crops:
                res = next(self._model.predict(input=c.image, batch_size=1))
                data = res.json.get("res", res.json) if hasattr(res, "json") else dict(res)
                out.append(ReaderResult(
                    text=str(data.get("rec_text", "")),
                    confidence=float(np.clip(data.get("rec_score", 0.0), 0, 1))))
            return out
        except Exception as exc:  # noqa: BLE001
            obs.fallback("reader_paddle", exc)
            return None
