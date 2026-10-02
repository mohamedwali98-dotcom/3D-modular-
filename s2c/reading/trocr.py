"""TrOCR handwritten, the one local reader shared by the Studio and the sketch package.
Optional dependency: the ai extra (torch, transformers). The model loads once per process and device,
under a lock, on warm() or on the first read."""
from __future__ import annotations

import logging
import os
import threading

import cv2
import numpy as np

from s2c.reading.base import Crop, ReaderResult

log = logging.getLogger(__name__)
DEFAULT_MODEL = "microsoft/trocr-base-handwritten"
_LOADED: dict[tuple[str, str], tuple] = {}
_LOCK = threading.Lock()
_READ_LOCK = threading.Lock()  # concurrent batches queue instead of thrashing the same model and device
READ_LOCK_S = 120.0  # how long a read waits for the model: one abandoned by a timed-out caller cannot block it forever


def token_confidences(logprobs: np.ndarray, mask: np.ndarray) -> list[float]:
    """exp(mean log-probability) over each row's real tokens. Batched generation pads finished rows with
    tokens at log-probability 0; counting them would push a short read's confidence toward 1."""
    out = []
    for row, keep in zip(np.asarray(logprobs, float), np.asarray(mask).astype(bool)):
        real = row[keep]
        out.append(float(np.clip(np.exp(real.mean()), 0.0, 1.0)) if real.size else 0.0)
    return out


def memory_fraction(budget_gb: float, total_bytes: int) -> float:
    """`budget_gb` of `total_bytes` as a fraction for `torch.cuda.set_per_process_memory_fraction`,
    clamped to (0, 1] so a bad setting never disables or over-claims the budget."""
    if total_bytes <= 0:
        return 1.0
    return min(1.0, max((budget_gb * 1024**3) / total_bytes, 1e-3))


def _is_oom(e: BaseException) -> bool:
    import torch

    return isinstance(e, getattr(torch.cuda, "OutOfMemoryError", ())) or "out of memory" in str(e).lower()


def _load(model_id: str, device: str) -> tuple:
    with _LOCK:
        if (model_id, device) not in _LOADED:
            import torch
            from huggingface_hub import snapshot_download
            from transformers import RobertaTokenizer, TrOCRProcessor, VisionEncoderDecoderModel, ViTImageProcessor
            if device == "cuda":
                idx = torch.cuda.current_device()
                budget_gb = float(os.environ.get("GPU_BUDGET_TROCR_GB", "1.0"))
                total = torch.cuda.get_device_properties(idx).total_memory
                torch.cuda.set_per_process_memory_fraction(memory_fraction(budget_gb, total), idx)
            # transformers 5 does not fetch vocab.json and merges.txt for this repo by itself; take the small files
            local = snapshot_download(model_id, allow_patterns=["*.json", "*.txt"])
            processor = TrOCRProcessor(image_processor=ViTImageProcessor.from_pretrained(local),
                                       tokenizer=RobertaTokenizer.from_pretrained(local))
            # load straight into fp16 on cuda: casting after an fp32 load briefly needs both copies, which
            # can bust a tight GPU_BUDGET_TROCR_GB; CPU stays float32 (no fp16 kernels there)
            dtype = torch.float16 if device == "cuda" else None
            model = VisionEncoderDecoderModel.from_pretrained(model_id, torch_dtype=dtype).to(device).eval()
            _LOADED[(model_id, device)] = (processor, model)
        return _LOADED[(model_id, device)]


def _generate(processor, model, device: str, crops: list[Crop], max_new_tokens: int):
    import torch
    from PIL import Image

    images = [Image.fromarray(cv2.cvtColor(c.image, cv2.COLOR_BGR2RGB)) for c in crops]
    pixels = processor(images=images, return_tensors="pt").pixel_values.to(device, dtype=model.dtype)
    with torch.no_grad():
        out = model.generate(pixels, max_new_tokens=max_new_tokens, num_beams=1, output_scores=True,
                             return_dict_in_generate=True)
    texts = processor.batch_decode(out.sequences, skip_special_tokens=True)
    return out, texts, model.compute_transition_scores(out.sequences, out.scores, normalize_logits=True)


def _run(processor, model, device: str, crops: list[Crop], max_new_tokens: int) -> list[ReaderResult]:
    if not _READ_LOCK.acquire(timeout=READ_LOCK_S):  # the reading service records it as a reader error
        raise TimeoutError(f"TrOCR stayed busy for {READ_LOCK_S:.0f} s")
    try:
        out, texts, scores = _generate(processor, model, device, crops, max_new_tokens)
    finally:
        _READ_LOCK.release()
    generated = out.sequences[:, -scores.shape[1]:]  # the tokens the scores describe
    pad = model.generation_config.pad_token_id
    if pad is None:
        pad = processor.tokenizer.pad_token_id
    mask = (generated != pad).cpu().numpy()
    confidences = token_confidences(scores.float().cpu().numpy(), mask)
    return [ReaderResult(text=t.strip(), confidence=c) for t, c in zip(texts, confidences)]


class TrocrReader:
    name = "trocr"
    calibrated = True

    def __init__(self, model_id: str | None = None, device: str | None = None, timeout_s: float = 60.0,
                 max_new_tokens: int = 12):
        import torch
        import transformers  # noqa: F401 - fail early when the ai extra is missing

        self.model_id = model_id or os.environ.get("TROCR_MODEL", DEFAULT_MODEL)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.timeout_s, self.max_new_tokens = timeout_s, max_new_tokens
        self.cache_key = f"trocr:{self.model_id}"

    def warm(self) -> None:
        _load(self.model_id, self.device)

    def read(self, crops: list[Crop]) -> list[ReaderResult] | None:
        if not crops:
            return []
        device = self.device  # a snapshot: another concurrent read must not change which attempt this one is on
        try:
            processor, model = _load(self.model_id, device)
            return self._batched(processor, model, device, crops)
        except Exception as e:
            if device != "cuda" or not _is_oom(e):
                raise
            log.warning("TrOCR ran out of GPU memory (%s); moving to CPU for the rest of the process", e)
            self.device = "cpu"  # only ever moves towards cpu, for every later read on this reader
            processor, model = _load(self.model_id, "cpu")
            return self._batched(processor, model, "cpu", crops)

    def _batched(self, processor, model, device: str, crops: list[Crop]) -> list[ReaderResult]:
        # a whole sheet's crops in one batch bust GPU_BUDGET_TROCR_GB; TROCR_BATCH at a time stays inside it
        n = max(1, int(os.environ.get("TROCR_BATCH", "8")))
        out: list[ReaderResult] = []
        for i in range(0, len(crops), n):
            out += _run(processor, model, device, crops[i:i + n], self.max_new_tokens)
        return out
