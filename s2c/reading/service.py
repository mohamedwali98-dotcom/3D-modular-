"""Run every reader on the same crops in parallel, each within its time budget, with a crop cache,
a background warm-up and one log line per reader call. Trust decisions belong to the caller."""
from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

import numpy as np

from s2c import obs
from s2c.logdir import log_file
from s2c.reading.base import Crop, Reader, ReaderResult

log = logging.getLogger(__name__)
Status = Literal["ok", "error", "timeout", "none"]
DEFAULT_TIMEOUT_S = 30.0


def crop_key(crop: Crop) -> str:
    image = np.ascontiguousarray(crop.image)
    return hashlib.sha1(image.tobytes() + str(image.shape).encode()).hexdigest()


class CropCache:
    """Thread-safe LRU of (reader cache key, crop fingerprint) -> result."""

    def __init__(self, size: int = 4096):
        self.size = size
        self._items: OrderedDict[tuple[str, str], ReaderResult] = OrderedDict()
        self._lock = threading.Lock()

    def get(self, reader_key: str, key: str) -> ReaderResult | None:
        with self._lock:
            value = self._items.get((reader_key, key))
            if value is not None:
                self._items.move_to_end((reader_key, key))
            return value

    def put(self, reader_key: str, key: str, result: ReaderResult) -> None:
        with self._lock:
            self._items[(reader_key, key)] = result
            self._items.move_to_end((reader_key, key))
            while len(self._items) > self.size:
                self._items.popitem(last=False)

    def clear(self) -> None:
        with self._lock:
            self._items.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._items)


DEFAULT_CACHE = CropCache()


@dataclass
class ReaderRun:
    name: str
    calibrated: bool
    results: list[ReaderResult] | None   # None: the reader failed, returned nothing usable or ran out of time
    status: Status
    latency_ms: int
    cached: int = 0


def _ms(t0: float) -> int:
    return round((time.perf_counter() - t0) * 1000)


class ReadingService:
    def __init__(self, readers: list[Reader], cache: CropCache | None = DEFAULT_CACHE,
                 log_path: str | Path | None = None):
        self.readers = list(readers)
        self.cache = cache
        self.log_path = Path(log_path or os.environ.get("READING_LOG") or log_file("reading.jsonl"))
        self._log_lock = threading.Lock()

    def read(self, crops: list[Crop]) -> list[ReaderRun]:
        """One run per reader, in reader order. The call returns when every reader answered or ran out of time."""
        if not crops:
            return [ReaderRun(r.name, getattr(r, "calibrated", False), [], "ok", 0) for r in self.readers]
        if not self.readers:
            return []
        keys = [crop_key(c) for c in crops]
        pool = ThreadPoolExecutor(max_workers=len(self.readers), thread_name_prefix="reader")
        runs: list[ReaderRun] = []
        t0 = time.perf_counter()
        try:
            futures = [pool.submit(obs.carry(self._run_one), r, crops, keys) for r in self.readers]
            for reader, future in zip(self.readers, futures):
                budget = getattr(reader, "timeout_s", DEFAULT_TIMEOUT_S)
                try:
                    runs.append(future.result(timeout=max(budget - (time.perf_counter() - t0), 0.0)))
                except FutureTimeout:
                    obs.fallback(f"reader_{reader.name}_timeout", TimeoutError(f"ran out of time ({budget:.1f} s)"))
                    runs.append(ReaderRun(reader.name, getattr(reader, "calibrated", False), None, "timeout", _ms(t0)))
        finally:
            pool.shutdown(wait=False, cancel_futures=True)  # a hung reader must not hold the request
        for run in runs:
            self._log(run, len(crops))
        return runs

    def _run_one(self, reader: Reader, crops: list[Crop], keys: list[str]) -> ReaderRun:
        t0 = time.perf_counter()
        calibrated = getattr(reader, "calibrated", False)
        reader_key = getattr(reader, "cache_key", None)
        hits: dict[int, ReaderResult] = {}
        if self.cache is not None and reader_key:
            for i, k in enumerate(keys):
                cached = self.cache.get(reader_key, k)
                if cached is not None:
                    hits[i] = cached
        todo = [i for i in range(len(crops)) if i not in hits]
        fresh: list[ReaderResult] | None = []
        if todo:
            try:
                fresh = reader.read([crops[i] for i in todo])
            except Exception as e:  # noqa: BLE001 - a reader must never break the caller
                obs.fallback(f"reader_{reader.name}", e)
                return ReaderRun(reader.name, calibrated, None, "error", _ms(t0), len(hits))
            if fresh is None or len(fresh) != len(todo):
                if fresh is not None:
                    log.warning("reader %s returned %d results for %d crops", reader.name, len(fresh), len(todo))
                return ReaderRun(reader.name, calibrated, None, "none", _ms(t0), len(hits))
            if self.cache is not None and reader_key:
                for i, result in zip(todo, fresh):
                    if result.text.strip():  # an empty read is not a success worth remembering
                        self.cache.put(reader_key, keys[i], result)
        merged = {**hits, **dict(zip(todo, fresh))}
        return ReaderRun(reader.name, calibrated, [merged[i] for i in range(len(crops))], "ok", _ms(t0), len(hits))

    def warm(self) -> threading.Thread:
        """Load every reader that has a warm() in one background thread, so no request pays for a cold load."""
        def run():
            for reader in self.readers:
                warm = getattr(reader, "warm", None)
                if warm is None:
                    continue
                t0 = time.perf_counter()
                try:
                    warm()
                    log.info("reader %s ready in %.1f s", reader.name, time.perf_counter() - t0)
                except Exception as e:  # noqa: BLE001 - a failed warm-up only means a slower first read
                    obs.fallback(f"warmup_{reader.name}", e)

        thread = threading.Thread(target=run, name="reader-warmup", daemon=True)
        thread.start()
        return thread

    def _log(self, run: ReaderRun, crops: int) -> None:
        record = {"ts": datetime.now(UTC).isoformat(timespec="seconds"), "reader": run.name,
                  "crops": crops, "cached": run.cached, "status": run.status, "latency_ms": run.latency_ms}
        try:
            with self._log_lock:
                self.log_path.parent.mkdir(parents=True, exist_ok=True)
                with self.log_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(record) + "\n")
        except OSError as e:
            log.warning("reading log not written: %s", e)
