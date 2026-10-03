"""Logs and metrics (audit M8, M9).

Logging is configured once by the server or the Studio: JSON lines on stderr (S2C_LOG_FORMAT=text for plain
lines), our own loggers at S2C_LOG_LEVEL (INFO), everything else at WARNING. A record logged while an analysis
runs carries its `job_id`. Counters and timings live in this process and render in the Prometheus text format
(/api/metrics). Every provider fallback goes through `fallback`, so none is silent."""
from __future__ import annotations

import contextvars
import functools
import json
import logging
import os
import sys
import threading
from contextlib import contextmanager

log = logging.getLogger(__name__)
_job_id: contextvars.ContextVar[str | None] = contextvars.ContextVar("job_id", default=None)
_HANDLER = "s2c"
_lock = threading.Lock()
_counts: dict[tuple[str, tuple], float] = {}
_sums: dict[tuple[str, tuple], list] = {}


class _JobFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.job_id = _job_id.get()
        return True


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        out = {"time": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"), "level": record.levelname,
               "logger": record.name, "message": record.getMessage()}
        if getattr(record, "job_id", None):
            out["job_id"] = record.job_id
        if record.exc_info:
            out["exc"] = self.formatException(record.exc_info)
        return json.dumps(out, ensure_ascii=False)


def configure_logging(stream=None, force: bool = False) -> logging.Handler:
    """Our handler on the root logger, once (again with `force`). Uvicorn keeps its own loggers and handlers."""
    root = logging.getLogger()
    existing = next((h for h in root.handlers if h.get_name() == _HANDLER), None)
    if existing is not None and not force:
        return existing
    if existing is not None:
        root.removeHandler(existing)
    handler = logging.StreamHandler(stream or sys.stderr)
    handler.set_name(_HANDLER)
    handler.addFilter(_JobFilter())
    if os.environ.get("S2C_LOG_FORMAT", "json").lower() == "text":
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s [%(job_id)s] %(message)s"))
    else:
        handler.setFormatter(JsonFormatter())
    root.addHandler(handler)
    level = os.environ.get("S2C_LOG_LEVEL", "INFO").strip().upper()
    # an unknown level is named by the startup check (s2c/config.py), which runs after this
    logging.getLogger("s2c").setLevel(level if level in logging.getLevelNamesMapping() else "INFO")
    return handler


@contextmanager
def job_scope(job_id: str | None):
    """Records logged inside carry this analysis's id (None: no analysis)."""
    token = _job_id.set(job_id)
    try:
        yield
    finally:
        _job_id.reset(token)


def carry(fn):
    """fn, for another thread, run inside a copy of this thread's context: its records keep the analysis id. One
    copy per call, since a context runs on one thread at a time."""
    return functools.partial(contextvars.copy_context().run, fn)


def _key(labels: dict) -> tuple:
    return tuple(sorted((k, str(v)) for k, v in labels.items()))


def count(metric: str, /, **labels) -> None:
    with _lock:
        key = (metric, _key(labels))
        _counts[key] = _counts.get(key, 0) + 1


def time_spent(metric: str, seconds: float, /, **labels) -> None:
    with _lock:
        total = _sums.setdefault((metric, _key(labels)), [0.0, 0])
        total[0] += seconds
        total[1] += 1


def fallback(name: str, exc: BaseException) -> None:
    """A provider failed and the app went on without it: logged and counted, never silent."""
    log.warning("fallback %s: %s: %s", name, type(exc).__name__, exc,
                exc_info=exc if log.isEnabledFor(logging.DEBUG) else None)
    count("s2c_fallbacks_total", name=name)


def _labels(key: tuple) -> str:
    return "{" + ",".join(f'{k}="{v}"' for k, v in key) + "}" if key else ""


def _number(value: float) -> str:
    return str(int(value)) if float(value).is_integer() else repr(float(value))


def render() -> str:
    """Every counter and timing, in the Prometheus text format: each family's # TYPE line, then its series."""
    lines, typed = [], set()

    def family(name: str, kind: str) -> None:
        if name not in typed:
            typed.add(name)
            lines.append(f"# TYPE {name} {kind}")

    with _lock:
        for (name, key), v in sorted(_counts.items()):
            family(name, "counter")
            lines.append(f"{name}{_labels(key)} {_number(v)}")
        for (name, key), (total, n) in sorted(_sums.items()):
            family(name, "summary")
            lines += [f"{name}_sum{_labels(key)} {_number(total)}", f"{name}_count{_labels(key)} {n}"]
    return "\n".join(lines) + "\n"


def reset() -> None:
    with _lock:
        _counts.clear()
        _sums.clear()
