"""The /api perimeter: who may call it, how often, and how many builds run at once (audit C1).

With S2C_ACCESS_TOKEN set, every /api call must send "Authorization: Bearer <token>". Without it the server answers
this computer only (loopback), so a laptop on a shared network is never an open CAD and chat service. The health
check, the example images and the content-keyed part files stay open: <img>, the 3D viewer and download links send
no header. Behind a reverse proxy every caller looks local: set the token there."""
from __future__ import annotations

import hmac
import os
import threading
import time
from collections import deque
from contextlib import contextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from s2c import obs

LOCAL = frozenset({"127.0.0.1", "::1", "localhost"})
OPEN_PATHS = frozenset({"/api/status", "/api/examples"})  # GET (and HEAD) these exact paths...
OPEN_PREFIXES = ("/api/examples/", "/api/artifacts/")      # ...and anything under these
WINDOW_S = 60.0
LIMITS = {("POST", "/api/analyze"): 10, ("POST", "/api/merge"): 60, ("POST", "/api/model"): 30,
          ("POST", "/api/export"): 10, ("POST", "/api/chat"): 20}  # calls per client per minute
BUILD_SLOT_COUNT = 2  # /api/model and /api/export building at once; past this they answer 429
BUILD_SLOTS = threading.BoundedSemaphore(BUILD_SLOT_COUNT)
NO_TOKEN = ("This server only answers on the computer it runs on. Set S2C_ACCESS_TOKEN in .env to use it from "
            "another device.")
BAD_TOKEN = "Enter the access token for this server."
SLOW_DOWN = "Too many requests. Wait a minute and try again."
BUSY = "The server is busy building other parts. Try again in a moment."


def access_token() -> str | None:
    return os.environ.get("S2C_ACCESS_TOKEN") or None


def _open(method: str, path: str) -> bool:
    if method == "OPTIONS":  # a CORS preflight carries no credentials
        return True
    return method in ("GET", "HEAD") and (path in OPEN_PATHS or path.startswith(OPEN_PREFIXES))


def refusal(request: Request) -> str | None:
    """Why this /api call may not go on, or None when it may."""
    if _open(request.method, request.url.path):
        return None
    token = access_token()
    if token is None:
        host = request.client.host if request.client else ""
        return None if host in LOCAL else NO_TOKEN
    sent = request.headers.get("authorization", "")
    return None if hmac.compare_digest(sent.encode(), f"Bearer {token}".encode()) else BAD_TOKEN


class RateLimiter:
    """A sliding one-minute window of calls per (client, route)."""

    def __init__(self, window_s: float = WINDOW_S):
        self.window_s, self._calls, self._lock = window_s, {}, threading.Lock()

    def allow(self, key: tuple, limit: int, now: float | None = None) -> bool:
        now = time.monotonic() if now is None else now
        with self._lock:
            calls = self._calls.setdefault(key, deque())
            while calls and now - calls[0] >= self.window_s:
                calls.popleft()
            if len(calls) >= limit:
                return False
            calls.append(now)
            if len(self._calls) > 10_000:  # forget the clients that went quiet
                self._calls = {k: v for k, v in self._calls.items() if v and now - v[-1] < self.window_s}
            return True

    def reset(self) -> None:
        with self._lock:
            self._calls.clear()


LIMITER = RateLimiter()


@contextmanager
def build_slot():
    """One of BUILD_SLOT_COUNT builds, or 429 at once: a burst of builds never takes every worker thread."""
    if not BUILD_SLOTS.acquire(blocking=False):
        obs.count("s2c_refused_total", reason="busy")
        raise HTTPException(429, BUSY)
    try:
        yield
    finally:
        BUILD_SLOTS.release()


def install_guards(app: FastAPI) -> None:
    @app.middleware("http")
    async def guard(request: Request, call_next):
        path = request.url.path
        if not path.startswith("/api"):
            return await call_next(request)
        why = refusal(request)
        if why is not None:
            obs.count("s2c_refused_total", reason="token")
            return JSONResponse({"error": why}, status_code=401, headers={"WWW-Authenticate": "Bearer"})
        limit = LIMITS.get((request.method, path))
        host = request.client.host if request.client else ""
        if limit is not None and not LIMITER.allow((host, path), limit):
            obs.count("s2c_refused_total", reason="rate")
            return JSONResponse({"error": SLOW_DOWN}, status_code=429, headers={"Retry-After": str(int(WINDOW_S))})
        return await call_next(request)
