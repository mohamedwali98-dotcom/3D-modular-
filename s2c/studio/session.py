"""Per-browser-session state, kept on the server and keyed by an id: gr.State only holds the id, because the
pipeline's Observed (images, masks, meshes) is too big to deep-copy on every event. Spec 2026-09-23-studio §8.
Gradio's State.delete_callback only fires for a state value that was written back as some event's output
(gradio.state_holder.SessionState.__setitem__ is what starts its TTL clock); the session id here is never an
output, only an input, so that callback would never run. Idle sessions are swept lazily instead, from get()."""
from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass, field

from s2c.multiview.artifacts import ExportResult, Part
from s2c.multiview.pipeline import Observed
from s2c.multiview.settings import AiSettings, GeometrySettings
from s2c.multiview.spec import MultiViewSpec

SWEEP_EVERY_S = 60


@dataclass
class Item:
    id: str
    path: str
    name: str
    face: str = "auto"
    kind: str = "auto"
    sheet_id: str | None = None  # the drawing sheet this view was cut from
    view: int | None = None      # ... and its index among the part drawing's views
    hand_face: bool = False      # the user picked the face: the projection switch never renames it
    mm_per_px: float | None = None  # the sheet's scale from its dimensions, in the crop's pixels
    numbers: bool = True         # False: a sheet's view, its numbers read on the sheet; the crop is never read again
    line_art: bool = False       # True: a sketch's view, drawn again in lines from its strokes


@dataclass
class Session:
    id: str
    items: list[Item] = field(default_factory=list)
    reference: str = "none"
    ai: AiSettings = field(default_factory=AiSettings)
    geometry: GeometrySettings = field(default_factory=GeometrySettings)
    observed: Observed | None = None
    spec: MultiViewSpec | None = None
    edits: dict[str, float] = field(default_factory=dict)
    rejected: tuple[str, ...] = ()
    row_paths: list[str] = field(default_factory=list)
    shown: dict[str, float | None] = field(default_factory=dict)
    part: Part | None = None
    exported: ExportResult | None = None
    projection: str = "auto"
    sheets: dict[str, tuple] = field(default_factory=dict)  # sheet id -> (image path, Sheet, Naming)
    sheet_notes: list[str] = field(default_factory=list)
    touched: float = field(default_factory=time.time)


class SessionStore:
    def __init__(self, ttl_s: float = 3600):
        self.ttl_s, self._items, self._lock = ttl_s, {}, threading.Lock()
        self._last_sweep = 0.0

    def new(self) -> str:
        sid = uuid.uuid4().hex
        with self._lock:
            self._sweep(force=True)
            self._items[sid] = Session(sid)
        return sid

    def get(self, sid: str | None) -> Session:
        """The session, or a fresh one under that id (after a server restart the browser still holds its id)."""
        sid = sid or uuid.uuid4().hex
        with self._lock:
            self._sweep()
            session = self._items.get(sid)
            if session is None:
                session = self._items[sid] = Session(sid)
            session.touched = time.time()
            return session

    def ids(self) -> list[str]:
        """The sessions still alive: their files must outlive the file sweep."""
        with self._lock:
            return list(self._items)

    def drop(self, sid: str | None) -> None:
        with self._lock:
            self._items.pop(sid or "", None)

    def _sweep(self, force: bool = False) -> None:
        now = time.time()
        if not force and now - self._last_sweep < SWEEP_EVERY_S:
            return
        self._last_sweep = now
        for sid in [s for s, v in self._items.items() if now - v.touched > self.ttl_s]:
            del self._items[sid]
