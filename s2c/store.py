"""Temp files with a TTL. Nothing here outlives an hour.

This module sits behind a public HTTP endpoint that serves a file by the
identifier a caller supplies, so `path()`'s identifier validation is the
only thing standing between a caller and reading arbitrary files off the
machine. It must accept only an identifier the store could itself have
generated (see `SAFE`) and reject anything else by returning None -- never
by raising, and never with any signal that would let a caller distinguish a
rejected identifier from a well-formed but unknown one.

Sweeping is called explicitly by the HTTP layer on each request. There is no
background thread or scheduler here on purpose, to keep this module
dependency-free and easy to test.
"""

import re
import tempfile
import time
import uuid
from pathlib import Path

# Exactly what uuid4().hex + a short lowercase suffix looks like. No path
# separators, no "..", no uppercase -- anything outside this shape is
# rejected before it ever touches the filesystem.
#
# `_EXTENSION` is the single source of truth for what a suffix may look
# like. `SAFE` (the full identifier, checked by `path()`) and `_SAFE_SUFFIX`
# (the extension alone, checked by `put()`) are both built from it, so the
# two checks cannot drift apart: anything `put()` accepts is, by
# construction, something `path()` will later accept back. The end anchor
# is `\Z` (true end of string), not `$` (end of string, or just before a
# trailing newline) -- this is a public-facing allowlist and should have no
# latent exceptions to its strictness.
_HEX_ID = r"[a-f0-9]{32}"
_EXTENSION = r"[a-z0-9]{1,5}"
SAFE = re.compile(rf"^{_HEX_ID}\.{_EXTENSION}\Z")
_SAFE_SUFFIX = re.compile(rf"^\.{_EXTENSION}\Z")


class FileStore:
    def __init__(self, root: Path | None = None, ttl_s: int = 3600):
        self.root = Path(root) if root else Path(tempfile.gettempdir()) / "s2c_store"
        # Safe to call repeatedly: several components construct a store
        # independently against the same root.
        self.root.mkdir(parents=True, exist_ok=True)
        self.ttl_s = ttl_s

    def put(self, data: bytes, suffix: str) -> str:
        # Validate the extension at the point of generation rather than
        # trusting the caller. Without this, a caller-supplied suffix that
        # doesn't match what path() will later accept either orphans the
        # file silently, or -- if it contains a separator or is itself an
        # anchored absolute path -- can escape the store root entirely on
        # write. Raise rather than normalise: silently rewriting a caller's
        # extension would hide the caller's bug instead of surfacing it.
        if not _SAFE_SUFFIX.match(suffix):
            raise ValueError(f"invalid file suffix: {suffix!r}")
        file_id = uuid.uuid4().hex + suffix
        (self.root / file_id).write_bytes(data)
        return file_id

    def path(self, file_id: str) -> Path | None:
        if not SAFE.match(file_id):
            return None
        p = self.root / file_id
        return p if p.exists() else None

    def sweep(self) -> int:
        """Delete every stored file older than ttl_s. Returns the count removed.

        Two requests can sweep concurrently, and on Windows a file can be
        briefly locked by another process. This tolerates a file vanishing
        between listing and stat-ing or unlinking it, and tolerates an
        unexpected directory entry showing up inside the store root -- it
        never raises because of either.
        """
        now, removed = time.time(), 0
        try:
            entries = list(self.root.iterdir())
        except OSError:
            return removed
        for p in entries:
            try:
                if p.is_dir():
                    continue
                if now - p.stat().st_mtime > self.ttl_s:
                    p.unlink(missing_ok=True)
                    removed += 1
            except OSError:
                # The entry vanished (or became briefly inaccessible)
                # between listing, stat-ing and unlinking it. Not our file
                # to worry about any more either way.
                continue
        return removed
