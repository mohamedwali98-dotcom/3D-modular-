"""The files /api serves: built parts under the artifact root, by content key. Only names the server itself writes
are accepted, and the resolved path must stay inside the part's folder."""
from __future__ import annotations

import re
from pathlib import Path

from fastapi import HTTPException
from fastapi.responses import FileResponse

JOB_ID = re.compile(r"^[0-9a-f]{32}$")
_KEY = re.compile(r"^[0-9a-f]{20}$")
NAME = re.compile(r"^(?!\.+$)[\w.-]+$")  # a segment of only dots (e.g. "..") is never a real file name


def artifact(key: str, path: str, root: Path) -> FileResponse:
    parts = path.split("/")
    folder = (Path(root) / key).resolve()
    target = (folder / path).resolve() if _KEY.match(key) and all(NAME.match(p) for p in parts) else None
    if target is None or folder not in target.parents or not target.is_file():
        raise HTTPException(404, "File not found or expired.")
    return FileResponse(target)
