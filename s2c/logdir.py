"""Where the call logs go: S2C_LOG_DIR (default logs/), read when a writer is made, so tests and deployments can
keep them out of the source tree."""
from __future__ import annotations

import os
from pathlib import Path


def log_file(name: str) -> Path:
    return Path(os.environ.get("S2C_LOG_DIR") or "logs") / name
