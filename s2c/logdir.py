"""Where the call logs go: S2C_LOG_DIR (default <data dir>/logs), read when a writer is made, so tests and
deployments can keep them out of the source tree."""
from __future__ import annotations

import os
from pathlib import Path

from s2c.config import data_path


def log_file(name: str) -> Path:
    return Path(os.environ.get("S2C_LOG_DIR") or data_path("logs")) / name
