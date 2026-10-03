"""Every setting the app reads (audit M2), declared once with its kind and what it does. The server checks them at
startup and names each bad value; .env.example documents each one (tests/test_config.py keeps the code, this
registry and .env.example in step). Working folders live under S2C_DATA_DIR."""
from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent


@dataclass(frozen=True)
class Var:
    kind: str  # str, int, float, bool, path, url or choice
    help: str
    choices: tuple[str, ...] = ()


def _choice(help: str, *choices: str) -> Var:
    return Var("choice", help, choices)


VARIABLES: dict[str, Var] = {
    # the server
    "S2C_ACCESS_TOKEN": Var("str", "Bearer token for /api from other devices; unset: this computer only"),
    "ALLOWED_ORIGINS": Var("str", "extra web origins allowed by CORS, comma-separated"),
    "S2C_MAX_PIXELS": Var("int", "largest image accepted, in pixels"),
    "OPENCV_IO_MAX_IMAGE_PIXELS": Var("int", "set by the app from S2C_MAX_PIXELS; leave unset"),
    "S2C_DATA_DIR": Var("path", "folder for working files (built parts, logs); default: the project folder"),
    "S2C_LOG_DIR": Var("path", "folder for the call logs; default: <data dir>/logs"),
    "S2C_LOG_FORMAT": _choice("log lines as JSON or plain text", "json", "text"),
    "S2C_LOG_LEVEL": _choice("level of the app's own log lines", "DEBUG", "INFO", "WARNING", "WARN", "ERROR", "CRITICAL"),
    # the vision model and the Describe chat
    "VLM_BASE_URL": Var("url", "OpenAI-compatible endpoint of the vision model"),
    "VLM_MODEL": Var("str", "vision model name"),
    "VLM_API_KEY": Var("str", "vision model key"),
    "CHAT_BASE_URL": Var("url", "endpoint of the Describe chat model; default: VLM_BASE_URL"),
    "CHAT_MODEL": Var("str", "Describe chat model; default: VLM_MODEL"),
    "CHAT_API_KEY": Var("str", "Describe chat key; default: VLM_API_KEY"),
    # hosted models
    "QWEN_IMAGE_BACKEND": _choice("where Qwen-Image runs", "space", "dashscope"),
    "QWEN_IMAGE_SPACE": Var("str", "Hugging Face Space of Qwen-Image"),
    "QWEN_IMAGE_BASE_URL": Var("url", "DashScope endpoint of Qwen-Image"),
    "QWEN_IMAGE_MODEL": Var("str", "DashScope Qwen-Image model"),
    "SOLARIA_SPACE": Var("str", "Hugging Face Space of Solaria depth"),
    "TRIPOSR_SPACE": Var("str", "Hugging Face Space of TripoSR"),
    "TRIPOSR_DIR": Var("path", "local TripoSR checkout (scripts/setup_triposr.ps1)"),
    "HF_TOKEN": Var("str", "Hugging Face token; raises the ZeroGPU quota"),
    # handwriting reading
    "READ_TIMEOUT_S": Var("float", "time budget of one hosted read, in seconds"),
    "READING_LOG": Var("path", "per-call reading log"),
    "TROCR_MODEL": Var("str", "local TrOCR model"),
    "TROCR_BATCH": Var("int", "crops per TrOCR batch"),
    "GPU_BUDGET_TROCR_GB": Var("float", "TrOCR's share of the GPU, in GB"),
    "SKETCH_READERS": Var("str", "readers of the sketch path, comma-separated: trocr, paddle, vlm"),
    "SKETCH_PADDLE_MODEL": Var("str", "PaddleOCR recognition model"),
    "SKETCH_PADDLE_DET": Var("str", "PaddleOCR detection model"),
    "SKETCH_TEXT_DETECTOR": _choice("text detector of the sketch path", "auto", "paddle", "classical"),
    "SKETCH_DEBUG_DIR": Var("path", "folder for the sketch path's debug images"),
    # outside tools
    "BLENDER_PATH": Var("path", "Blender executable, for the rendered preview"),
    "BLENDER_PYTHON": Var("path", "Python of Blender's bpy module"),
    "SLICER_PATH": Var("path", "PrusaSlicer console executable, for G-code"),
    "SLICER_PROFILE": Var("path", "PrusaSlicer profile"),
}

def _bool(value: str) -> None:
    if value.lower() not in ("1", "0", "true", "false", "yes", "no"):
        raise ValueError


def _url(value: str) -> None:
    parsed = urlparse(value)
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        raise ValueError


_KINDS = {"int": ("a whole number", int), "float": ("a number", float),
          "bool": ("one of 1, 0, true, false, yes, no", _bool), "url": ("an http(s) URL", _url)}


def problems(env: Mapping[str, str] | None = None) -> list[str]:
    """One sentence per declared variable whose value is not of its kind. Unset and empty values are fine."""
    env = os.environ if env is None else env
    found = []
    for name, var in VARIABLES.items():
        value = (env.get(name) or "").strip()
        if not value:
            continue
        if var.kind == "choice":
            if value.lower() not in {c.lower() for c in var.choices}:
                found.append(f"{name}={value!r} must be one of {', '.join(var.choices)}")
            continue
        if var.kind not in _KINDS:
            continue
        expected, check = _KINDS[var.kind]
        try:
            check(value)
        except ValueError:
            found.append(f"{name}={value!r} must be {expected}")
    return found


def dotenv_path() -> Path | None:
    """The one .env of the app: the first found going up from this package. The search never starts in the working
    directory, so a value read at import (S2C_DATA_DIR, S2C_MAX_PIXELS) and one read later agree wherever the app
    is started from."""
    for folder in (HERE, *HERE.parents):
        if (folder / ".env").is_file():
            return folder / ".env"
        if folder == PROJECT and (PROJECT / "pyproject.toml").is_file():
            return None  # a checkout without its own .env never takes a parent folder's
    return None


def load_env() -> None:
    """That .env into the environment, for every entry point (values already set win)."""
    path = dotenv_path()
    if path is not None:
        from dotenv import load_dotenv
        load_dotenv(path)


def setting(name: str, default: str) -> str:
    """A value from the environment, else from the app's .env: for what is read at import, before load_env."""
    if name in os.environ:
        return os.environ[name]
    path = dotenv_path()
    if path is None:
        return default
    try:
        from dotenv import dotenv_values
        return dotenv_values(path).get(name) or default
    except Exception:  # noqa: BLE001 - an unreadable .env leaves the default
        return default


def data_dir() -> Path:
    """S2C_DATA_DIR; else the project folder when running from a checkout; else the working directory."""
    chosen = setting("S2C_DATA_DIR", "")
    if chosen:
        return Path(chosen)
    return PROJECT if (PROJECT / "pyproject.toml").is_file() else Path.cwd()


def data_path(*parts: str) -> Path:
    return data_dir().joinpath(*parts)
