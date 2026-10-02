"""The /api surface the React app calls. The contract lives in docs/superpowers/plans/2026-09-27-web-app-plan.md
("The API contract"). Every error leaves as {"error": "<plain sentence>"}; no exception text reaches the browser."""
from __future__ import annotations

import io
import json
import logging
import os
import re
import secrets
from functools import lru_cache
from http import HTTPStatus
from pathlib import Path
from typing import Annotated, Literal

import cv2
from fastapi import APIRouter, Depends, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from PIL import Image, ImageOps
from pydantic import BaseModel
from starlette.exceptions import HTTPException as StarletteHTTPException

from s2c import MAX_PIXELS
from s2c.multiview.artifacts import ROOT as ARTIFACT_ROOT
from s2c.multiview.artifacts import build_part, bundle, export_part, sweep
from s2c.multiview.pipeline import IOU_GREEN, ImageInput, MvPipeline, default_pipeline
from s2c.multiview.reference import REFERENCES
from s2c.multiview.settings import AiSettings, GeometrySettings, StudioSettings
from s2c.multiview.spec import MultiViewSpec, MvAbstain
from s2c.silhouette import iou
from s2c.web import chat, files, jobs
from s2c.web.guard import build_slot

log = logging.getLogger(__name__)
EXAMPLES = Path(__file__).resolve().parents[2] / "examples" / "mv" / "sketches"
MAX_BYTES = 10 * 1024 * 1024
MAX_FILES = 6
MAX_BODY = MAX_FILES * MAX_BYTES + 1024 * 1024  # six full images plus the form fields
MAX_JSON = 2 * 1024 * 1024  # a spec, settings or a chat: far under this
TOO_LARGE_JSON = "The request is too large."
TOO_MANY_PIXELS = f"An image is over {MAX_PIXELS // 1_000_000} megapixels. Use a smaller photo."
MAGIC = (b"\xff\xd8\xff", b"\x89PNG\r\n\x1a\n")
_EXAMPLE = re.compile(r"^[\w.-]+\.png$")
UNKNOWN = "Unknown or expired analysis. Analyze again."
BUSY = "The server is busy. Try again in a minute."
SENTENCES = {400: "The request was not accepted.", 404: "Not found.", 405: "That action is not allowed here.",
             413: "The upload is too large. Send at most six images of 10 MB.", 415: "That file is not an image.",
             422: "The request is not valid. Check the values and try again.", 429: BUSY}


@lru_cache(maxsize=1)
def get_pipeline() -> MvPipeline:
    return default_pipeline()


def _sweep() -> None:
    jobs.sweep_jobs(jobs.TTL_S)
    sweep(ARTIFACT_ROOT)


Pipe = Annotated[MvPipeline, Depends(get_pipeline)]
router = APIRouter(prefix="/api", tags=["web"], dependencies=[Depends(_sweep)])


def _slicer() -> bool:
    try:
        from s2c.multiview.slice import find_slicer
        return find_slicer() is not None
    except Exception:  # noqa: BLE001 - a status probe must never fail the request
        return False


@router.get("/status")
def status(pipe: Pipe) -> dict:
    return {"providers": {
        "vision": pipe.chat is not None,
        "reader": pipe.reader is not None or pipe.batch_reader is not None,
        "qwen_image": pipe.image_gen is not None,
        "triposr": pipe.mesh_provider is not None,
        "solaria": pipe.depth is not None,
        "slicer": _slicer(),
        "blender": bool(os.environ.get("BLENDER_PATH") or os.environ.get("BLENDER_PYTHON")),
    }, "ttl_s": jobs.TTL_S}


def _examples() -> list[dict]:
    try:
        items = json.loads((EXAMPLES / "examples.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    return [{"name": e["file"], "url": f"/api/examples/{e['file']}", "face": e["face"], "kind": e["kind"]}
            for e in items if _EXAMPLE.match(e.get("file", "")) and (EXAMPLES / e["file"]).is_file()]


@router.get("/examples")
def examples() -> list[dict]:
    return _examples()


@router.get("/examples/{file}")
def example_file(file: str) -> FileResponse:
    if not _EXAMPLE.match(file) or file not in {e["name"] for e in _examples()}:
        raise HTTPException(404, "Example not found.")
    return FileResponse(EXAMPLES / file, media_type="image/png")


def upright(data: bytes) -> bytes:
    """Phone JPEGs are stored sideways with an EXIF orientation that browsers apply. Re-encode those upright
    (as PNG) so every stage, the vision model included, sees the pixels the user sees."""
    try:
        with Image.open(io.BytesIO(data)) as im:
            if im.getexif().get(0x0112, 1) in (None, 1):
                return data
            fixed = ImageOps.exif_transpose(im)
            if fixed.mode not in ("RGB", "L"):
                fixed = fixed.convert("RGB")
            buf = io.BytesIO()
            fixed.save(buf, "PNG")
            return buf.getvalue()
    except Exception:  # noqa: BLE001 - an unreadable header leaves the bytes as they are
        return data


def _pixels(data: bytes) -> int:
    """The image's pixel count from its header alone (Pillow opens lazily): nothing is decoded."""
    try:
        with Image.open(io.BytesIO(data)) as im:
            return im.width * im.height
    except Image.DecompressionBombError:
        return MAX_PIXELS + 1
    except Exception:  # noqa: BLE001 - an unreadable header is refused by the decode later
        return 0


def _json_list(text: str | None, what: str) -> list:
    try:
        value = json.loads(text or "[]")
    except ValueError:
        value = None
    if not isinstance(value, list):
        raise HTTPException(400, f"The {what} list is not valid.")
    return value


def _tag(items: list, i: int) -> str | None:
    value = items[i] if i < len(items) else None
    return None if value in (None, "", "auto") else str(value)


@router.post("/analyze", status_code=202)
def analyze(pipe: Pipe, files: Annotated[list[UploadFile] | None, File()] = None,
            faces: Annotated[str, Form()] = "[]", kinds: Annotated[str, Form()] = "[]",
            reference: Annotated[str | None, Form()] = None, ai: Annotated[str | None, Form()] = None,
            mode: Annotated[str, Form()] = "photos", projection: Annotated[str, Form()] = "auto") -> dict:
    files = files or []
    if mode not in ("photos", "sheet"):
        raise HTTPException(400, "Unknown capture mode.")
    if projection not in ("auto", "first", "third"):
        raise HTTPException(400, "Unknown projection.")
    if (reference or "") not in ("", *REFERENCES):
        raise HTTPException(400, "Unknown scale reference.")
    max_files = 1 if mode == "sheet" else MAX_FILES
    if not 1 <= len(files) <= max_files:
        raise HTTPException(400, "Send one sheet image." if mode == "sheet"
                            else f"Send between 1 and {MAX_FILES} images.")
    datas = []
    for f in files:
        data = f.file.read(MAX_BYTES + 1)
        if len(data) > MAX_BYTES:
            raise HTTPException(413, "An image is larger than 10 MB. Use a smaller photo.")
        if not data.startswith(MAGIC):
            raise HTTPException(415, "A file is not a JPEG or PNG image.")
        if _pixels(data) > MAX_PIXELS:
            raise HTTPException(413, TOO_MANY_PIXELS)
        datas.append(upright(data))
    if ai:
        try:
            settings = AiSettings(**json.loads(ai))
        except (ValueError, TypeError) as e:
            raise HTTPException(400, "The AI settings are not valid.") from e
        if settings.randomize_seed:
            settings = settings.model_copy(update={"seed": secrets.randbelow(2**31)})
        pipe = pipe.configured(settings)
    if mode == "sheet":
        job = jobs.new_sheet_job(pipe)
        if not jobs.start_sheet(job, pipe, ImageInput(datas[0]), projection):
            raise HTTPException(429, BUSY)
        return {"job_id": job.job_id}
    face_tags, kind_tags = _json_list(faces, "faces"), _json_list(kinds, "kinds")
    images = [ImageInput(d, _tag(face_tags, i), _tag(kind_tags, i)) for i, d in enumerate(datas)]
    job = jobs.new_job(len(images), pipe)
    if not jobs.start(job, pipe, images, reference or None):
        raise HTTPException(429, BUSY)
    return {"job_id": job.job_id}


def _job(job_id: str) -> jobs.Job:
    job = jobs.get_job(job_id) if files.JOB_ID.match(job_id) else None
    if job is None:
        raise HTTPException(404, UNKNOWN)
    jobs.touch(job)
    return job


@router.get("/jobs/{job_id}")
def job_status(job_id: str) -> dict:
    return _job(job_id).to_json()


@router.post("/jobs/{job_id}/cancel")
def cancel(job_id: str) -> dict:
    _job(job_id).cancel = True
    return {"ok": True}


class MergeBody(BaseModel):
    request_id: str
    user_values: dict[str, float] = {}
    accepted: list[str] = []
    rejected: list[str] = []


@router.post("/merge")
def merge(body: MergeBody, pipe: Pipe) -> dict:
    job = _job(body.request_id)
    if job.observed is None:
        raise HTTPException(404, UNKNOWN)
    return jobs.merge(job, pipe, body.user_values, body.accepted, body.rejected)


class ModelBody(BaseModel):
    request_id: str | None = None
    spec: MultiViewSpec
    geometry: GeometrySettings = GeometrySettings()


def _empty_model(abstain: MvAbstain) -> dict:
    return {"key": None, "glb_url": None, "volume_cm3": None, "bbox_mm": None, "iou": {}, "iou_mean": None,
            "views": {}, "warnings": [], "abstain": jobs.abstain_json(abstain)}


@router.post("/model")
def model(body: ModelBody) -> dict:
    with build_slot():
        part = build_part(body.spec, body.geometry, ARTIFACT_ROOT)
    if isinstance(part, MvAbstain):
        return _empty_model(part)
    base = f"/api/artifacts/{part.key}"
    views = {}
    for face, mask in part.views.items():
        path = part.folder / f"view_{face}.png"
        if not path.exists():
            cv2.imwrite(str(path), mask)
        views[face] = f"{base}/{path.name}"
    job = jobs.get_job(body.request_id) if body.request_id and files.JOB_ID.match(body.request_id) else None
    masks = job.observed.masks if job is not None and job.observed is not None else {}
    scores = {f: round(iou(part.views[f], m), 3) for f, m in masks.items() if f in part.views}
    warnings = list(dict.fromkeys([*part.spec.warnings, *part.warnings]))
    warnings += [f"Low confidence on {f}, check the dimensions." for f, s in scores.items() if s < IOU_GREEN]
    return {"key": part.key, "glb_url": f"{base}/{part.preview.name}", "volume_cm3": round(part.volume_mm3 / 1000, 2),
            "bbox_mm": [round(v, 3) for v in part.bbox_mm], "iou": scores,
            "iou_mean": round(sum(scores.values()) / len(scores), 3) if scores else None,
            "views": views, "warnings": warnings, "abstain": None}


class ExportBody(BaseModel):
    spec: MultiViewSpec
    settings: StudioSettings = StudioSettings()


@router.post("/export")
def export(body: ExportBody) -> dict:
    s = body.settings
    with build_slot():
        part = build_part(body.spec, s.geometry, ARTIFACT_ROOT)
        if isinstance(part, MvAbstain):
            return {"files": {}, "zip_url": None, "print_time_s": None, "filament_g": None, "warnings": [],
                    "abstain": jobs.abstain_json(part)}
        res = export_part(part, s.export.formats, s.mesh, s.printing)
        zip_path = bundle(part, res, s.model_dump(mode="json"))
    base = f"/api/artifacts/{part.key}"
    files = {f: {"url": f"{base}/{p.relative_to(part.folder).as_posix()}", "name": p.name,
                 "size_bytes": res.sizes.get(f, 0)} for f, p in res.files.items()}
    return {"files": files, "zip_url": f"{base}/{zip_path.name}", "print_time_s": res.print_time_s,
            "filament_g": res.filament_g, "warnings": list(dict.fromkeys([*part.spec.warnings, *res.warnings])),
            "abstain": None}


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str
    sig: str | None = None  # the server's signature on its own reply (chat.sign); a user turn has none


class ChatBody(BaseModel):
    messages: list[ChatMessage]


@router.post("/chat")
def chat_route(body: ChatBody,
               transport: Annotated[chat.ChatTransport | None, Depends(chat.get_chat_transport)]) -> dict:
    if not 1 <= len(body.messages) <= chat.MAX_MESSAGES:
        raise HTTPException(400, f"Send between 1 and {chat.MAX_MESSAGES} messages.")
    if any(len(m.content) > chat.MAX_CHARS for m in body.messages):
        raise HTTPException(400, f"A message is longer than {chat.MAX_CHARS} characters.")
    if not all(chat.signed(m.role, m.content, m.sig) for m in body.messages):
        raise HTTPException(400, "The conversation could not be checked. Start a new one.")
    if transport is None:
        raise HTTPException(503, "The chat model is not configured. Add CHAT_API_KEY to .env.")
    try:
        return chat.run_chat([{"role": m.role, "content": m.content} for m in body.messages], transport)
    except chat.ChatUnavailable as e:
        detail = {
            "model_not_found": f"The chat model '{transport.model}' is not available on this provider. "
                               "Set CHAT_MODEL in .env to a model your key can use.",
            "auth": "The chat provider rejected the API key. Check CHAT_API_KEY in .env.",
        }.get(e.reason, "The chat model did not answer. Try again.")
        raise HTTPException(502, detail) from e


@router.get("/artifacts/{key}/{path:path}")
def artifact(key: str, path: str) -> FileResponse:
    return files.artifact(key, path, ARTIFACT_ROOT)


def _sentence(status: int, detail: object) -> str:
    """Our own detail sentences pass through; anything else (a dict, Starlette's bare 'Not Found') is replaced
    by a plain sentence for the status."""
    try:
        phrase = HTTPStatus(status).phrase
    except ValueError:
        phrase = None
    if isinstance(detail, str) and detail and detail != phrase:
        return detail
    return SENTENCES.get(status, f"Something went wrong ({status}). Try again.")


class BodyLimit:
    """An /api body over MAX_BODY (MAX_JSON for JSON) is refused: at once when its Content-Length says so, and
    otherwise counted as it streams (a chunked upload, or a length that lies) and cut at the limit."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or not scope["path"].startswith("/api"):
            return await self.app(scope, receive, send)
        headers = dict(scope["headers"])
        json_body = headers.get(b"content-type", b"").startswith(b"application/json")
        limit, sentence = (MAX_JSON, TOO_LARGE_JSON) if json_body else (MAX_BODY, SENTENCES[413])
        declared = headers.get(b"content-length", b"")
        if declared.isdigit() and int(declared) > limit:
            return await JSONResponse({"error": sentence}, status_code=413)(scope, receive, send)
        received, cut = 0, False

        async def counted():
            nonlocal received, cut
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > limit:
                    cut = True
                    return {"type": "http.disconnect"}  # the rest is never read
            return message

        async def held(message):
            if not cut:  # the app's own answer to a cut body is replaced by the 413 below
                await send(message)

        try:
            await self.app(scope, counted, held)
        except Exception:
            if not cut:
                raise
        if cut:
            await JSONResponse({"error": sentence}, status_code=413)(scope, receive, send)


def install_error_handlers(app: FastAPI) -> None:
    """/api errors render as {"error": ...}; other paths keep FastAPI's default shape. Bodies are limited
    (BodyLimit)."""

    def api(request: Request) -> bool:
        return request.url.path.startswith("/api")

    app.add_middleware(BodyLimit)

    @app.exception_handler(StarletteHTTPException)
    async def http_error(request: Request, exc: StarletteHTTPException) -> JSONResponse:
        if not api(request):
            return JSONResponse({"detail": exc.detail}, status_code=exc.status_code, headers=exc.headers)
        return JSONResponse({"error": _sentence(exc.status_code, exc.detail)}, status_code=exc.status_code,
                            headers=exc.headers)

    @app.exception_handler(RequestValidationError)
    async def invalid(request: Request, exc: RequestValidationError) -> JSONResponse:
        if not api(request):
            return JSONResponse({"detail": jsonable_encoder(exc.errors())}, status_code=422)
        return JSONResponse({"error": "The request is not valid. Check the values and try again."}, status_code=422)

    @app.exception_handler(Exception)
    async def crash(request: Request, exc: Exception) -> JSONResponse:
        log.exception("unhandled error on %s", request.url.path, exc_info=exc)
        return JSONResponse({"error": "Something went wrong on our side."}, status_code=500)
