"""The web app server: /api for the React app and the built app at /.
Run: uv run uvicorn s2c.web.server:app --port 8000"""
from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from s2c import obs
from s2c.web.api import get_pipeline, install_error_handlers
from s2c.web.api import router as api_router
from s2c.web.guard import access_token, install_guards

load_dotenv()
obs.configure_logging()
if access_token() is None:
    logging.getLogger(__name__).warning("S2C_ACCESS_TOKEN is not set: /api answers this computer only")
DIST = Path(__file__).resolve().parents[2] / "web" / "dist"
ORIGINS = ["http://localhost:5173", "http://127.0.0.1:5173"]
ORIGINS += [o.strip() for o in os.environ.get("ALLOWED_ORIGINS", "").split(",") if o.strip()]



@asynccontextmanager
async def lifespan(app: FastAPI):
    get_pipeline()  # the models load at startup, not on the first request
    yield


app = FastAPI(title="Sketch-to-CAD", lifespan=lifespan)
install_error_handlers(app)
install_guards(app)
# Added last, so it wraps the others: the guard's 401 and the body limit's 413 carry CORS headers too.
app.add_middleware(CORSMiddleware, allow_origins=ORIGINS, allow_methods=["*"], allow_headers=["*"])
app.include_router(api_router)
if (DIST / "index.html").is_file():
    app.mount("/", StaticFiles(directory=DIST, html=True), name="web")
