"""The web app server: /api for the React app and the built app at /.
Run: uv run uvicorn s2c.web.server:app --port 8000"""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from s2c.web.api import install_error_handlers
from s2c.web.api import router as api_router

load_dotenv()
DIST = Path(__file__).resolve().parents[2] / "web" / "dist"
ORIGINS = ["http://localhost:5173", "http://127.0.0.1:5173"]
ORIGINS += [o.strip() for o in os.environ.get("ALLOWED_ORIGINS", "").split(",") if o.strip()]

app = FastAPI(title="Sketch-to-CAD")
app.add_middleware(CORSMiddleware, allow_origins=ORIGINS, allow_methods=["*"], allow_headers=["*"])
install_error_handlers(app)
app.include_router(api_router)
if (DIST / "index.html").is_file():
    app.mount("/", StaticFiles(directory=DIST, html=True), name="web")
