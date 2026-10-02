# syntax=docker/dockerfile:1.7

FROM node:20-bookworm-slim AS web-builder

WORKDIR /build/web
COPY web/package.json web/package-lock.json ./
RUN --mount=type=cache,target=/root/.npm npm ci
COPY web/ ./
RUN npm run build


FROM python:3.11-slim-bookworm AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PATH="/app/.venv/bin:$PATH"

# CadQuery/VTK need these shared libraries even when rendering without a desktop.
RUN apt-get update \
    && apt-get install --no-install-recommends -y \
        libgl1 \
        libglib2.0-0 \
        libgomp1 \
        libxext6 \
        libxrender1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.11.21 /uv /uvx /bin/

WORKDIR /app
COPY pyproject.toml uv.lock README.md ./
RUN --mount=type=cache,target=/root/.cache/uv uv sync --frozen --no-dev --no-install-project

COPY s2c/ ./s2c/
COPY examples/ ./examples/
COPY profiles/ ./profiles/
COPY --from=web-builder /build/web/dist ./web/dist
COPY --chmod=755 docker/entrypoint.sh /usr/local/bin/docker-entrypoint

RUN --mount=type=cache,target=/root/.cache/uv uv sync --frozen --no-dev \
    && useradd --create-home --uid 10001 appuser \
    && mkdir -p /app/tmp/mv_gradio /app/logs \
    && chown -R appuser:appuser /app/tmp /app/logs

USER appuser

EXPOSE 8000
ENTRYPOINT ["docker-entrypoint"]
# One worker: analysis jobs live in this process's memory (s2c/web/jobs.py); a second worker would not find them.
CMD ["uvicorn", "s2c.web.server:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]

HEALTHCHECK --interval=10s --timeout=5s --start-period=60s --retries=5 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/status', timeout=4)"]

