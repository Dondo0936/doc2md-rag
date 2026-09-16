"""FastAPI application entrypoint for the Parser Platform Module."""

from __future__ import annotations

import logging
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Ensure repo root is on sys.path so `converter`, `claude_wrapper`, etc. import.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from backend.app.api.routes import health, parse  # noqa: E402
from backend.app.config import get_settings  # noqa: E402

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    # Best-effort Docling warmup (non-fatal if deps missing in lightweight envs)
    try:
        import converter

        converter.warmup_converter()
        settings.docling_warmed_up = True
        logger.info("Docling converter warmed up")
    except Exception as exc:  # noqa: BLE001
        settings.docling_warmed_up = False
        logger.warning("Docling warmup skipped: %s", exc)
    yield


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(
        title="Doc2MD Parser API",
        version="0.1.0",
        description="Structured document parser — blocks as source of truth, markdown as export.",
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins or ["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(health.router)
    app.include_router(parse.router)
    return app


app = create_app()
