"""Health / readiness routes."""

from __future__ import annotations

from fastapi import APIRouter

from backend.app.config import get_settings
from backend.app.models.blocks import HealthResponse

router = APIRouter(tags=["health"])


@router.get("/api/v1/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    settings = get_settings()
    ready = settings.docling_warmed_up
    return HealthResponse(
        status="ok" if ready else "starting",
        docling_ready=ready,
    )


@router.get("/health")
async def health_alias():
    return await health()
