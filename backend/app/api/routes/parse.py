"""Parse API routes."""

from __future__ import annotations

import logging
import os
from typing import Annotated

from fastapi import APIRouter, BackgroundTasks, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from backend.app.config import get_settings
from backend.app.models.blocks import (
    JobStatusResponse,
    ParseResult,
    TextParseRequest,
    UrlParseRequest,
)
from backend.app.pipeline.adapters.file import (
    build_file_context,
    build_text_context,
    build_url_context,
    write_upload_temp,
)
from backend.app.pipeline.adapters.url import UrlFetchError
from backend.app.pipeline.orchestrator import run_pipeline
from backend.app.services.job_store import job_store

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/parse", tags=["parse"])

SUPPORTED_UPLOAD_EXTS = {
    ".pdf",
    ".docx",
    ".pptx",
    ".csv",
    ".txt",
    ".md",
    ".markdown",
    ".png",
    ".jpg",
    ".jpeg",
    ".webp",
    ".html",
    ".htm",
    ".xlsx",
    ".xls",
}


def _ext_ok(filename: str) -> bool:
    from pathlib import Path

    return Path(filename).suffix.lower() in SUPPORTED_UPLOAD_EXTS


async def _run_job(job_id: str, ctx, source, cleanup_paths: list[str]) -> None:
    await job_store.set_status(job_id, "running")
    try:
        result = await run_pipeline(ctx, job_id=job_id, source=source)
        await job_store.complete(job_id, result)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Background parse failed for %s", job_id)
        await job_store.fail(job_id, str(exc))
    finally:
        for path in cleanup_paths:
            try:
                if path and os.path.exists(path):
                    os.unlink(path)
            except OSError:
                pass


@router.post("/file", response_model=ParseResult)
async def parse_file(
    background_tasks: BackgroundTasks,
    file: Annotated[UploadFile, File(...)],
    async_mode: Annotated[bool | None, Form()] = None,
):
    """Parse an uploaded file. Large files return 202 + job_id."""
    settings = get_settings()
    filename = file.filename or "upload.bin"
    if not _ext_ok(filename):
        raise HTTPException(status_code=415, detail=f"Unsupported file type: {filename}")

    content = await file.read()
    size = len(content)
    if size > settings.max_file_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"File too large ({size / 1024 / 1024:.1f}MB). Maximum is {settings.parser_max_file_mb}MB.",
        )
    if size == 0:
        raise HTTPException(status_code=400, detail="Empty file")

    path = write_upload_temp(content, filename)
    try:
        ctx, source = build_file_context(
            file_path=path,
            filename=filename,
            size_bytes=size,
            mime=file.content_type,
            settings=settings,
        )
    except ValueError as exc:
        os.unlink(path)
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    force_async = async_mode is True
    force_sync = async_mode is False
    use_async = force_async or (not force_sync and size > settings.sync_max_bytes)

    if use_async:
        rec = await job_store.create()
        background_tasks.add_task(_run_job, rec.job_id, ctx, source, [path])
        return JSONResponse(
            status_code=202,
            content={"job_id": rec.job_id, "status": "pending"},
        )

    try:
        result = await run_pipeline(ctx, source=source)
        return result
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass


@router.post("/text", response_model=ParseResult)
async def parse_text(body: TextParseRequest):
    """Parse pasted plain text, markdown, or HTML."""
    if not body.text or not body.text.strip():
        raise HTTPException(status_code=400, detail="text must be non-empty")
    try:
        ctx, source = build_text_context(body.text, fmt=body.format)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return await run_pipeline(ctx, source=source)


@router.post("/url", response_model=ParseResult)
async def parse_url(body: UrlParseRequest, background_tasks: BackgroundTasks):
    """Fetch a URL (SSRF-safe) and parse HTML or linked files."""
    settings = get_settings()
    cleanup: list[str] = []
    try:
        ctx, source, tmp = await build_url_context(body.url, settings=settings)
        if tmp:
            cleanup.append(tmp)
    except UrlFetchError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    size = ctx.size_bytes or 0
    if size > settings.sync_max_bytes:
        rec = await job_store.create()
        background_tasks.add_task(_run_job, rec.job_id, ctx, source, cleanup)
        return JSONResponse(
            status_code=202,
            content={"job_id": rec.job_id, "status": "pending"},
        )

    try:
        return await run_pipeline(ctx, source=source)
    finally:
        for path in cleanup:
            try:
                if path and os.path.exists(path):
                    os.unlink(path)
            except OSError:
                pass


@router.get("/{job_id}", response_model=JobStatusResponse)
async def get_job(job_id: str):
    """Poll an async parse job."""
    rec = await job_store.get(job_id)
    if not rec:
        raise HTTPException(status_code=404, detail="Job not found")
    return JobStatusResponse(
        job_id=rec.job_id,
        status=rec.status,
        result=rec.result,
        error=rec.error,
    )
