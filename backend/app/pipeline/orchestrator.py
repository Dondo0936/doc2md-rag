"""Pipeline orchestrator — runs extractor chains with retry + quality gate."""

from __future__ import annotations

import asyncio
import logging
import time
import uuid

from backend.app.config import Settings, get_settings
from backend.app.exporters.markdown import blocks_to_markdown, compute_stats
from backend.app.models.blocks import ParseResult, PipelineStep, SourceInfo
from backend.app.pipeline.context import ExtractContext
from backend.app.pipeline.quality import passes_quality
from backend.app.pipeline.registry import resolve_chain

logger = logging.getLogger(__name__)


def _detect_scanned_pdf(file_path: str) -> bool:
    try:
        import converter

        return bool(converter._is_scanned_pdf(file_path))
    except Exception as exc:  # noqa: BLE001
        logger.debug("Scanned PDF check failed: %s", exc)
        return False


def prepare_context(ctx: ExtractContext, settings: Settings | None = None) -> ExtractContext:
    """Enrich context with OCR detection and vision client."""
    settings = settings or get_settings()
    ext = ctx.resolve_extension().lstrip(".").lower()

    if ctx.file_path and ext == "pdf" and not ctx.use_ocr:
        ctx.use_ocr = _detect_scanned_pdf(ctx.file_path)

    if settings.enable_image_desc and ext in {"pdf", "docx", "pptx"}:
        ctx.describe_images = True

    if ctx.llm_client is None and settings.anthropic_api_key:
        try:
            from claude_wrapper import get_llm_client

            ctx.llm_client = get_llm_client(settings.anthropic_api_key)
        except Exception as exc:  # noqa: BLE001
            logger.debug("LLM client init failed: %s", exc)

    return ctx


async def run_pipeline(
    ctx: ExtractContext,
    *,
    job_id: str | None = None,
    source: SourceInfo | None = None,
    settings: Settings | None = None,
) -> ParseResult:
    """Execute the extractor chain and return a ParseResult."""
    settings = settings or get_settings()
    job_id = job_id or uuid.uuid4().hex
    ctx = prepare_context(ctx, settings)

    warnings: list[str] = []
    if ctx.use_ocr and ctx.resolve_extension().lstrip(".").lower() == "pdf":
        warnings.append("Scanned PDF detected — OCR-enabled extractors preferred")

    chain = resolve_chain(ctx)
    trace: list[PipelineStep] = []
    last_error: str | None = None
    winning_blocks = None
    winning_extractor: str | None = None
    winner_index: int | None = None

    for idx, step in enumerate(chain):
        extractor = step.extractor
        attempts = max(1, step.retries)

        for attempt in range(1, attempts + 1):
            t0 = time.perf_counter()
            try:
                blocks = await extractor.extract(ctx)
                duration_ms = int((time.perf_counter() - t0) * 1000)

                if not passes_quality(blocks, min_chars=settings.quality_min_chars):
                    trace.append(
                        PipelineStep(
                            extractor=extractor.name,
                            attempt=attempt,
                            status="failed",
                            duration_ms=duration_ms,
                            error="quality_gate_rejected",
                        )
                    )
                    last_error = f"{extractor.name}: quality gate rejected"
                    if attempt < attempts:
                        await asyncio.sleep(min(1.0, 0.05 * (2 ** (attempt - 1))))
                    continue

                trace.append(
                    PipelineStep(
                        extractor=extractor.name,
                        attempt=attempt,
                        status="success",
                        duration_ms=duration_ms,
                        error=None,
                    )
                )
                winning_blocks = blocks
                winning_extractor = extractor.name
                winner_index = idx
                break
            except Exception as exc:  # noqa: BLE001
                duration_ms = int((time.perf_counter() - t0) * 1000)
                err = f"{type(exc).__name__}: {exc}"
                last_error = err
                trace.append(
                    PipelineStep(
                        extractor=extractor.name,
                        attempt=attempt,
                        status="failed",
                        duration_ms=duration_ms,
                        error=err,
                    )
                )
                logger.warning("Extractor %s attempt %s failed: %s", extractor.name, attempt, err)
                if attempt < attempts:
                    await asyncio.sleep(min(1.0, 0.05 * (2 ** (attempt - 1))))

        if winning_blocks is not None:
            break

    if winner_index is not None:
        for later in chain[winner_index + 1 :]:
            trace.append(
                PipelineStep(
                    extractor=later.extractor.name,
                    attempt=0,
                    status="skipped",
                    duration_ms=0,
                    error=None,
                )
            )

    if source is None:
        input_type = ctx.input_kind if ctx.input_kind in {"file", "text", "url", "image"} else "file"
        source = SourceInfo(
            input_type=input_type,  # type: ignore[arg-type]
            filename=ctx.filename,
            url=ctx.url,
            mime=ctx.mime,
            size_bytes=ctx.size_bytes,
            format=ctx.extra.get("format"),
        )

    if winning_blocks is None:
        return ParseResult(
            job_id=job_id,
            source=source,
            blocks=[],
            markdown="",
            stats=compute_stats([], ""),
            pipeline_trace=trace,
            warnings=warnings,
            status="failed",
            error=last_error or "All extractors failed",
        )

    markdown = blocks_to_markdown(winning_blocks, tables_as_lists=True)
    try:
        import converter

        markdown, _images_found = converter._handle_images(markdown, llm_client=ctx.llm_client)
    except Exception as exc:  # noqa: BLE001
        logger.debug("Image post-process skipped: %s", exc)

    stats = compute_stats(winning_blocks, markdown)
    if winning_extractor:
        warnings.append(f"Selected extractor: {winning_extractor}")

    return ParseResult(
        job_id=job_id,
        source=source,
        blocks=winning_blocks,
        markdown=markdown,
        stats=stats,
        pipeline_trace=trace,
        warnings=warnings,
        status="completed",
    )
