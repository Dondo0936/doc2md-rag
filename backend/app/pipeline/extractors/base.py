"""Extractor protocol and helpers."""

from __future__ import annotations

import asyncio
from typing import Protocol, runtime_checkable

from backend.app.exporters.markdown import markdown_to_blocks, stamp_extractor
from backend.app.models.blocks import Block
from backend.app.pipeline.context import ExtractContext


@runtime_checkable
class Extractor(Protocol):
    name: str
    supports: set[str]

    async def extract(self, ctx: ExtractContext) -> list[Block]:
        ...


def supports_ctx(extractor: Extractor, ctx: ExtractContext) -> bool:
    """True if extractor supports this context's extension or mime."""
    if not extractor.supports:
        return True
    ext = ctx.resolve_extension().lstrip(".").lower()
    mime = (ctx.mime or "").lower()
    kind = ctx.input_kind
    tokens = {ext, mime, kind, f".{ext}" if ext else ""}
    tokens.discard("")
    return bool(tokens & {s.lower() for s in extractor.supports})


async def run_sync(fn, *args, **kwargs):
    """Run a blocking extractor helper off the event loop."""
    return await asyncio.to_thread(fn, *args, **kwargs)


def markdown_result(md: str, extractor_name: str, *, extra_metadata: dict | None = None) -> list[Block]:
    blocks = markdown_to_blocks(md, extractor=extractor_name, extra_metadata=extra_metadata)
    return stamp_extractor(blocks, extractor_name)
