"""Structured document models — stable API contract for the parser platform."""

from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field


class BlockType(str, Enum):
    HEADING = "heading"
    PARAGRAPH = "paragraph"
    LIST = "list"
    TABLE = "table"
    IMAGE = "image"
    CODE = "code"
    QUOTE = "quote"
    PAGE_BREAK = "page_break"


class Block(BaseModel):
    id: str
    type: BlockType
    text: str | None = None
    level: int | None = None
    rows: list[list[str]] | None = None
    items: list[str] | None = None
    image_ref: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    children: list[Block] = Field(default_factory=list)


class SourceInfo(BaseModel):
    input_type: Literal["file", "text", "url", "image"]
    filename: str | None = None
    url: str | None = None
    mime: str | None = None
    size_bytes: int | None = None
    format: str | None = None  # plain | markdown | html for text inputs


class ParseStats(BaseModel):
    char_count: int = 0
    tables: int = 0
    images: int = 0
    pages: int | None = None
    block_counts: dict[str, int] = Field(default_factory=dict)


class PipelineStep(BaseModel):
    extractor: str
    attempt: int
    status: Literal["success", "failed", "skipped"]
    duration_ms: int
    error: str | None = None


class ParseResult(BaseModel):
    job_id: str
    source: SourceInfo
    blocks: list[Block]
    markdown: str
    stats: ParseStats
    pipeline_trace: list[PipelineStep] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    status: Literal["completed", "failed", "pending", "running"] = "completed"
    error: str | None = None


class TextParseRequest(BaseModel):
    text: str
    format: Literal["plain", "markdown", "html"] = "plain"


class UrlParseRequest(BaseModel):
    url: str


class JobStatusResponse(BaseModel):
    job_id: str
    status: Literal["pending", "running", "completed", "failed"]
    result: ParseResult | None = None
    error: str | None = None


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded", "starting"]
    docling_ready: bool
    version: str = "0.1.0"
