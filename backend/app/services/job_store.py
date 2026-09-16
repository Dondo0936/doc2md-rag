"""In-memory async job store (MVP). Swap for Redis later if needed."""

from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from typing import Literal

from backend.app.models.blocks import ParseResult


JobStatus = Literal["pending", "running", "completed", "failed"]


@dataclass
class JobRecord:
    job_id: str
    status: JobStatus = "pending"
    result: ParseResult | None = None
    error: str | None = None
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)


class JobStore:
    def __init__(self) -> None:
        self._jobs: dict[str, JobRecord] = {}
        self._lock = asyncio.Lock()

    def new_id(self) -> str:
        return uuid.uuid4().hex

    async def create(self, job_id: str | None = None) -> JobRecord:
        async with self._lock:
            jid = job_id or self.new_id()
            rec = JobRecord(job_id=jid, status="pending")
            self._jobs[jid] = rec
            return rec

    async def set_status(self, job_id: str, status: JobStatus, error: str | None = None) -> None:
        async with self._lock:
            rec = self._jobs.get(job_id)
            if not rec:
                return
            rec.status = status
            rec.error = error
            rec.updated_at = time.time()

    async def complete(self, job_id: str, result: ParseResult) -> None:
        async with self._lock:
            rec = self._jobs.get(job_id)
            if not rec:
                return
            rec.status = "completed" if result.status == "completed" else "failed"
            rec.result = result
            rec.error = result.error
            rec.updated_at = time.time()

    async def fail(self, job_id: str, error: str) -> None:
        async with self._lock:
            rec = self._jobs.get(job_id)
            if not rec:
                return
            rec.status = "failed"
            rec.error = error
            rec.updated_at = time.time()

    async def get(self, job_id: str) -> JobRecord | None:
        async with self._lock:
            return self._jobs.get(job_id)


# Process-wide singleton
job_store = JobStore()
