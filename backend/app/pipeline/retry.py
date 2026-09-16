"""Retry helpers with exponential backoff."""

from __future__ import annotations

import asyncio
import random
from collections.abc import Awaitable, Callable
from typing import TypeVar

T = TypeVar("T")


async def with_retries(
    fn: Callable[[], Awaitable[T]],
    *,
    attempts: int,
    base_delay_s: float = 0.05,
    max_delay_s: float = 1.0,
    jitter: bool = True,
) -> T:
    """Run an async callable up to `attempts` times with exponential backoff.

    attempts=1 means a single try (no retry). attempts=0 or less raises ValueError.
    """
    if attempts < 1:
        raise ValueError("attempts must be >= 1")

    last_exc: BaseException | None = None
    for i in range(attempts):
        try:
            return await fn()
        except BaseException as exc:  # noqa: BLE001 — extractors raise varied errors
            last_exc = exc
            if i >= attempts - 1:
                break
            delay = min(max_delay_s, base_delay_s * (2**i))
            if jitter:
                delay *= 0.5 + random.random()
            await asyncio.sleep(delay)
    assert last_exc is not None
    raise last_exc
