"""Quality gate for extractor output."""

from __future__ import annotations

from backend.app.models.blocks import Block, BlockType


def total_text_chars(blocks: list[Block]) -> int:
    total = 0

    def walk(items: list[Block]) -> None:
        nonlocal total
        for b in items:
            if b.text:
                total += len(b.text.strip())
            if b.items:
                total += sum(len(i.strip()) for i in b.items)
            if b.rows:
                for row in b.rows:
                    total += sum(len(c.strip()) for c in row)
            if b.children:
                walk(b.children)

    walk(blocks)
    return total


def passes_quality(
    blocks: list[Block],
    *,
    min_chars: int = 10,
    min_blocks: int = 1,
) -> bool:
    """Reject empty/trivial extraction results so the orchestrator tries the next extractor."""
    if not blocks or len(blocks) < min_blocks:
        return False
    # A single empty paragraph / empty heading does not count
    meaningful = [
        b
        for b in blocks
        if b.type != BlockType.PAGE_BREAK
        and (
            (b.text and b.text.strip())
            or (b.items and any(i.strip() for i in b.items))
            or (b.rows and any(any(c.strip() for c in row) for row in b.rows))
            or b.image_ref
        )
    ]
    if not meaningful:
        return False
    return total_text_chars(blocks) >= min_chars or any(b.type == BlockType.IMAGE for b in blocks)
