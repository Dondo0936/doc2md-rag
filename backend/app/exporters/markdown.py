"""Markdown ↔ block normalization and export.

Blocks are the source of truth. Markdown is always derived via `blocks_to_markdown`.
Extraction backends often emit markdown first; `markdown_to_blocks` converts that
into structured blocks without losing content.
"""

from __future__ import annotations

import re
import uuid
from typing import Iterable

from backend.app.models.blocks import Block, BlockType, ParseStats


def _new_id() -> str:
    return uuid.uuid4().hex[:12]


def _pipe_row_cells(line: str) -> list[str]:
    line = line.strip()
    if line.startswith("|"):
        line = line[1:]
    if line.endswith("|"):
        line = line[:-1]
    return [c.strip() for c in line.split("|")]


def _is_separator(line: str) -> bool:
    stripped = line.strip()
    if not stripped.startswith("|") or not stripped.endswith("|"):
        return False
    inner = stripped[1:-1]
    if not inner:
        return False
    cells = inner.split("|")
    return all(re.match(r"^\s*:?-{3,}:?\s*$", cell) for cell in cells)


def markdown_to_blocks(
    markdown: str,
    *,
    extractor: str | None = None,
    extra_metadata: dict | None = None,
) -> list[Block]:
    """Parse markdown text into structured blocks.

    Recognizes headings, fenced code, pipe tables, lists, images, quotes, and
    paragraphs. Provenance is stamped onto every block via metadata.
    """
    text = (markdown or "").replace("\r\n", "\n").replace("\r", "\n")
    lines = text.split("\n")
    blocks: list[Block] = []
    i = 0
    base_meta = dict(extra_metadata or {})
    if extractor:
        base_meta["extractor"] = extractor

    def meta(**kwargs) -> dict:
        m = dict(base_meta)
        m.update(kwargs)
        return m

    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        if not stripped:
            i += 1
            continue

        # Fenced code block
        if stripped.startswith("```"):
            lang = stripped[3:].strip() or None
            code_lines: list[str] = []
            i += 1
            while i < len(lines) and not lines[i].strip().startswith("```"):
                code_lines.append(lines[i])
                i += 1
            if i < len(lines):
                i += 1
            blocks.append(
                Block(
                    id=_new_id(),
                    type=BlockType.CODE,
                    text="\n".join(code_lines),
                    metadata=meta(language=lang) if lang else meta(),
                )
            )
            continue

        # Heading
        heading = re.match(r"^(#{1,6})\s+(.+)$", stripped)
        if heading:
            blocks.append(
                Block(
                    id=_new_id(),
                    type=BlockType.HEADING,
                    text=heading.group(2).strip(),
                    level=len(heading.group(1)),
                    metadata=meta(),
                )
            )
            i += 1
            continue

        # Pipe table
        if stripped.startswith("|") and stripped.endswith("|"):
            table_lines: list[str] = []
            while i < len(lines) and lines[i].strip().startswith("|") and lines[i].strip().endswith("|"):
                table_lines.append(lines[i].strip())
                i += 1
            rows: list[list[str]] = []
            for tl in table_lines:
                if _is_separator(tl):
                    continue
                rows.append(_pipe_row_cells(tl))
            if rows:
                blocks.append(
                    Block(
                        id=_new_id(),
                        type=BlockType.TABLE,
                        rows=rows,
                        metadata=meta(),
                    )
                )
            continue

        # Image-only line: ![alt](url) or [Image: ...]
        img_md = re.fullmatch(r"!\[([^\]]*)\]\(([^)]+)\)", stripped)
        if img_md:
            blocks.append(
                Block(
                    id=_new_id(),
                    type=BlockType.IMAGE,
                    text=img_md.group(1).strip() or None,
                    image_ref=img_md.group(2).strip(),
                    metadata=meta(),
                )
            )
            i += 1
            continue

        img_ph = re.fullmatch(r"\[Image:\s*(.+?)\]", stripped)
        if img_ph:
            blocks.append(
                Block(
                    id=_new_id(),
                    type=BlockType.IMAGE,
                    text=img_ph.group(1).strip(),
                    metadata=meta(),
                )
            )
            i += 1
            continue

        # Blockquote
        if stripped.startswith(">"):
            quote_lines: list[str] = []
            while i < len(lines) and lines[i].strip().startswith(">"):
                quote_lines.append(re.sub(r"^>\s?", "", lines[i].strip()))
                i += 1
            blocks.append(
                Block(
                    id=_new_id(),
                    type=BlockType.QUOTE,
                    text="\n".join(quote_lines),
                    metadata=meta(),
                )
            )
            continue

        # Unordered / ordered list
        if re.match(r"^([-*+]|\d+[.)])\s+", stripped):
            items: list[str] = []
            while i < len(lines):
                s = lines[i].strip()
                if not s:
                    # allow blank line inside list only if next is indented/list
                    if i + 1 < len(lines) and (
                        re.match(r"^([-*+]|\d+[.)])\s+", lines[i + 1].strip())
                        or lines[i + 1].startswith("  ")
                    ):
                        i += 1
                        continue
                    break
                m = re.match(r"^([-*+]|\d+[.)])\s+(.*)$", s)
                if m:
                    items.append(m.group(2))
                    i += 1
                    # gather continuation indented lines
                    while i < len(lines) and lines[i].startswith("  ") and lines[i].strip():
                        cont = lines[i].strip()
                        if re.match(r"^([-*+]|\d+[.)])\s+", cont):
                            break
                        items[-1] = items[-1] + "\n" + cont
                        i += 1
                    continue
                if lines[i].startswith("  "):
                    if items:
                        items[-1] = items[-1] + "\n" + s
                    i += 1
                    continue
                break
            if items:
                blocks.append(
                    Block(
                        id=_new_id(),
                        type=BlockType.LIST,
                        items=items,
                        metadata=meta(),
                    )
                )
            continue

        # Paragraph — gather until blank line or structural marker
        para_lines: list[str] = [stripped]
        i += 1
        while i < len(lines):
            nxt = lines[i]
            ns = nxt.strip()
            if not ns:
                break
            if (
                ns.startswith("#")
                or ns.startswith("|")
                or ns.startswith(">")
                or ns.startswith("```")
                or re.match(r"^([-*+]|\d+[.)])\s+", ns)
                or re.fullmatch(r"!\[([^\]]*)\]\(([^)]+)\)", ns)
                or re.fullmatch(r"\[Image:\s*(.+?)\]", ns)
            ):
                break
            para_lines.append(ns)
            i += 1
        blocks.append(
            Block(
                id=_new_id(),
                type=BlockType.PARAGRAPH,
                text=" ".join(para_lines),
                metadata=meta(),
            )
        )

    return blocks


def _table_rows_to_list_markdown(rows: list[list[str]]) -> str:
    """Convert structured table rows to nested bullet lists (LLM-friendly).

    Reuses the spirit of converter._table_to_list without requiring pipe syntax.
    """
    if not rows:
        return ""

    # Import table helpers from root converter for identical cell cleaning /
    # list formatting when we can round-trip through pipe tables.
    try:
        from converter import _table_to_list  # type: ignore

        max_cols = max(len(r) for r in rows)
        pipe_lines: list[str] = []
        for idx, row in enumerate(rows):
            padded = list(row) + [""] * (max_cols - len(row))
            cells = [c.replace("|", "∣").replace("\n", "<br>") for c in padded]
            pipe_lines.append("| " + " | ".join(cells) + " |")
            if idx == 0:
                pipe_lines.append("| " + " | ".join(["---"] * max_cols) + " |")
        return _table_to_list(pipe_lines).rstrip()
    except Exception:
        # Fallback without converter dependency
        headers = rows[0]
        data = rows[1:] if len(rows) > 1 else []
        out: list[str] = []
        for i, row in enumerate(data or rows):
            label = (row[0].split("\n")[0].strip() if row and row[0].strip() else f"Row {i + 1}")
            out.append(f"- **{label}**:")
            for j, header in enumerate(headers if data else [f"Col {k+1}" for k in range(len(row))]):
                if j >= len(row):
                    continue
                value = row[j]
                if not value or not header.strip():
                    continue
                if j == 0 and value.split("\n")[0].strip() == label:
                    continue
                out.append(f"  - {header}: {value}")
        return "\n".join(out)


def blocks_to_markdown(
    blocks: Iterable[Block],
    *,
    tables_as_lists: bool = True,
) -> str:
    """Export structured blocks to markdown.

    Tables become nested bullet lists by default (matching converter quality).
    """
    parts: list[str] = []
    for block in blocks:
        if block.type == BlockType.HEADING:
            level = max(1, min(block.level or 1, 6))
            parts.append(f"{'#' * level} {block.text or ''}")
        elif block.type == BlockType.PARAGRAPH:
            parts.append(block.text or "")
        elif block.type == BlockType.LIST:
            items = block.items or ([] if not block.text else [block.text])
            for item in items:
                first, *rest = (item or "").split("\n")
                parts.append(f"- {first}")
                for cont in rest:
                    parts.append(f"  {cont}")
        elif block.type == BlockType.TABLE:
            rows = block.rows or []
            if tables_as_lists:
                parts.append(_table_rows_to_list_markdown(rows))
            else:
                if not rows:
                    continue
                max_cols = max(len(r) for r in rows)
                for idx, row in enumerate(rows):
                    padded = list(row) + [""] * (max_cols - len(row))
                    cells = [c.replace("|", "∣").replace("\n", "<br>") for c in padded]
                    parts.append("| " + " | ".join(cells) + " |")
                    if idx == 0:
                        parts.append("| " + " | ".join(["---"] * max_cols) + " |")
        elif block.type == BlockType.IMAGE:
            if block.image_ref:
                alt = block.text or ""
                parts.append(f"![{alt}]({block.image_ref})")
            else:
                parts.append(f"[Image: {block.text or 'visual content'}]")
        elif block.type == BlockType.CODE:
            lang = (block.metadata or {}).get("language") or ""
            parts.append(f"```{lang}\n{block.text or ''}\n```")
        elif block.type == BlockType.QUOTE:
            for ql in (block.text or "").split("\n"):
                parts.append(f"> {ql}")
        elif block.type == BlockType.PAGE_BREAK:
            parts.append("\n---\n")
        else:
            if block.text:
                parts.append(block.text)

        if block.children:
            child_md = blocks_to_markdown(block.children, tables_as_lists=tables_as_lists)
            if child_md.strip():
                parts.append(child_md)

    md = "\n\n".join(p for p in parts if p is not None and p != "")
    md = re.sub(r"\n{4,}", "\n\n\n", md)
    return md.strip() + ("\n" if md.strip() else "")


def compute_stats(blocks: list[Block], markdown: str) -> ParseStats:
    counts: dict[str, int] = {}
    tables = 0
    images = 0
    pages = None

    def walk(items: list[Block]) -> None:
        nonlocal tables, images, pages
        for b in items:
            key = b.type.value if isinstance(b.type, BlockType) else str(b.type)
            counts[key] = counts.get(key, 0) + 1
            if b.type == BlockType.TABLE:
                tables += 1
            if b.type == BlockType.IMAGE:
                images += 1
            page = (b.metadata or {}).get("page")
            if isinstance(page, int):
                pages = page if pages is None else max(pages, page)
            if b.children:
                walk(b.children)

    walk(blocks)
    return ParseStats(
        char_count=len(markdown),
        tables=tables,
        images=images,
        pages=pages,
        block_counts=counts,
    )


def stamp_extractor(blocks: list[Block], extractor: str) -> list[Block]:
    """Ensure every block carries metadata.extractor."""
    for b in blocks:
        b.metadata = dict(b.metadata or {})
        b.metadata.setdefault("extractor", extractor)
        if b.children:
            stamp_extractor(b.children, extractor)
    return blocks
