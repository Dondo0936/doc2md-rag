"""Kind-aware diffs and the LLM-facing changelog.

A hash tells you *that* something changed. Generation needs to say *what*
changed, so diffs are computed per chunk kind: a text patch, table cells,
image asset plus description. We never look at pixels; if you never stored a
description we say so instead of inventing one.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from difflib import SequenceMatcher, ndiff
from typing import Any

from .models import ChunkChange, Diff, Posting

ARROW = "\u2192"
_MAX_SNIPPET = 160
_MAX_PROMPT_CELLS = 8


def truncate(value: Any, limit: int = _MAX_SNIPPET) -> str:
    text = "" if value is None else str(value)
    collapsed = " ".join(text.split())
    if len(collapsed) <= limit:
        return collapsed
    return collapsed[: limit - 1].rstrip() + "\u2026"


def text_patch(old: str, new: str) -> str:
    """Minus/plus lines for the changed portions of two text payloads."""
    lines = [
        line
        for line in ndiff(old.splitlines() or [""], new.splitlines() or [""])
        if line.startswith(("-", "+")) and line[2:].strip() != ""
    ]
    return "\n".join(lines)


def summarize_text_change(old: str, new: str, *, max_spans: int = 3, max_span_tokens: int = 10) -> str | None:
    """Narrow a text edit to the words that moved, or ``None`` if it is too broad."""
    old_tokens = old.split()
    new_tokens = new.split()
    matcher = SequenceMatcher(a=old_tokens, b=new_tokens, autojunk=False)
    operations = [op for op in matcher.get_opcodes() if op[0] != "equal"]
    if not operations or len(operations) > max_spans:
        return None
    parts: list[str] = []
    for tag, i1, i2, j1, j2 in operations:
        if (i2 - i1) > max_span_tokens or (j2 - j1) > max_span_tokens:
            return None
        old_span = " ".join(old_tokens[i1:i2])
        new_span = " ".join(new_tokens[j1:j2])
        if tag == "replace" and max(i2 - i1, j2 - j1) < 3:
            # "30" -> "14" reads better as "30 days" -> "14 days".
            context = new_tokens[j2 : j2 + 1]
            if context and old_tokens[i2 : i2 + 1] == context:
                old_span = " ".join(old_tokens[i1:i2] + context)
                new_span = " ".join(new_tokens[j1:j2] + context)
        if tag == "replace":
            parts.append(f'"{old_span}" {ARROW} "{new_span}"')
        elif tag == "delete":
            parts.append(f'removed "{old_span}"')
        else:
            parts.append(f'added "{new_span}"')
    return "; ".join(parts)


def a1_ref(row: int, col: int) -> str:
    """Spreadsheet-style reference for a zero-indexed cell, e.g. (1, 1) -> B2."""
    label = ""
    remaining = col
    while True:
        label = chr(ord("A") + remaining % 26) + label
        remaining = remaining // 26 - 1
        if remaining < 0:
            break
    return f"{label}{row + 1}"


def table_cell_diff(from_rows: Sequence[Sequence[str]], to_rows: Sequence[Sequence[str]]) -> dict[str, Any]:
    """Cell-level table diff, aligning rows so an inserted row is not a cascade of edits."""
    old_grid = [list(row) for row in from_rows]
    new_grid = [list(row) for row in to_rows]
    matcher = SequenceMatcher(a=[tuple(row) for row in old_grid], b=[tuple(row) for row in new_grid], autojunk=False)

    cells: list[dict[str, Any]] = []
    rows_added: list[dict[str, Any]] = []
    rows_removed: list[dict[str, Any]] = []

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            continue
        if tag == "replace" and (i2 - i1) == (j2 - j1):
            for offset in range(i2 - i1):
                old_row = old_grid[i1 + offset]
                new_row = new_grid[j1 + offset]
                width = max(len(old_row), len(new_row))
                for col in range(width):
                    old_cell = old_row[col] if col < len(old_row) else ""
                    new_cell = new_row[col] if col < len(new_row) else ""
                    if old_cell != new_cell:
                        cells.append(
                            {
                                "row": j1 + offset,
                                "col": col,
                                "ref": a1_ref(j1 + offset, col),
                                "from": old_cell,
                                "to": new_cell,
                            }
                        )
            continue
        if tag in {"replace", "delete"}:
            rows_removed.extend({"row": i1 + offset, "cells": old_grid[i1 + offset]} for offset in range(i2 - i1))
        if tag in {"replace", "insert"}:
            rows_added.extend({"row": j1 + offset, "cells": new_grid[j1 + offset]} for offset in range(j2 - j1))

    return {"cells": cells, "rows_added": rows_added, "rows_removed": rows_removed}


def _describing_change(
    from_payload: Mapping[str, Any], to_payload: Mapping[str, Any], fields: Sequence[str]
) -> dict[str, Any]:
    detail: dict[str, Any] = {}
    for name in fields:
        old = from_payload.get(name)
        new = to_payload.get(name)
        if old != new:
            detail[name] = {"from": old, "to": new}
    return detail


def _image_detail(from_payload: Mapping[str, Any], to_payload: Mapping[str, Any]) -> dict[str, Any]:
    detail: dict[str, Any] = {
        "from_sha256": from_payload.get("sha256"),
        "to_sha256": to_payload.get("sha256"),
        "from_uri": from_payload.get("uri"),
        "to_uri": to_payload.get("uri"),
    }
    from_description = from_payload.get("description")
    to_description = to_payload.get("description")
    if from_description is not None or to_description is not None:
        detail["description"] = {"from": from_description, "to": to_description}
    if from_payload.get("caption") != to_payload.get("caption"):
        detail["caption"] = {"from": from_payload.get("caption"), "to": to_payload.get("caption")}
    return detail


def _changed_entry(before: Posting, after: Posting) -> ChunkChange:
    from_payload = dict(before.payload)
    to_payload = dict(after.payload)

    if before.kind != after.kind:
        return ChunkChange(
            key=after.chunk_key,
            kind=after.kind,
            change="kind_changed",
            from_id=before.id,
            to_id=after.id,
            from_kind=before.kind,
            to_kind=after.kind,
            detail={"from_payload": from_payload, "to_payload": to_payload},
        )

    asset_changed = before.asset_hash != after.asset_hash
    detail: dict[str, Any] = {}
    change = "changed"

    if not asset_changed:
        change = "metadata_changed"
        detail.update(_describing_change(from_payload, to_payload, ("uri", "caption", "description", "mime")))
        if dict(before.metadata) != dict(after.metadata):
            detail["metadata"] = {"from": dict(before.metadata), "to": dict(after.metadata)}
        if after.kind == "text":
            detail.setdefault("note", "payload unchanged; index metadata changed")
    elif after.kind == "text":
        old_text = str(from_payload.get("text", ""))
        new_text = str(to_payload.get("text", ""))
        detail["patch"] = text_patch(old_text, new_text)
        detail["from_text"] = old_text
        detail["to_text"] = new_text
    elif after.kind == "table":
        from_rows = from_payload.get("rows")
        to_rows = to_payload.get("rows")
        if from_rows is not None and to_rows is not None:
            detail.update(table_cell_diff(from_rows, to_rows))
            detail["from_shape"] = from_payload.get("shape")
            detail["to_shape"] = to_payload.get("shape")
        else:
            detail["note"] = "table changed; rows not stored so no cell diff"
            detail["from_hash"] = from_payload.get("hash", before.asset_hash)
            detail["to_hash"] = to_payload.get("hash", after.asset_hash)
        detail.update(_describing_change(from_payload, to_payload, ("caption",)))
    elif after.kind == "image":
        change = "replaced"
        detail.update(_image_detail(from_payload, to_payload))
        if detail.get("description") is None:
            detail["note"] = "image replaced; no description stored so no readable changelog"
    else:
        change = "replaced"
        detail["mime"] = to_payload.get("mime")
        detail["from_hash"] = before.asset_hash
        detail["to_hash"] = after.asset_hash
        if dict(before.metadata) != dict(after.metadata):
            detail["metadata"] = {"from": dict(before.metadata), "to": dict(after.metadata)}

    return ChunkChange(
        key=after.chunk_key,
        kind=after.kind,
        change=change,
        from_id=before.id,
        to_id=after.id,
        detail=detail,
    )


def compute_diff(
    doc_id: str,
    *,
    from_rev: int | None,
    to_rev: int,
    from_postings: Sequence[Posting],
    to_postings: Sequence[Posting],
    deleted: bool = False,
) -> Diff:
    before = {posting.chunk_key: posting for posting in from_postings}
    after = {posting.chunk_key: posting for posting in to_postings}

    added: list[ChunkChange] = []
    removed: list[ChunkChange] = []
    changed: list[ChunkChange] = []
    unchanged = 0

    for key, posting in after.items():
        if key not in before:
            added.append(
                ChunkChange(
                    key=key,
                    kind=posting.kind,
                    change="added",
                    to_id=posting.id,
                    detail={"payload": dict(posting.payload)},
                )
            )
        elif before[key].content_hash == posting.content_hash:
            unchanged += 1
        else:
            changed.append(_changed_entry(before[key], posting))

    for key, posting in before.items():
        if key not in after:
            removed.append(
                ChunkChange(
                    key=key,
                    kind=posting.kind,
                    change="removed",
                    from_id=posting.id,
                    detail={"payload": dict(posting.payload)},
                )
            )

    return Diff(
        doc_id=doc_id,
        from_rev=from_rev,
        to_rev=to_rev,
        added=tuple(added),
        removed=tuple(removed),
        changed=tuple(changed),
        unchanged_count=unchanged,
        deleted=deleted,
    )


def _changed_lines(change: ChunkChange) -> list[str]:
    detail = dict(change.detail)
    label = f"{change.kind}/{change.key}"
    lines: list[str] = []

    if change.change == "kind_changed":
        return [f"- {label}: kind changed {change.from_kind} {ARROW} {change.to_kind} (re-indexed as a new chunk)"]

    if change.change == "metadata_changed":
        parts = []
        for name in ("description", "caption", "uri", "mime"):
            if name in detail:
                parts.append(f'{name} "{truncate(detail[name]["from"])}" {ARROW} "{truncate(detail[name]["to"])}"')
        if "metadata" in detail:
            parts.append("metadata changed")
        summary = "; ".join(parts) or "index metadata changed"
        suffix = " (asset unchanged)" if change.kind in {"image", "other"} else ""
        return [f"- {label}: {summary}{suffix}"]

    if change.kind == "text":
        summary = summarize_text_change(str(detail.get("from_text", "")), str(detail.get("to_text", "")))
        if summary:
            return [f"- {label}: {summary}"]
        lines.append(f"- {label}: changed")
        lines.extend(f"    {line}" for line in str(detail.get("patch", "")).splitlines())
        return lines

    if change.kind == "table":
        if "cells" in detail:
            cells = detail["cells"]
            for cell in cells[:_MAX_PROMPT_CELLS]:
                lines.append(
                    f"- {label} cell {cell['ref']}: {truncate(cell['from'])} {ARROW} {truncate(cell['to'])}"
                )
            if len(cells) > _MAX_PROMPT_CELLS:
                lines.append(f"- {label}: and {len(cells) - _MAX_PROMPT_CELLS} more cells changed")
            added_rows = len(detail.get("rows_added", []))
            removed_rows = len(detail.get("rows_removed", []))
            if added_rows or removed_rows:
                lines.append(f"- {label}: {added_rows} rows added, {removed_rows} rows removed")
            if not lines:
                lines.append(f"- {label}: table changed")
        else:
            lines.append(f"- {label}: table changed (rows not stored, so no cell detail)")
        if "caption" in detail:
            lines.append(
                f'- {label}: caption "{truncate(detail["caption"]["from"])}" '
                f'{ARROW} "{truncate(detail["caption"]["to"])}"'
            )
        return lines

    if change.kind == "image":
        description = detail.get("description")
        if description and (description.get("from") or description.get("to")):
            lines.append(
                f'- {label}: "{truncate(description.get("from"))}" {ARROW} "{truncate(description.get("to"))}"'
            )
        else:
            lines.append(f"- {label}: image replaced (no description stored)")
        if detail.get("from_uri") or detail.get("to_uri"):
            lines.append(f"  old: {detail.get('from_uri')}")
            lines.append(f"  new: {detail.get('to_uri')}")
        return lines

    lines.append(f"- {label}: replaced ({detail.get('mime')})")
    return lines


def _added_line(change: ChunkChange) -> str:
    payload = dict(change.detail.get("payload", {}))
    label = f"{change.kind}/{change.key}"
    if change.kind == "text":
        return f'- {label}: added "{truncate(payload.get("text"))}"'
    if change.kind == "table":
        shape = payload.get("shape")
        shape_text = f"{shape[0]}x{shape[1]} " if shape else ""
        return f"- {label}: added ({shape_text}table)"
    if change.kind == "image":
        description = payload.get("description") or payload.get("caption")
        if description:
            return f'- {label}: added "{truncate(description)}"'
        return f"- {label}: added (image {truncate(payload.get('uri') or payload.get('sha256'))})"
    return f"- {label}: added ({payload.get('mime')})"


def _removed_line(change: ChunkChange) -> str:
    payload = dict(change.detail.get("payload", {}))
    label = f"{change.kind}/{change.key}"
    if change.kind == "text":
        return f'- {label}: removed (was "{truncate(payload.get("text"))}")'
    return f"- {label}: removed"


def render_prompt(diff: Diff, *, current_rev: int) -> str:
    """Short markdown a caller can staple onto any prompt."""
    doc = diff.doc_id
    if diff.deleted:
        header = f"Document {doc} was deleted at rev {diff.to_rev} (was rev {diff.from_rev})."
    elif diff.from_rev is None or diff.from_rev == 0:
        header = f"Document {doc} is at rev {diff.to_rev} (first revision)."
    else:
        header = f"Document {doc} is now rev {diff.to_rev} (was {diff.from_rev})."

    if diff.is_empty:
        if diff.from_rev == diff.to_rev:
            return f"Document {doc} has no recorded changes at rev {diff.to_rev} (current)."
        return f"{header}\nNo chunk-level changes between rev {diff.from_rev} and rev {diff.to_rev}."

    lines = [header]
    for change in diff.changed:
        lines.extend(_changed_lines(change))
    for change in diff.added:
        lines.append(_added_line(change))
    for change in diff.removed:
        lines.append(_removed_line(change))

    footer = []
    if diff.unchanged_count:
        noun = "chunk" if diff.unchanged_count == 1 else "chunks"
        footer.append(f"{diff.unchanged_count} {noun} unchanged.")
    footer.append(f"Current revision is {current_rev}.")
    lines.append(" ".join(footer))
    return "\n".join(lines)


__all__ = [
    "a1_ref",
    "compute_diff",
    "render_prompt",
    "summarize_text_change",
    "table_cell_diff",
    "text_patch",
    "truncate",
]
