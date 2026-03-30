# converter.py — Document → Markdown conversion with post-processing
#
# Pipeline: Extract raw markdown → Convert tables to lists → Handle image descriptions
# Uses pymupdf4llm (PDF), python-docx (DOCX), python-pptx (PPTX), pandas (CSV)

import logging
import re
import os
import tempfile
import pandas as pd

logger = logging.getLogger(__name__)

MAX_FILE_SIZE_MB = 100
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024


def _extract_pdf(file_path: str, embed_images: bool = False) -> str:
    import pymupdf4llm
    return pymupdf4llm.to_markdown(file_path, embed_images=embed_images)


def _extract_docx(file_path: str) -> str:
    from docx import Document
    doc = Document(file_path)
    parts = []
    for para in doc.paragraphs:
        style = para.style.name if para.style else ""
        text = para.text.strip()
        if not text:
            parts.append("")
            continue
        # Map heading styles to markdown headers
        if style.startswith("Heading 1"):
            parts.append(f"# {text}")
        elif style.startswith("Heading 2"):
            parts.append(f"## {text}")
        elif style.startswith("Heading 3"):
            parts.append(f"### {text}")
        elif style.startswith("List"):
            parts.append(f"- {text}")
        else:
            parts.append(text)

    # Extract tables
    for table in doc.tables:
        if not table.rows:
            continue
        headers = [cell.text.strip() for cell in table.rows[0].cells]
        parts.append("")
        parts.append("| " + " | ".join(headers) + " |")
        parts.append("| " + " | ".join(["---"] * len(headers)) + " |")
        for row in table.rows[1:]:
            cells = [cell.text.strip() for cell in row.cells]
            parts.append("| " + " | ".join(cells) + " |")
        parts.append("")

    return "\n".join(parts)


def _extract_pptx(file_path: str) -> str:
    from pptx import Presentation
    prs = Presentation(file_path)
    parts = []
    for i, slide in enumerate(prs.slides, 1):
        parts.append(f"## Slide {i}")
        for shape in slide.shapes:
            if shape.has_text_frame:
                for para in shape.text_frame.paragraphs:
                    text = para.text.strip()
                    if text:
                        parts.append(text)
            if shape.has_table:
                table = shape.table
                headers = [cell.text.strip() for cell in table.rows[0].cells]
                parts.append("")
                parts.append("| " + " | ".join(headers) + " |")
                parts.append("| " + " | ".join(["---"] * len(headers)) + " |")
                for row in list(table.rows)[1:]:
                    cells = [cell.text.strip() for cell in row.cells]
                    parts.append("| " + " | ".join(cells) + " |")
                parts.append("")
        parts.append("")
    return "\n".join(parts)


def _extract_csv(file_path: str) -> str:
    for encoding in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            df = pd.read_csv(file_path, encoding=encoding)
            return df.to_markdown(index=False)
        except UnicodeDecodeError:
            continue
    # Last resort: replace errors
    df = pd.read_csv(file_path, encoding="utf-8", errors="replace")
    return df.to_markdown(index=False)


def _extract_markdown(file_path: str, llm_client=None) -> str:
    """Extract markdown from PDF, DOCX, PPTX, or CSV."""
    file_size = os.path.getsize(file_path)
    if file_size > MAX_FILE_SIZE_BYTES:
        raise ValueError(f"File too large ({file_size / 1024 / 1024:.1f}MB). Maximum is {MAX_FILE_SIZE_MB}MB.")

    ext = os.path.splitext(file_path)[1].lower()

    if ext == ".pdf":
        # Embed images as base64 when we have an LLM client for Vision
        return _extract_pdf(file_path, embed_images=bool(llm_client))
    elif ext == ".docx":
        return _extract_docx(file_path)
    elif ext == ".pptx":
        return _extract_pptx(file_path)
    elif ext == ".csv":
        return _extract_csv(file_path)
    else:
        raise ValueError(f"Unsupported file type: {ext}")


def _clean_cell(text: str) -> str:
    """Clean a table cell: convert <br> to newlines, strip markdown artifacts."""
    text = re.sub(r"<br\s*/?>", "\n", text)  # <br> → newline
    text = re.sub(r"\u200b", "", text)        # zero-width space
    text = text.strip()
    # Convert bullet markers to plain dashes
    text = re.sub(r"^[●○]\s*", "- ", text, flags=re.MULTILINE)
    return text


def _parse_pipe_row(line: str) -> list[str]:
    """Parse a single pipe-delimited row into cells, handling <br> within cells."""
    line = line.strip()
    if line.startswith("|"):
        line = line[1:]
    if line.endswith("|"):
        line = line[:-1]
    return [_clean_cell(c) for c in line.split("|")]


def _is_separator(line: str) -> bool:
    """Check if a line is a markdown table separator (|---|---|)."""
    return bool(re.match(r"^\|[-:\s|]+\|[ \t]*$", line.strip()))


def _table_to_list(lines: list[str]) -> str:
    """Convert parsed table lines into a nested bullet list."""
    if not lines:
        return ""

    # Find separator line
    sep_idx = -1
    for i, line in enumerate(lines):
        if _is_separator(line):
            sep_idx = i
            break

    if sep_idx >= 0:
        # Header row is above separator
        headers = _parse_pipe_row(lines[sep_idx - 1]) if sep_idx > 0 else []
        # The row above separator might also be a data row if headers are generic
        # (e.g. "Col1", "Col2") — detect and include it as data
        generic_headers = all(
            re.match(r"^(Col\d+|Component|Visual|Description|Data sample|)$", h, re.IGNORECASE)
            for h in headers
        )

        data_start = sep_idx + 1
        # If header row appears before the first separator AND there's real data above it, treat it as data
        pre_header_rows = []
        if sep_idx > 1:
            for i in range(0, sep_idx - 1):
                if lines[i].strip().startswith("|") and not _is_separator(lines[i]):
                    pre_header_rows.append(lines[i])
    else:
        # No separator — treat first row as header
        headers = _parse_pipe_row(lines[0])
        generic_headers = True
        data_start = 1
        pre_header_rows = []

    # Collect data rows (skip any additional separators)
    data_rows = []
    for row_line in pre_header_rows:
        data_rows.append(_parse_pipe_row(row_line))

    if generic_headers and sep_idx > 0:
        # The "header" is actually data — include it
        data_rows.append(_parse_pipe_row(lines[sep_idx - 1]))

    for line in lines[data_start:]:
        if _is_separator(line) or not line.strip():
            continue
        data_rows.append(_parse_pipe_row(line))

    if not data_rows:
        return "\n".join(lines)

    # Determine real column names
    if not generic_headers or not headers:
        # Use Col1, Col2, ... as fallback
        max_cols = max(len(r) for r in data_rows) if data_rows else 1
        headers = [f"Col {i+1}" for i in range(max_cols)]

    result_lines = []
    for i, row in enumerate(data_rows):
        # Use first cell as row label if it has content
        label = row[0].split("\n")[0].strip() if row and row[0].strip() else f"Row {i+1}"
        result_lines.append(f"- **{label}**:")
        for j, header in enumerate(headers):
            value = row[j] if j < len(row) else ""
            if value and value != label:
                # Multi-line values get indented
                value_lines = value.split("\n")
                if len(value_lines) > 1:
                    result_lines.append(f"  - {header}:")
                    for vl in value_lines:
                        vl = vl.strip()
                        if vl:
                            result_lines.append(f"    - {vl}")
                else:
                    result_lines.append(f"  - {header}: {value}")

    return "\n".join(result_lines) + "\n"


def _normalize_double_pipes(markdown: str) -> str:
    """Normalize || (double pipes) in table rows — pymupdf4llm artifact.

    Handles two cases:
    1. Empty cells: |Cell1||Cell3| → |Cell1| |Cell3|
    2. Joined rows: |---|---||Data| → split into two lines
    """
    lines = markdown.split("\n")
    result = []
    for line in lines:
        stripped = line.strip()
        if "||" in stripped and stripped.startswith("|"):
            # Split at || — this may join a separator row with a data row
            parts = stripped.split("||")
            reconstructed = []
            for p in parts:
                p = p.strip()
                if not p:
                    continue
                if not p.startswith("|"):
                    p = "|" + p
                if not p.endswith("|"):
                    p = p + "|"
                reconstructed.append(p)
            if len(reconstructed) > 1:
                # Check if any part is a separator — if so, split into separate lines
                has_sep = any(_is_separator(r) for r in reconstructed)
                if has_sep:
                    result.extend(reconstructed)
                else:
                    # Just empty-cell normalization: rejoin with | |
                    result.append(re.sub(r"\|\|", "| |", line))
            else:
                result.append(re.sub(r"\|\|", "| |", line))
        else:
            result.append(line)
    return "\n".join(result)


def _tables_to_lists(markdown: str) -> tuple:
    """Find markdown pipe-tables and convert them to nested bullet lists.

    Handles:
    - Standard tables (header + separator + data)
    - Tables with <br> in cells (from PDF extraction)
    - Tables split across pages (continuation with new separator)
    - Headerless/generic-header tables
    - Double-pipe || artifacts (empty cells)

    Returns (transformed_markdown, tables_found_count).
    """
    # Pre-process: normalize || to | | before table detection
    markdown = _normalize_double_pipes(markdown)

    # Pattern: any block of consecutive pipe-rows (possibly with separator lines)
    # A table is 2+ consecutive lines starting with |
    table_pattern = re.compile(
        r"((?:^\|.+\|[ \t]*$\n?){2,})",
        re.MULTILINE,
    )

    tables_found = 0

    def _replace(match):
        nonlocal tables_found
        block = match.group(1).strip()
        lines = [l for l in block.split("\n") if l.strip()]

        # Must have at least one non-separator line
        data_lines = [l for l in lines if not _is_separator(l)]
        if not data_lines:
            return block

        tables_found += 1
        return _table_to_list(lines)

    transformed = table_pattern.sub(_replace, markdown)
    return transformed, tables_found


def _describe_image_with_vision(data_uri: str, llm_client) -> str:
    """Send a base64 image to Claude Vision and get a text description."""
    try:
        response = llm_client.chat.completions.create(
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": data_uri}},
                    {"type": "text", "text": (
                        "Describe this image concisely for a technical document. "
                        "If it's a chart or diagram, describe the data and key takeaways. "
                        "If it's a UI screenshot, describe the interface elements and layout. "
                        "Keep it under 3 sentences."
                    )},
                ],
            }],
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Image could not be described: {e}"


def _handle_images(markdown: str, llm_client=None) -> tuple:
    """Replace image references with AI descriptions (if client available) or placeholders."""
    img_pattern = re.compile(r"!\[([^\]]*)\]\(([^)]+)\)")

    images_found = 0
    images_described = 0

    def _replace_img(match):
        nonlocal images_found, images_described
        images_found += 1
        alt_text = match.group(1).strip()
        url = match.group(2).strip()

        # If we have an LLM client and a base64 data URI, use Claude Vision
        if llm_client and url.startswith("data:image/"):
            images_described += 1
            desc = _describe_image_with_vision(url, llm_client)
            return f"[Image: {desc}]"

        # Fallback to alt text or generic placeholder
        if alt_text:
            return f"[Image: {alt_text}]"
        return "[Image: visual content present in original document]"

    transformed = img_pattern.sub(_replace_img, markdown)
    return transformed, images_found


def process_document(file_path: str, llm_client=None) -> dict:
    """Full conversion pipeline: extract → tables to lists → image handling.

    Returns:
        {
            "raw_markdown": str,
            "tables_converted": str,
            "final_markdown": str,
            "stats": {"tables_found": int, "images_found": int, "char_count": int}
        }
    """
    # Validate file path is within temp directory to prevent path traversal
    real_path = os.path.realpath(file_path)
    temp_dir = os.path.realpath(tempfile.gettempdir())
    if not real_path.startswith(temp_dir) and not os.path.isabs(file_path):
        raise ValueError(f"Invalid file path: must be an absolute path or within temp directory")
    if not os.path.isfile(real_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    raw_markdown = _extract_markdown(real_path, llm_client=llm_client)

    tables_converted, tables_found = _tables_to_lists(raw_markdown)

    final_markdown, images_found = _handle_images(tables_converted, llm_client=llm_client)

    return {
        "raw_markdown": raw_markdown,
        "tables_converted": tables_converted,
        "final_markdown": final_markdown,
        "stats": {
            "tables_found": tables_found,
            "images_found": images_found,
            "char_count": len(final_markdown),
        },
    }
