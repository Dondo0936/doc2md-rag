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
    from docx.table import Table
    from docx.text.paragraph import Paragraph
    from docx.oxml.ns import qn

    doc = Document(file_path)
    parts = []

    def _process_paragraph(para):
        style = para.style.name if para.style else ""
        text = para.text.strip()
        if not text:
            parts.append("")
            return
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

    def _sanitize_cell(text):
        """Replace newlines and pipes in cell text to preserve pipe-table format."""
        text = text.strip()
        text = text.replace("\n", "<br>")  # newlines → <br> (converted back by _clean_cell)
        text = text.replace("|", "∣")      # avoid breaking pipe delimiters
        return text

    def _process_table(table):
        if not table.rows:
            return

        def _is_merged_row(row):
            """Detect merged rows where all non-empty cells contain identical text."""
            texts = [cell.text.strip() for cell in row.cells]
            unique = set(t for t in texts if t)
            return len(unique) == 1 and unique

        # Find the real header row (skip leading merged section headers)
        header_idx = 0
        for idx, row in enumerate(table.rows):
            if _is_merged_row(row):
                text = next(t for t in (cell.text.strip() for cell in row.cells) if t)
                parts.append("")
                parts.append(f"### {text}")
                header_idx = idx + 1
            else:
                break

        if header_idx >= len(table.rows):
            return

        headers = [_sanitize_cell(cell.text) for cell in table.rows[header_idx].cells]
        parts.append("")
        parts.append("| " + " | ".join(headers) + " |")
        parts.append("| " + " | ".join(["---"] * len(headers)) + " |")
        for row in table.rows[header_idx + 1:]:
            if _is_merged_row(row):
                text = next(t for t in (cell.text.strip() for cell in row.cells) if t)
                parts.append("")
                parts.append(f"### {text}")
                parts.append("")
                parts.append("| " + " | ".join(headers) + " |")
                parts.append("| " + " | ".join(["---"] * len(headers)) + " |")
                continue
            cells = [_sanitize_cell(cell.text) for cell in row.cells]
            parts.append("| " + " | ".join(cells) + " |")
        parts.append("")

    # Iterate body elements in document order so tables appear inline
    for element in doc.element.body:
        if element.tag == qn("w:p"):
            _process_paragraph(Paragraph(element, doc))
        elif element.tag == qn("w:tbl"):
            _process_table(Table(element, doc))

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

                def _pptx_sanitize(text):
                    text = text.strip().replace("\n", "<br>").replace("|", "∣")
                    return text

                headers = [_pptx_sanitize(cell.text) for cell in table.rows[0].cells]
                parts.append("")
                parts.append("| " + " | ".join(headers) + " |")
                parts.append("| " + " | ".join(["---"] * len(headers)) + " |")
                for row in list(table.rows)[1:]:
                    cells = [_pptx_sanitize(cell.text) for cell in row.cells]
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
    """Clean a table cell: convert <br> to newlines, strip markdown artifacts.

    Rejoins word-wrap fragments from narrow PDF columns while preserving
    genuine line breaks (bullets, sentence boundaries).
    """
    text = re.sub(r"\u200b", "", text)        # zero-width space
    text = text.strip()
    text = re.sub(r"<br\s*/?>", "\n", text)   # <br> → newline
    # Convert bullet markers to plain dashes
    text = re.sub(r"^[●○]\s*", "- ", text, flags=re.MULTILINE)
    # Rejoin lines that are just word-wrap artifacts from narrow PDF columns
    lines = text.split("\n")
    if len(lines) > 1:
        merged = [lines[0]]
        in_list = False  # after a colon-ended line, keep items separate
        for line in lines[1:]:
            stripped = line.strip()
            if not stripped:
                merged.append("")
                in_list = False
                continue
            # Keep as new line if it starts with a bullet/list marker
            if re.match(r"^[-●○*■□▪▸►•]\s", stripped) or re.match(r"^\d+[.)]\s", stripped):
                merged.append(stripped)
                in_list = False
                continue
            prev = merged[-1].rstrip() if merged else ""
            # After a colon-ended line, keep subsequent items on separate lines
            if in_list:
                merged.append(stripped)
                if stripped.endswith((".", ":", "!", "?", ";")):
                    in_list = stripped.endswith(":")
                continue
            # Join if previous line doesn't end with sentence punctuation
            if prev and not prev.endswith((".", ":", "!", "?", ";")) and not prev.endswith("  "):
                # Detect mid-word/identifier breaks: both fragments have no spaces
                # and at least one contains identifier chars like _ or .
                prev_token = prev.split("\n")[-1].strip()
                prev_is_token = " " not in prev_token
                cur_is_token = " " not in stripped
                has_id_char = ("_" in prev_token or "_" in stripped
                               or "." in prev_token or "." in stripped)
                if (prev_is_token and cur_is_token and has_id_char
                        and prev[-1].isalnum() and stripped[0].isalnum()):
                    merged[-1] = prev + stripped
                else:
                    merged[-1] = prev + " " + stripped
            else:
                merged.append(stripped)
                if prev.endswith(":"):
                    in_list = True
        text = "\n".join(merged)
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
    """Check if a line is a markdown table separator (|---|---|).

    Each cell must be only dashes with optional leading/trailing colons
    (for alignment), e.g. |---|, |:---|, |:---:|, |---:|.
    This avoids false-positives on data rows containing dashes or spaces.
    """
    stripped = line.strip()
    if not stripped.startswith("|") or not stripped.endswith("|"):
        return False
    # Remove outer pipes, split into cells
    inner = stripped[1:-1]
    if not inner:
        return False
    cells = inner.split("|")
    # Every cell must match the separator pattern: optional colons around 3+ dashes
    # (markdown spec requires at least 3 dashes per cell)
    return all(re.match(r"^\s*:?-{3,}:?\s*$", cell) for cell in cells)


def _table_to_list(lines: list[str]) -> str:
    """Convert parsed table lines into a nested bullet list."""
    if not lines:
        return ""

    # Strip leading whitespace from each line (handle indented tables)
    lines = [l.strip() for l in lines]

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
            re.match(r"^(Col\s*\d+|)$", h, re.IGNORECASE)
            for h in headers
        )

        data_start = sep_idx + 1
        # Collect any rows above the header as data
        pre_header_rows = []
        if sep_idx > 1:
            for i in range(0, sep_idx - 1):
                if lines[i].startswith("|") and not _is_separator(lines[i]):
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
        # Header-only table or empty — return just the header names as a note
        if headers:
            return "- " + ", ".join(h for h in headers if h) + "\n"
        return ""

    # Determine real column names
    if generic_headers or not headers:
        # Use Col1, Col2, ... as fallback
        max_cols = max(len(r) for r in data_rows) if data_rows else 1
        headers = [f"Col {i+1}" for i in range(max_cols)]

    # Merge continuation rows: if a row has empty first cell, append its
    # content to the previous row's cells (common in PDF tables with merged cells)
    merged_rows = []
    for row in data_rows:
        first_cell = row[0].strip() if row else ""
        if not first_cell and merged_rows:
            # Continuation of previous row — merge cell contents
            prev = merged_rows[-1]
            for j in range(len(row)):
                val = row[j].strip() if j < len(row) else ""
                if not val:
                    continue
                if j < len(prev) and prev[j].strip():
                    prev[j] = prev[j].rstrip() + "\n" + val
                elif j < len(prev):
                    prev[j] = val
                else:
                    prev.extend([""] * (j - len(prev) + 1))
                    prev[j] = val
        else:
            merged_rows.append(list(row))
    data_rows = merged_rows

    result_lines = []
    for i, row in enumerate(data_rows):
        # Detect merged rows: all non-empty cells identical → section heading
        non_empty = [c.strip() for c in row if c.strip()]
        unique_vals = set(non_empty)
        if len(unique_vals) == 1 and len(non_empty) > 1:
            result_lines.append(f"\n### {non_empty[0]}\n")
            continue

        # Use first cell as label; if empty, try second cell; last resort "Row N"
        label = row[0].split("\n")[0].strip() if row and row[0].strip() else ""
        if not label and len(row) > 1 and row[1].strip():
            label = row[1].split("\n")[0].strip()[:80]
        if not label:
            label = f"Row {i+1}"
        result_lines.append(f"- **{label}**:")
        for j, header in enumerate(headers):
            value = row[j] if j < len(row) else ""
            if not value:
                continue
            # Skip only if value is identical to label (avoid repeating the row header)
            if j == 0 and value.split("\n")[0].strip() == label:
                # Still emit remaining lines if multi-line
                extra_lines = value.split("\n")[1:]
                extra_lines = [vl.strip() for vl in extra_lines if vl.strip()]
                if extra_lines:
                    result_lines.append(f"  - {header}:")
                    for vl in extra_lines:
                        result_lines.append(f"    - {vl}")
                continue
            # If label came from this column (j==1 fallback), skip repeating it
            if j == 1 and not row[0].strip() and value.split("\n")[0].strip()[:80] == label:
                extra_lines = [vl.strip() for vl in value.split("\n")[1:] if vl.strip()]
                if extra_lines:
                    result_lines.append(f"  - {header}:")
                    for vl in extra_lines:
                        result_lines.append(f"    - {vl}")
                continue
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

    Only handles the empty-cell case: |Cell1||Cell3| → |Cell1| |Cell3|
    For joined rows (separator||data), we try to split them into separate lines,
    but only when both halves look like complete pipe-table rows.
    """
    lines = markdown.split("\n")
    result = []
    for line in lines:
        stripped = line.strip()
        if "||" not in stripped or not stripped.startswith("|"):
            result.append(line)
            continue

        # Try to detect joined rows: |---|---||Data|Data|
        # Each half must start and end with | and look like a full row
        parts = stripped.split("||")
        if len(parts) == 2:
            left, right = parts
            left = left.strip()
            right = right.strip()
            # Ensure both halves are well-formed pipe rows
            if (left.startswith("|") and left.endswith("|") and
                    right and not right.startswith("|")):
                right = "|" + right
                if not right.endswith("|"):
                    right = right + "|"
                # Only split if one half is a separator
                if _is_separator(left) or _is_separator(right):
                    result.extend([left, right])
                    continue

        # Default: treat || as empty cells
        result.append(re.sub(r"\|\|", "| |", line))
    return "\n".join(result)


def _tables_to_lists(markdown: str) -> tuple:
    """Find markdown pipe-tables and convert them to nested bullet lists.

    Handles:
    - Standard tables (header + separator + data)
    - Tables with <br> in cells (from PDF extraction)
    - Tables split across pages (continuation with new separator)
    - Headerless/generic-header tables
    - Double-pipe || artifacts (empty cells)
    - CRLF line endings
    - Indented tables (leading whitespace before |)

    Returns (transformed_markdown, tables_found_count).
    """
    # Normalize line endings to LF
    markdown = markdown.replace("\r\n", "\n").replace("\r", "\n")

    # Pre-process: normalize || to | | before table detection
    markdown = _normalize_double_pipes(markdown)

    # Pattern: any block of consecutive pipe-rows (possibly with separator lines)
    # A table is 2+ consecutive lines starting with | (with optional leading whitespace)
    table_pattern = re.compile(
        r"((?:^[ \t]*\|.+\|[ \t]*$\n?){2,})",
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

    # Attach table titles to their content: collapse blank lines between
    # a line that looks like a table caption and the bullet-list table body.
    # This prevents chunking from splitting the title from the table data.
    transformed = re.sub(
        r"(_[^_\n]+_)\s*\n\n+(- \*\*)",
        r"\1\n\2",
        transformed,
    )

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
        return "[Image: visual content]"

    transformed = img_pattern.sub(_replace_img, markdown)

    # pymupdf4llm placeholders: **==> picture [WxH] intentionally omitted <==**
    pymupdf_pattern = re.compile(r"\*?\*?=+>.*?picture\s*\[.*?\].*?omitted.*?<==\*?\*?")

    def _replace_pymupdf(match):
        nonlocal images_found
        images_found += 1
        return "[Image: visual content]"

    transformed = pymupdf_pattern.sub(_replace_pymupdf, transformed)
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
    # Strip stray DOCX tab/sheet names (e.g. "Thẻ 1", "Thẻ 2")
    final_markdown = re.sub(r"^\s*Thẻ\s+\d+\s*$", "", final_markdown, flags=re.MULTILINE)

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
