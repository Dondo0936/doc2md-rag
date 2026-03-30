import sys
import os
import pytest

# Ensure doc2md-rag is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


SAMPLE_MARKDOWN = """\
# Introduction

This is a sample document for testing the RAG pipeline.

## Section One

The quick brown fox jumps over the lazy dog. This sentence is used for testing purposes.
Another sentence here with different content about machine learning and neural networks.

## Section Two

| Name | Age | City |
| --- | --- | --- |
| Alice | 30 | NYC |
| Bob | 25 | LA |

Some text after the table. It discusses database optimization and query performance.

## Section Three

Final paragraph with unique terms like photosynthesis and mitochondria.
"""

SAMPLE_MARKDOWN_NO_TABLES = """\
# Document

This is a simple document with no tables at all.

Paragraph one has some content about artificial intelligence.

Paragraph two discusses natural language processing.
"""

SAMPLE_TABLE_DOUBLE_PIPES = """\
|Header1||Header3|
|---|---|---|
|A||C|
"""

SAMPLE_MARKDOWN_WITH_IMAGES = """\
# Document

Some text here.

![Chart of revenue](data:image/png;base64,iVBOR)

More text.

![](https://example.com/image.png)
"""


class MockLLMClient:
    """Mock LLM client for testing image description."""

    class _Chat:
        class _Completions:
            def create(self, messages=None, **kwargs):
                from claude_wrapper import CompletionResponse, Choice, Message
                return CompletionResponse(
                    choices=[Choice(message=Message(content="A test image description"))]
                )

        completions = _Completions()

    chat = _Chat()
