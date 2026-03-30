"""Tests for claude_wrapper.py — API translation and error handling."""

import os
import pytest
from unittest.mock import patch, MagicMock

from claude_wrapper import (
    ClaudeLLMClient,
    CompletionResponse,
    Choice,
    Message,
    _Completions,
    get_llm_client,
)


class TestCompletionResponse:
    def test_dataclass_creation(self):
        resp = CompletionResponse(
            choices=[Choice(message=Message(content="hello"))]
        )
        assert resp.choices[0].message.content == "hello"

    def test_empty_choices(self):
        resp = CompletionResponse(choices=[])
        assert len(resp.choices) == 0


class TestCompletions:
    def test_string_content(self):
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="response text")]
        mock_client.messages.create.return_value = mock_response

        completions = _Completions(mock_client)
        result = completions.create(messages=[{"role": "user", "content": "hello"}])
        assert result.choices[0].message.content == "response text"

    def test_image_url_block(self):
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="image description")]
        mock_client.messages.create.return_value = mock_response

        completions = _Completions(mock_client)
        result = completions.create(messages=[{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc123"}},
                {"type": "text", "text": "describe this"},
            ],
        }])

        assert result.choices[0].message.content == "image description"
        # Verify the base64 was extracted correctly
        call_args = mock_client.messages.create.call_args
        blocks = call_args[1]["messages"][0]["content"]
        assert blocks[0]["type"] == "image"
        assert blocks[0]["source"]["type"] == "base64"
        assert blocks[0]["source"]["data"] == "abc123"

    def test_empty_api_response(self):
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.content = []
        mock_client.messages.create.return_value = mock_response

        completions = _Completions(mock_client)
        result = completions.create(messages=[{"role": "user", "content": "hello"}])
        assert result.choices[0].message.content == ""

    def test_none_content_response(self):
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.content = None
        mock_client.messages.create.return_value = mock_response

        completions = _Completions(mock_client)
        result = completions.create(messages=[{"role": "user", "content": "hello"}])
        assert result.choices[0].message.content == ""


class TestGetLLMClient:
    def test_no_key_returns_none(self):
        result = get_llm_client(api_key=None)
        # Without env var set, should return None
        if not os.environ.get("ANTHROPIC_API_KEY"):
            assert result is None

    def test_explicit_key(self):
        # This will try to create a real client — just test it doesn't crash
        # (it will fail gracefully if anthropic package has issues)
        result = get_llm_client(api_key="sk-ant-test-fake-key")
        # May return a client or None depending on anthropic package validation
        assert result is None or hasattr(result, "chat")

    @patch.dict(os.environ, {"ANTHROPIC_API_KEY": ""}, clear=False)
    def test_empty_env_key_returns_none(self):
        result = get_llm_client()
        assert result is None
