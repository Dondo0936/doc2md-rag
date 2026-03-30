# claude_wrapper.py — OpenAI-compatible wrapper for Anthropic API (for MarkItDown)
#
# MarkItDown expects an OpenAI-compatible llm_client with chat.completions.create().
# This module adapts the Anthropic SDK to that interface.

import logging
import os
import re
from dataclasses import dataclass, field
from config import CLAUDE_VISION_MODEL

logger = logging.getLogger(__name__)


@dataclass
class Choice:
    message: "Message"

@dataclass
class Message:
    content: str

@dataclass
class CompletionResponse:
    choices: list = field(default_factory=list)


class _Completions:
    """Mimics openai.chat.completions with Anthropic backend."""

    def __init__(self, anthropic_client):
        self._client = anthropic_client

    def create(self, model=None, messages=None, **kwargs):
        model = model or CLAUDE_VISION_MODEL
        anthropic_messages = []

        for msg in (messages or []):
            role = msg.get("role", "user")
            content = msg.get("content", "")

            if isinstance(content, str):
                anthropic_messages.append({"role": role, "content": content})
            elif isinstance(content, list):
                # Translate OpenAI content blocks to Anthropic format
                blocks = []
                for block in content:
                    if block.get("type") == "text":
                        blocks.append({"type": "text", "text": block["text"]})
                    elif block.get("type") == "image_url":
                        url = block["image_url"]["url"]
                        # Handle base64 data URIs
                        if url.startswith("data:"):
                            match = re.match(r"data:(image/\w+);base64,(.+)", url)
                            if match:
                                media_type = match.group(1)
                                data = match.group(2)
                                blocks.append({
                                    "type": "image",
                                    "source": {
                                        "type": "base64",
                                        "media_type": media_type,
                                        "data": data,
                                    },
                                })
                        else:
                            # URL-based image
                            blocks.append({
                                "type": "image",
                                "source": {"type": "url", "url": url},
                            })
                anthropic_messages.append({"role": role, "content": blocks})

        response = self._client.messages.create(
            model=model,
            max_tokens=1024,
            messages=anthropic_messages,
        )

        text = ""
        if response.content and len(response.content) > 0:
            text = response.content[0].text
        return CompletionResponse(
            choices=[Choice(message=Message(content=text))]
        )


class _Chat:
    """Mimics openai.chat namespace."""

    def __init__(self, anthropic_client):
        self.completions = _Completions(anthropic_client)


class ClaudeLLMClient:
    """Drop-in replacement for OpenAI() client that routes to Anthropic's API.

    Usage:
        client = ClaudeLLMClient(api_key="sk-ant-...")
        md = MarkItDown(llm_client=client, llm_model="claude-sonnet-4-20250514")
    """

    def __init__(self, api_key=None):
        import anthropic
        self._client = anthropic.Anthropic(api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"))
        self.chat = _Chat(self._client)


def get_llm_client(api_key=None):
    """Returns a ClaudeLLMClient if an API key is available, else None."""
    key = api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        return None
    try:
        return ClaudeLLMClient(api_key=key)
    except ImportError:
        logger.warning("anthropic package not installed — Claude Vision disabled")
        return None
    except Exception as e:
        logger.warning("Failed to initialize ClaudeLLMClient: %s", e)
        return None
