"""Parser service configuration (env-overridable)."""

from __future__ import annotations

import os
from functools import lru_cache


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    return int(raw)


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    return float(raw)


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


class Settings:
    """Runtime settings for the parser API."""

    def __init__(self) -> None:
        self.parser_max_file_mb: int = _env_int("PARSER_MAX_FILE_MB", 100)
        self.parser_sync_max_mb: int = _env_int("PARSER_SYNC_MAX_MB", 10)
        self.parser_fetch_timeout: float = _env_float("PARSER_FETCH_TIMEOUT", 30.0)
        self.parser_fetch_max_mb: int = _env_int("PARSER_FETCH_MAX_MB", 25)
        self.parser_fetch_max_redirects: int = _env_int("PARSER_FETCH_MAX_REDIRECTS", 5)
        self.quality_min_chars: int = _env_int("PARSER_QUALITY_MIN_CHARS", 10)
        self.enable_image_desc: bool = _env_bool("ENABLE_IMAGE_DESC", False)
        self.anthropic_api_key: str | None = os.environ.get("ANTHROPIC_API_KEY") or None
        cors = os.environ.get("PARSER_CORS_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000")
        self.cors_origins: list[str] = [o.strip() for o in cors.split(",") if o.strip()]
        self.docling_warmed_up: bool = False

    @property
    def max_file_bytes(self) -> int:
        return self.parser_max_file_mb * 1024 * 1024

    @property
    def sync_max_bytes(self) -> int:
        return self.parser_sync_max_mb * 1024 * 1024

    @property
    def fetch_max_bytes(self) -> int:
        return self.parser_fetch_max_mb * 1024 * 1024


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
