"""SSRF-safe URL fetcher — download, sniff type, return bytes + metadata."""

from __future__ import annotations

import ipaddress
import logging
import mimetypes
import socket
from dataclasses import dataclass
from urllib.parse import urlparse

import httpx

from backend.app.config import Settings, get_settings

logger = logging.getLogger(__name__)

# Magic-byte sniffing for common document types
_MAGIC: list[tuple[bytes, str, str]] = [
    (b"%PDF", "application/pdf", ".pdf"),
    (b"PK\x03\x04", "application/zip", ".zip"),  # docx/pptx/xlsx are zip
    (b"\x89PNG\r\n\x1a\n", "image/png", ".png"),
    (b"\xff\xd8\xff", "image/jpeg", ".jpg"),
    (b"RIFF", "image/webp", ".webp"),  # need WEBP check
    (b"<!DOCTYPE", "text/html", ".html"),
    (b"<!doctype", "text/html", ".html"),
    (b"<html", "text/html", ".html"),
    (b"<HTML", "text/html", ".html"),
]


class UrlFetchError(ValueError):
    """Raised when a URL cannot be fetched safely."""


@dataclass
class FetchedResource:
    url: str
    final_url: str
    content: bytes
    content_type: str
    extension: str
    filename: str


def _is_private_ip(ip: str) -> bool:
    try:
        addr = ipaddress.ip_address(ip)
    except ValueError:
        return False  # not an IP literal
    return bool(
        addr.is_private
        or addr.is_loopback
        or addr.is_link_local
        or addr.is_reserved
        or addr.is_multicast
        or addr.is_unspecified
    )


def validate_url(url: str) -> str:
    """Validate URL scheme and host; reject SSRF targets."""
    parsed = urlparse(url.strip())
    if parsed.scheme not in {"http", "https"}:
        raise UrlFetchError("Only http and https URLs are allowed")
    if not parsed.hostname:
        raise UrlFetchError("URL must include a hostname")
    host = parsed.hostname.lower()
    if host in {"localhost", "metadata.google.internal"} or host.endswith(".localhost"):
        raise UrlFetchError("Hostname is not allowed")
    # Block literal IPs that are private
    try:
        if _is_private_ip(host):
            raise UrlFetchError("Private/reserved IP addresses are not allowed")
    except UrlFetchError:
        raise
    except Exception:
        pass

    # Resolve DNS and block private results
    try:
        infos = socket.getaddrinfo(host, None)
    except socket.gaierror as exc:
        raise UrlFetchError(f"Could not resolve hostname: {exc}") from exc
    for info in infos:
        ip = info[4][0]
        if _is_private_ip(ip):
            raise UrlFetchError("URL resolves to a private/reserved address")

    return url.strip()


def _sniff(content: bytes, content_type: str | None, url: str) -> tuple[str, str]:
    ct = (content_type or "").split(";")[0].strip().lower()
    path = urlparse(url).path
    ext = ""
    if "." in path.rsplit("/", 1)[-1]:
        ext = "." + path.rsplit(".", 1)[-1].lower()

    head = content[:64]
    for magic, mime, magic_ext in _MAGIC:
        if head.startswith(magic) or magic in head[:16]:
            if magic == b"RIFF" and b"WEBP" not in content[:16]:
                continue
            if magic == b"PK\x03\x04":
                # Distinguish office docs by URL extension / content-type
                if ext in {".docx", ".pptx", ".xlsx"}:
                    return {
                        ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                        ".pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
                        ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    }[ext], ext
                if "word" in ct:
                    return "application/vnd.openxmlformats-officedocument.wordprocessingml.document", ".docx"
                if "presentation" in ct or "powerpoint" in ct:
                    return "application/vnd.openxmlformats-officedocument.presentationml.presentation", ".pptx"
                if "sheet" in ct or "excel" in ct:
                    return "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", ".xlsx"
                # default zip-based office — leave as zip unknown
            else:
                return mime, magic_ext if not ext else ext

    if ct and ct != "application/octet-stream":
        guessed_ext = mimetypes.guess_extension(ct) or ext or ""
        return ct, guessed_ext

    if ext:
        guessed = mimetypes.guess_type("x" + ext)[0] or "application/octet-stream"
        return guessed, ext

    # Heuristic: treat as HTML if looks like markup, else plain text
    sample = content[:2000].decode("utf-8", errors="replace")
    if "<html" in sample.lower() or "<body" in sample.lower():
        return "text/html", ".html"
    return "text/plain", ".txt"


def _filename_from_url(url: str, ext: str) -> str:
    name = urlparse(url).path.rsplit("/", 1)[-1] or "download"
    if "." not in name and ext:
        name = name + ext
    return name[:200]


async def fetch_url(url: str, settings: Settings | None = None) -> FetchedResource:
    """Fetch a remote URL with SSRF guards, size cap, and redirect limit."""
    settings = settings or get_settings()
    url = validate_url(url)

    timeout = httpx.Timeout(settings.parser_fetch_timeout)
    max_bytes = settings.fetch_max_bytes
    redirects = 0

    async with httpx.AsyncClient(
        timeout=timeout,
        follow_redirects=False,
        headers={"User-Agent": "doc2md-parser/0.1"},
    ) as client:
        current = url
        while True:
            validate_url(current)
            response = await client.get(current)
            if response.is_redirect:
                redirects += 1
                if redirects > settings.parser_fetch_max_redirects:
                    raise UrlFetchError("Too many redirects")
                location = response.headers.get("location")
                if not location:
                    raise UrlFetchError("Redirect without Location header")
                # Resolve relative redirects
                current = str(httpx.URL(current).join(location))
                continue

            if response.status_code >= 400:
                raise UrlFetchError(f"HTTP {response.status_code} fetching URL")

            # Stream-ish size check via content-length then body
            cl = response.headers.get("content-length")
            if cl and int(cl) > max_bytes:
                raise UrlFetchError(
                    f"Remote file too large ({int(cl) / 1024 / 1024:.1f}MB). "
                    f"Maximum is {settings.parser_fetch_max_mb}MB."
                )

            content = response.content
            if len(content) > max_bytes:
                raise UrlFetchError(
                    f"Remote file too large ({len(content) / 1024 / 1024:.1f}MB). "
                    f"Maximum is {settings.parser_fetch_max_mb}MB."
                )

            ct_header = response.headers.get("content-type")
            mime, ext = _sniff(content, ct_header, str(response.url))
            filename = _filename_from_url(str(response.url), ext)
            return FetchedResource(
                url=url,
                final_url=str(response.url),
                content=content,
                content_type=mime,
                extension=ext,
                filename=filename,
            )
