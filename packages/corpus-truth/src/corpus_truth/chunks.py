"""Typed chunks: what you hand us.

We hash chunks and record them. We never open a PDF, a PNG, or run a captioner --
your parser produces these objects and we version them.

Two hashes are derived per chunk:

* ``asset_hash`` -- the payload that decides "is this the same content?"
  (the text, the table cells, the image bytes digest).
* ``content_hash`` -- asset plus describing fields (uri, caption, description,
  metadata). A change here means the index record is stale even though the asset
  is the same, which is how ``diff()`` can report ``metadata_changed``.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from functools import cached_property
from typing import Any, ClassVar, Literal

from .errors import InvalidChunkError

ChunkKind = Literal["text", "table", "image", "other"]

#: Keys corpus_truth writes onto index records itself. Chunk metadata may not use them.
RESERVED_METADATA_KEYS = frozenset(
    {
        "id",
        "doc_id",
        "key",
        "kind",
        "rev",
        "is_current",
        "valid_from",
        "valid_to",
        "parent_rev",
        "replaces",
        "content_hash",
        "asset_hash",
        "text",
        "rows",
        "shape",
        "hash",
        "sha256",
        "uri",
        "caption",
        "description",
        "mime",
    }
)


def canonical_json(value: Any) -> str:
    """Deterministic JSON for hashing: sorted keys, no whitespace, no NaN."""
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise InvalidChunkError(f"value is not JSON-serializable: {exc}") from exc


def sha256_hex(data: bytes | str) -> str:
    if isinstance(data, str):
        data = data.encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def normalize_cell(value: Any) -> str:
    """Collapse a table cell to a comparable string.

    Whitespace and numeric formatting noise from a parser must not look like an
    edit, so ``" 9 "``, ``"9"`` and ``9`` all normalize to ``"9"``.
    """
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    return " ".join(str(value).split())


def normalize_rows(rows: Sequence[Any]) -> list[list[str]]:
    """Normalize list-of-lists or list-of-dicts rows into a rectangular grid of strings.

    Dict rows gain a header row built from the ordered union of their keys.
    """
    materialized = list(rows)
    if materialized and all(isinstance(row, Mapping) for row in materialized):
        header: list[str] = []
        for row in materialized:
            for key in row:
                if key not in header:
                    header.append(key)
        grid = [[normalize_cell(key) for key in header]]
        grid.extend([[normalize_cell(row.get(key)) for key in header] for row in materialized])
        return grid

    grid = []
    for row in materialized:
        if isinstance(row, (str, bytes)) or not isinstance(row, Sequence):
            raise InvalidChunkError("table rows must be sequences of cells or mappings")
        grid.append([normalize_cell(cell) for cell in row])
    width = max((len(row) for row in grid), default=0)
    for row in grid:
        row.extend([""] * (width - len(row)))
    return grid


class Chunk:
    """Base behaviour shared by the four chunk kinds.

    Subclasses are frozen dataclasses; they declare their payload fields first so
    the common case reads naturally, e.g. ``TextChunk("refunds take 30 days")``.
    """

    kind: ClassVar[ChunkKind]

    key: str | None
    id: str | None
    metadata: Mapping[str, Any]

    def payload(self) -> dict[str, Any]:  # pragma: no cover - overridden
        raise NotImplementedError

    def asset_hash(self) -> str:  # pragma: no cover - overridden
        raise NotImplementedError

    def _validate_common(self) -> None:
        if self.key is not None and (not isinstance(self.key, str) or not self.key.strip()):
            raise InvalidChunkError("key must be a non-empty string when provided")
        if self.id is not None and (not isinstance(self.id, str) or not self.id.strip()):
            raise InvalidChunkError("id must be a non-empty string when provided")
        metadata = dict(self.metadata or {})
        bad = sorted(key for key in metadata if key in RESERVED_METADATA_KEYS)
        if bad:
            raise InvalidChunkError(f"metadata keys are reserved by corpus_truth: {', '.join(bad)}")
        canonical_json(metadata)
        object.__setattr__(self, "metadata", metadata)

    @cached_property
    def content_hash(self) -> str:
        """Identity used to decide whether the index record must be rewritten."""
        return sha256_hex(
            canonical_json({"kind": self.kind, "payload": self.payload(), "metadata": dict(self.metadata)})
        )

    @property
    def slot_key(self) -> str:
        """Stable slot this chunk occupies in the document.

        An explicit ``key`` gives you update semantics (``uuid-x -> uuid-y``).
        Without one we fall back to hash identity, so any edit reads as a removal
        plus an addition.
        """
        if self.key is not None:
            return self.key
        return f"{self.kind}:{self.content_hash[:16]}"


@dataclass(frozen=True)
class TextChunk(Chunk):
    """A span of text. Hashed exactly as given -- we do not normalize whitespace."""

    text: str
    key: str | None = field(default=None, kw_only=True)
    id: str | None = field(default=None, kw_only=True)
    metadata: Mapping[str, Any] = field(default_factory=dict, kw_only=True)

    kind: ClassVar[ChunkKind] = "text"

    def __post_init__(self) -> None:
        if not isinstance(self.text, str):
            raise InvalidChunkError("TextChunk.text must be a string")
        self._validate_common()

    def payload(self) -> dict[str, Any]:
        return {"text": self.text}

    def asset_hash(self) -> str:
        return sha256_hex("text:" + self.text)


@dataclass(frozen=True)
class TableChunk(Chunk):
    """A table.

    Pass ``rows`` to get cell-level diffs, or just ``hash`` (any string you
    already computed -- a digest, a CSV dump, a markdown rendering) if you only
    need add/remove/skip. Passing both is rejected: two competing change
    detectors would silently disagree.
    """

    rows: Sequence[Any] | None = None
    hash: str | None = field(default=None, kw_only=True)
    caption: str | None = field(default=None, kw_only=True)
    key: str | None = field(default=None, kw_only=True)
    id: str | None = field(default=None, kw_only=True)
    metadata: Mapping[str, Any] = field(default_factory=dict, kw_only=True)

    kind: ClassVar[ChunkKind] = "table"

    def __post_init__(self) -> None:
        if self.rows is None and self.hash is None:
            raise InvalidChunkError("TableChunk needs rows or hash")
        if self.rows is not None and self.hash is not None:
            raise InvalidChunkError("TableChunk takes rows or hash, not both")
        if self.hash is not None and (not isinstance(self.hash, str) or not self.hash.strip()):
            raise InvalidChunkError("TableChunk.hash must be a non-empty string")
        if self.rows is not None:
            object.__setattr__(self, "rows", normalize_rows(self.rows))
        self._validate_common()

    @property
    def shape(self) -> tuple[int, int] | None:
        """(rows, cols) when rows were supplied."""
        if self.rows is None:
            return None
        grid = list(self.rows)
        return (len(grid), len(grid[0]) if grid else 0)

    def payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {}
        if self.rows is not None:
            shape = self.shape or (0, 0)
            payload["rows"] = [list(row) for row in self.rows]
            payload["shape"] = [shape[0], shape[1]]
        else:
            payload["hash"] = self.hash
        if self.caption is not None:
            payload["caption"] = self.caption
        return payload

    def asset_hash(self) -> str:
        if self.rows is not None:
            return sha256_hex("table:" + canonical_json([list(row) for row in self.rows]))
        return sha256_hex("table-hash:" + str(self.hash))


@dataclass(frozen=True)
class ImageChunk(Chunk):
    """An image.

    ``sha256`` (or raw ``data`` we hash for you) is the change detector. ``uri``
    lets a vision model look at both revisions; ``description`` from your own
    captioner is what makes a text-only changelog readable. We store neither
    pixels nor bytes.
    """

    sha256: str | None = None
    data: bytes | None = field(default=None, kw_only=True)
    uri: str | None = field(default=None, kw_only=True)
    caption: str | None = field(default=None, kw_only=True)
    description: str | None = field(default=None, kw_only=True)
    key: str | None = field(default=None, kw_only=True)
    id: str | None = field(default=None, kw_only=True)
    metadata: Mapping[str, Any] = field(default_factory=dict, kw_only=True)

    kind: ClassVar[ChunkKind] = "image"

    def __post_init__(self) -> None:
        digest = self.sha256.strip() if isinstance(self.sha256, str) else self.sha256
        if self.data is not None:
            if not isinstance(self.data, (bytes, bytearray)):
                raise InvalidChunkError("ImageChunk.data must be bytes")
            computed = sha256_hex(bytes(self.data))
            if digest and digest.lower() != computed:
                raise InvalidChunkError("ImageChunk.sha256 does not match the supplied data")
            digest = computed
        if not digest:
            raise InvalidChunkError("ImageChunk needs sha256 or data")
        object.__setattr__(self, "sha256", digest)
        object.__setattr__(self, "data", None)  # we version assets, we do not store them
        self._validate_common()

    def payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"sha256": self.sha256}
        for name in ("uri", "caption", "description"):
            value = getattr(self, name)
            if value is not None:
                payload[name] = value
        return payload

    def asset_hash(self) -> str:
        return str(self.sha256)


@dataclass(frozen=True)
class OtherChunk(Chunk):
    """Anything else: audio, an HTML fragment, code, a blob.

    ``mime`` plus a ``hash`` (or ``data`` we hash) is the minimum. Extra JSON
    belongs in ``metadata``; changing it reads as ``metadata_changed``.
    """

    mime: str
    hash: str | None = field(default=None, kw_only=True)
    data: bytes | None = field(default=None, kw_only=True)
    key: str | None = field(default=None, kw_only=True)
    id: str | None = field(default=None, kw_only=True)
    metadata: Mapping[str, Any] = field(default_factory=dict, kw_only=True)

    kind: ClassVar[ChunkKind] = "other"

    def __post_init__(self) -> None:
        if not isinstance(self.mime, str) or not self.mime.strip():
            raise InvalidChunkError("OtherChunk.mime must be a non-empty string")
        digest = self.hash.strip() if isinstance(self.hash, str) else self.hash
        if self.data is not None:
            if not isinstance(self.data, (bytes, bytearray)):
                raise InvalidChunkError("OtherChunk.data must be bytes")
            computed = sha256_hex(bytes(self.data))
            if digest and digest != computed:
                raise InvalidChunkError("OtherChunk.hash does not match the supplied data")
            digest = computed
        if not digest:
            raise InvalidChunkError("OtherChunk needs hash or data")
        object.__setattr__(self, "hash", digest)
        object.__setattr__(self, "data", None)
        self._validate_common()

    def payload(self) -> dict[str, Any]:
        return {"mime": self.mime, "hash": self.hash}

    def asset_hash(self) -> str:
        return str(self.hash)


__all__ = [
    "RESERVED_METADATA_KEYS",
    "Chunk",
    "ChunkKind",
    "ImageChunk",
    "OtherChunk",
    "TableChunk",
    "TextChunk",
    "canonical_json",
    "normalize_cell",
    "normalize_rows",
    "sha256_hex",
]
