"""Exceptions raised by corpus_truth."""

from __future__ import annotations


class CorpusTruthError(Exception):
    """Base class for every error raised by this library."""


class InvalidChunkError(CorpusTruthError):
    """A chunk is missing a required field or carries a non-serializable payload."""


class DuplicateChunkKeyError(CorpusTruthError):
    """Two chunks in the same commit claim the same slot key."""


class DuplicateIdError(CorpusTruthError):
    """A caller-supplied posting id is already recorded in the ledger."""


class DocumentNotFoundError(CorpusTruthError):
    """The ledger has never seen this doc_id."""


class RevisionNotFoundError(CorpusTruthError):
    """The requested revision does not exist for this document."""
