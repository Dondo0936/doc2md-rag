"""Chunk construction, canonicalization and hashing."""

import pytest
from conftest import policy_v1

from corpus_truth import ImageChunk, InvalidChunkError, OtherChunk, Posting, TableChunk, TextChunk
from corpus_truth.chunks import normalize_rows, sha256_hex


def test_text_chunk_hashes_text_exactly():
    assert TextChunk("a").content_hash != TextChunk("b").content_hash
    assert TextChunk("a").content_hash == TextChunk("a").content_hash
    # No hidden whitespace normalization: text is hashed as given.
    assert TextChunk("a ").content_hash != TextChunk("a").content_hash


def test_explicit_key_is_the_slot_and_hash_identity_is_the_fallback():
    keyed_old = TextChunk("old", key="refund")
    keyed_new = TextChunk("new", key="refund")
    assert keyed_old.slot_key == keyed_new.slot_key == "refund"

    unkeyed_old = TextChunk("old")
    unkeyed_new = TextChunk("new")
    assert unkeyed_old.slot_key != unkeyed_new.slot_key
    assert unkeyed_old.slot_key.startswith("text:")
    assert TextChunk("old").slot_key == unkeyed_old.slot_key


def test_metadata_participates_in_content_hash_but_not_asset_hash():
    plain = TextChunk("same", key="k")
    tagged = TextChunk("same", key="k", metadata={"source": "policy.pdf"})
    assert plain.asset_hash() == tagged.asset_hash()
    assert plain.content_hash != tagged.content_hash


def test_reserved_metadata_keys_are_rejected():
    with pytest.raises(InvalidChunkError, match="reserved"):
        TextChunk("x", metadata={"rev": 4})
    with pytest.raises(InvalidChunkError, match="reserved"):
        TextChunk("x", metadata={"is_current": True, "doc_id": "policy"})


def test_non_serializable_metadata_is_rejected():
    with pytest.raises(InvalidChunkError, match="JSON-serializable"):
        TextChunk("x", metadata={"when": object()})


@pytest.mark.parametrize(
    "rows, expected",
    [
        ([[" A ", 9]], [["A", "9"]]),
        ([["A\nB", None]], [["A B", ""]]),
        ([["A"], ["B", "C"]], [["A", ""], ["B", "C"]]),
        ([{"item": "A", "price": 9}], [["item", "price"], ["A", "9"]]),
        ([{"item": "A"}, {"price": 9}], [["item", "price"], ["A", ""], ["", "9"]]),
    ],
)
def test_table_rows_are_normalized_into_a_string_grid(rows, expected):
    assert normalize_rows(rows) == expected


def test_table_layout_noise_does_not_create_a_revision():
    tidy = TableChunk([["item", "price"], ["A", "9"]], key="pricing")
    noisy = TableChunk([["  item", "price  "], ["A", 9]], key="pricing")
    assert tidy.content_hash == noisy.content_hash
    assert tidy.shape == (2, 2)


def test_table_needs_rows_or_hash_but_not_both():
    with pytest.raises(InvalidChunkError, match="rows or hash"):
        TableChunk(key="pricing")
    with pytest.raises(InvalidChunkError, match="not both"):
        TableChunk([["a"]], hash="h1", key="pricing")


def test_table_hash_only_chunk_is_versionable():
    chunk = TableChunk(hash="csv-digest-1", key="pricing")
    assert chunk.payload() == {"hash": "csv-digest-1"}
    assert chunk.shape is None
    assert chunk.content_hash != TableChunk(hash="csv-digest-2", key="pricing").content_hash


def test_image_accepts_bytes_and_never_keeps_them():
    chunk = ImageChunk(data=b"pixels", key="fig1")
    assert chunk.sha256 == sha256_hex(b"pixels")
    assert chunk.data is None


def test_image_rejects_a_digest_that_contradicts_the_bytes():
    with pytest.raises(InvalidChunkError, match="does not match"):
        ImageChunk("deadbeef", data=b"pixels", key="fig1")


def test_image_requires_a_change_detector():
    with pytest.raises(InvalidChunkError, match="sha256 or data"):
        ImageChunk(key="fig1")


def test_image_describing_fields_change_content_hash_only():
    bare = ImageChunk("aaa", key="fig1")
    described = ImageChunk("aaa", description="Three connectors.", key="fig1")
    assert bare.asset_hash() == described.asset_hash() == "aaa"
    assert bare.content_hash != described.content_hash


def test_other_chunk_requires_mime_and_a_hash():
    with pytest.raises(InvalidChunkError, match="mime"):
        OtherChunk("", hash="h")
    with pytest.raises(InvalidChunkError, match="hash or data"):
        OtherChunk("audio/wav")
    chunk = OtherChunk("audio/wav", data=b"riff", key="intro")
    assert chunk.payload() == {"mime": "audio/wav", "hash": sha256_hex(b"riff")}


def test_embed_text_gives_each_kind_something_to_embed():
    postings = [
        Posting(
            id=f"id-{position}",
            doc_id="policy",
            chunk_key=chunk.slot_key,
            kind=chunk.kind,
            rev=1,
            valid_from=1,
            valid_to=None,
            parent_rev=None,
            replaces=None,
            content_hash=chunk.content_hash,
            asset_hash=chunk.asset_hash(),
            payload=chunk.payload(),
        )
        for position, chunk in enumerate(policy_v1())
    ]
    text, table, image = postings
    assert text.embed_text() == "Refunds take 30 days."
    assert table.embed_text() == "item | price\nA | 9"
    assert image.embed_text() == "Wiring diagram with 3 connectors."
    assert OtherChunk("audio/wav", hash="h").payload()["mime"] == "audio/wav"
