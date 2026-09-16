"""End-to-end walkthrough: the loop an engineer actually plugs corpus_truth into.

Mirrors the README so the documented flow stays honest.
"""

import json

from conftest import policy_v1, policy_v2

from corpus_truth import ImageChunk, InMemoryIndex, Ledger, TableChunk, TextChunk


class TinyVectorStore:
    """A stand-in for FAISS/Chroma/Pinecone: id -> (vector, payload), plus naive search."""

    def __init__(self):
        self.rows: dict[str, tuple[list[float], dict]] = {}
        self.embed_calls: list[str] = []

    def embed(self, text: str) -> list[float]:
        self.embed_calls.append(text)
        return [float(len(text))]

    def add(self, postings):
        for posting in postings:
            self.rows[posting.id] = (self.embed(posting.embed_text()), posting.to_index_record())

    def remove(self, posting_ids):
        for posting_id in posting_ids:
            self.rows.pop(posting_id, None)

    def list_ids(self):
        return list(self.rows)

    def search(self, term: str) -> list[dict]:
        return [payload for _, payload in self.rows.values() if term in str(payload)]


def test_two_revisions_through_an_existing_embed_and_upsert_loop(tmp_path):
    store = TinyVectorStore()

    with Ledger.open(tmp_path / ".corpus") as ledger:
        plan = ledger.commit("policy", policy_v1())
        store.add(plan.add)
        store.remove(plan.remove)
        assert len(store.embed_calls) == 3

        # Same content again: nothing to embed, nothing to write.
        noop = ledger.commit("policy", policy_v1())
        store.add(noop.add)
        store.remove(noop.remove)
        assert len(store.embed_calls) == 3

        # One slot edited: only that chunk is re-embedded.
        edited = ledger.commit("policy", [TextChunk("Refunds take 14 days.", key="refund"), *policy_v1()[1:]])
        store.add(edited.add)
        store.remove(edited.remove)

        assert store.embed_calls[-1] == "Refunds take 14 days."
        assert len(store.embed_calls) == 4
        assert store.search("30 days") == []  # the ghost cannot retrieve
        assert len(store.search("14 days")) == 1
        assert store.search("14 days")[0]["rev"] == 2
        assert ledger.reconcile(store).is_clean

        prompt = ledger.diff_prompt("policy")
        assert prompt.startswith("Document policy is now rev 2 (was 1).")
        assert '"30 days." \u2192 "14 days."' in prompt
        assert prompt.endswith("2 chunks unchanged. Current revision is 2.")


def test_plan_json_matches_the_documented_shape(tmp_path):
    with Ledger.open(":memory:") as ledger:
        first = ledger.commit("policy", policy_v1()).to_dict()

        assert first["doc_id"] == "policy"
        assert first["rev"] == 1
        assert first["parent_rev"] is None
        assert first["remove"] == []
        assert [record["kind"] for record in first["add"]] == ["text", "table", "image"]

        second = ledger.commit("policy", policy_v2()).to_dict()

        assert (second["rev"], second["parent_rev"]) == (2, 1)
        assert len(second["add"]) == 3
        assert sorted(second["remove"]) == sorted(record["id"] for record in first["add"])
        assert second["remap"] == {
            record["id"]: next(new["id"] for new in second["add"] if new["key"] == record["key"])
            for record in first["add"]
        }

        diff = ledger.diff("policy").to_dict()
        assert (diff["from_rev"], diff["to_rev"]) == (1, 2)
        assert [change["key"] for change in diff["changed"]] == ["refund", "pricing", "fig1"]
        assert diff["changed"][1]["cells"] == [{"row": 1, "col": 1, "ref": "B2", "from": "9", "to": "12"}]


def test_every_returned_object_serializes_to_json(tmp_path):
    index = InMemoryIndex()
    with Ledger.open(":memory:") as ledger:
        index.apply(ledger.commit("policy", policy_v1()))
        plan = ledger.commit("policy", policy_v2())
        index.apply(plan)

        for payload in (plan.to_dict(), ledger.diff("policy").to_dict(), ledger.reconcile(index).to_dict()):
            assert json.loads(json.dumps(payload)) == payload


def test_a_hash_only_corpus_still_versions_correctly(tmp_path):
    """Minimum viable input: a table hash and an image digest, no rows, no captions."""
    index = InMemoryIndex()
    with Ledger.open(tmp_path / ".corpus") as ledger:
        index.apply(
            ledger.commit(
                "scan",
                [TableChunk(hash="csv-digest-1", key="t1"), ImageChunk("aaa", key="i1")],
            )
        )
        plan = ledger.commit(
            "scan",
            [TableChunk(hash="csv-digest-2", key="t1"), ImageChunk("aaa", key="i1")],
        )
        index.apply(plan)

        assert len(plan.add) == 1 and len(plan.remove) == 1
        prompt = ledger.diff_prompt("scan")
        assert "table/t1: table changed (rows not stored, so no cell detail)" in prompt
        assert "1 chunk unchanged." in prompt
        assert ledger.reconcile(index).is_clean
