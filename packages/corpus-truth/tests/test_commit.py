"""commit(): revisions, hash skip, id mapping and apply order."""

import pytest
from conftest import ids_by_key, policy_v1, policy_v2

from corpus_truth import (
    DuplicateChunkKeyError,
    DuplicateIdError,
    InvalidChunkError,
    TableChunk,
    TextChunk,
    apply_plan,
)


def test_first_commit_adds_everything_and_removes_nothing(ledger):
    plan = ledger.commit("policy", policy_v1())

    assert plan.rev == 1
    assert plan.parent_rev is None
    assert plan.remove == ()
    assert plan.remap == {}
    assert plan.tombstone is False
    assert not plan.is_noop
    assert [posting.chunk_key for posting in plan.add] == ["refund", "pricing", "fig1"]
    assert [posting.kind for posting in plan.add] == ["text", "table", "image"]
    assert all(posting.rev == 1 and posting.is_current for posting in plan.add)
    assert all(posting.parent_rev is None and posting.replaces is None for posting in plan.add)


def test_index_records_carry_kind_rev_and_current_flag(ledger):
    plan = ledger.commit("policy", policy_v1())
    records = {record["key"]: record for record in plan.to_dict()["add"]}

    assert records["refund"]["text"] == "Refunds take 30 days."
    assert records["pricing"]["shape"] == [2, 2]
    assert records["fig1"]["caption"] == "Fig 1"
    assert records["fig1"]["description"] == "Wiring diagram with 3 connectors."
    for record in records.values():
        assert record["doc_id"] == "policy"
        assert record["rev"] == 1
        assert record["is_current"] is True
        assert record["valid_to"] is None


def test_chunk_metadata_is_passed_through_to_the_index_record(ledger):
    plan = ledger.commit("policy", [TextChunk("hi", key="k", metadata={"source": "policy.pdf", "page": 2})])
    record = plan.add[0].to_index_record()
    assert record["source"] == "policy.pdf"
    assert record["page"] == 2


def test_identical_recommit_is_a_noop_and_creates_no_revision(ledger):
    first = ledger.commit("policy", policy_v1())
    again = ledger.commit("policy", policy_v1())

    assert again.is_noop
    assert again.rev == 1
    assert again.add == ()
    assert again.remove == ()
    assert set(again.unchanged) == {posting.id for posting in first.add}
    assert [revision.rev for revision in ledger.revisions("policy")] == [1]


def test_unchanged_chunks_keep_their_id_and_stay_out_of_the_plan(ledger):
    first = ledger.commit("policy", policy_v1())
    before = ids_by_key(first.add)

    plan = ledger.commit(
        "policy",
        [
            TextChunk("Refunds take 14 days.", key="refund"),
            *policy_v1()[1:],
        ],
    )

    assert [posting.chunk_key for posting in plan.add] == ["refund"]
    assert plan.remove == (before["refund"],)
    assert set(plan.unchanged) == {before["pricing"], before["fig1"]}
    assert ids_by_key(ledger.current("policy"))["pricing"] == before["pricing"]


def test_update_records_the_old_to_new_id_map(ledger):
    first = ledger.commit("policy", policy_v1())
    before = ids_by_key(first.add)
    plan = ledger.commit("policy", policy_v2())
    after = ids_by_key(plan.add)

    assert plan.rev == 2
    assert plan.parent_rev == 1
    assert plan.remap == {before[key]: after[key] for key in ("refund", "pricing", "fig1")}
    assert set(plan.remove) == set(before.values())
    assert all(posting.replaces == before[posting.chunk_key] for posting in plan.add)
    assert all(posting.parent_rev == 1 and posting.rev == 2 for posting in plan.add)


def test_lineage_persists_the_uuid_x_to_uuid_y_trail(ledger):
    first = ledger.commit("policy", policy_v1())
    second = ledger.commit("policy", policy_v2())

    lineage = ledger.lineage("policy", "refund")
    assert [posting.id for posting in lineage] == [
        ids_by_key(first.add)["refund"],
        ids_by_key(second.add)["refund"],
    ]
    assert lineage[0].valid_to == 2 and not lineage[0].is_current
    assert lineage[1].is_current


def test_a_dropped_slot_is_removed_from_the_index(ledger):
    first = ledger.commit("policy", policy_v1())
    before = ids_by_key(first.add)

    plan = ledger.commit("policy", policy_v1()[:2])

    assert plan.add == ()
    assert plan.remove == (before["fig1"],)
    assert "fig1" not in ids_by_key(ledger.current("policy"))


def test_metadata_only_change_rewrites_the_posting(ledger):
    ledger.commit("policy", [TextChunk("body", key="k")])
    plan = ledger.commit("policy", [TextChunk("body", key="k", metadata={"source": "v2.pdf"})])

    assert plan.rev == 2
    assert len(plan.add) == 1
    assert plan.add[0].asset_hash == ledger.lineage("policy", "k")[0].asset_hash
    assert plan.add[0].content_hash != ledger.lineage("policy", "k")[0].content_hash


def test_apply_order_is_add_then_remove(ledger, recording_index):
    apply_plan(recording_index, ledger.commit("policy", policy_v1()))
    recording_index.calls.clear()

    plan = ledger.commit("policy", policy_v2())
    apply_plan(recording_index, plan)

    assert [call[0] for call in recording_index.calls] == ["add", "remove"]
    assert recording_index.calls[0][1] == tuple(posting.id for posting in plan.add)
    assert recording_index.calls[1][1] == plan.remove
    assert set(recording_index.list_ids()) == {posting.id for posting in plan.add}


def test_a_noop_plan_touches_the_index_at_all(ledger, recording_index):
    apply_plan(recording_index, ledger.commit("policy", policy_v1()))
    recording_index.calls.clear()

    apply_plan(recording_index, ledger.commit("policy", policy_v1()))

    assert recording_index.calls == []


def test_caller_supplied_ids_are_used_verbatim(ledger):
    plan = ledger.commit("policy", [TextChunk("body", key="k", id="uuid-1")])
    assert plan.add[0].id == "uuid-1"

    updated = ledger.commit("policy", [TextChunk("new body", key="k", id="uuid-2")])
    assert updated.remap == {"uuid-1": "uuid-2"}


def test_an_unchanged_slot_keeps_its_id_even_if_a_new_one_is_offered(ledger):
    ledger.commit("policy", [TextChunk("body", key="k", id="uuid-1")])
    plan = ledger.commit("policy", [TextChunk("body", key="k", id="uuid-2")])

    assert plan.is_noop
    assert plan.unchanged == ("uuid-1",)
    assert [posting.id for posting in ledger.current("policy")] == ["uuid-1"]


def test_reusing_an_id_for_new_content_is_refused(ledger):
    ledger.commit("policy", [TextChunk("body", key="k", id="uuid-1")])
    with pytest.raises(DuplicateIdError, match="already recorded"):
        ledger.commit("policy", [TextChunk("new body", key="k", id="uuid-1")])


def test_ids_repeated_inside_one_commit_are_refused(ledger):
    with pytest.raises(DuplicateIdError, match="repeated within one commit"):
        ledger.commit(
            "policy",
            [TextChunk("a", key="a", id="uuid-1"), TextChunk("b", key="b", id="uuid-1")],
        )


def test_two_chunks_claiming_one_slot_are_refused(ledger):
    with pytest.raises(DuplicateChunkKeyError, match="twice in one commit"):
        ledger.commit("policy", [TextChunk("a", key="dup"), TableChunk([["b"]], key="dup")])


def test_non_chunk_input_is_refused(ledger):
    with pytest.raises(InvalidChunkError, match="expected a TextChunk"):
        ledger.commit("policy", ["just a string"])


def test_empty_doc_id_is_refused(ledger):
    with pytest.raises(ValueError, match="doc_id"):
        ledger.commit("  ", policy_v1())


def test_a_document_with_no_chunks_still_gets_revision_one(ledger):
    plan = ledger.commit("empty", [])

    assert plan.rev == 1
    assert plan.is_noop
    assert ledger.current("empty") == []
    assert ledger.head("empty").rev == 1


def test_documents_are_independent(ledger):
    ledger.commit("policy", policy_v1())
    ledger.commit("handbook", [TextChunk("Welcome.", key="intro")])

    assert ledger.documents() == ["handbook", "policy"]
    assert len(ledger.current("policy")) == 3
    assert len(ledger.current("handbook")) == 1


def test_revision_metadata_is_recorded(ledger):
    ledger.commit("policy", policy_v1(), metadata={"source": "policy.pdf", "actor": "etl"})
    revision = ledger.revisions("policy")[0]

    assert revision.action == "commit"
    assert revision.metadata == {"source": "policy.pdf", "actor": "etl"}
    assert revision.parent_rev is None
    assert revision.created_at
