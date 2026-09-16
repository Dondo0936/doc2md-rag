"""delete() and supersede(): tombstones, and the ban on ghost chunks."""

import pytest
from conftest import ids_by_key, policy_v1, policy_v2

from corpus_truth import DocumentNotFoundError, TextChunk, apply_plan


def test_delete_tombstones_the_document_and_removes_every_current_id(ledger, index):
    first = ledger.commit("policy", policy_v1())
    apply_plan(index, first)

    plan = ledger.delete("policy")
    apply_plan(index, plan)

    assert plan.tombstone is True
    assert plan.add == ()
    assert set(plan.remove) == {posting.id for posting in first.add}
    assert plan.rev == 2 and plan.parent_rev == 1
    assert ledger.current("policy") == []
    assert index.list_ids() == []
    assert ledger.head("policy").is_deleted is True
    assert ledger.revisions("policy")[-1].action == "delete"


def test_deleted_chunks_are_never_current_but_stay_visible_as_of_an_earlier_rev(ledger):
    first = ledger.commit("policy", policy_v1())
    ghost_ids = {posting.id for posting in first.add}
    ledger.delete("policy")

    assert {posting.id for posting in ledger.current("policy")} == set()
    assert {posting.id for posting in ledger.as_of("policy", rev=1)} == ghost_ids
    assert all(not posting.is_current for posting in ledger.as_of("policy", rev=1))
    assert all(posting.valid_to == 2 for posting in ledger.as_of("policy", rev=1))


def test_a_ghost_left_in_the_index_is_reported_not_silently_current(ledger, index):
    first = ledger.commit("policy", policy_v1())
    apply_plan(index, first)
    ledger.delete("policy")  # caller forgets to apply the removals

    report = ledger.reconcile(index)

    assert set(report.orphans) == {posting.id for posting in first.add}
    assert report.missing == ()
    assert not report.is_clean


def test_deleting_twice_is_idempotent(ledger):
    ledger.commit("policy", policy_v1())
    ledger.delete("policy")
    again = ledger.delete("policy")

    assert again.is_noop
    assert again.tombstone is True
    assert again.rev == 2
    assert [revision.action for revision in ledger.revisions("policy")] == ["commit", "delete"]


def test_committing_after_a_delete_revives_the_document_with_fresh_ids(ledger, index):
    first = ledger.commit("policy", policy_v1())
    apply_plan(index, first)
    apply_plan(index, ledger.delete("policy"))

    revived = ledger.commit("policy", policy_v1())
    apply_plan(index, revived)

    assert revived.rev == 3
    assert revived.parent_rev == 2
    assert len(revived.add) == 3
    assert revived.remove == ()
    assert {posting.id for posting in revived.add}.isdisjoint({posting.id for posting in first.add})
    assert ledger.head("policy").is_deleted is False
    assert ledger.reconcile(index).is_clean


def test_delete_of_an_unknown_document_raises(ledger):
    with pytest.raises(DocumentNotFoundError, match="policy"):
        ledger.delete("policy")


def test_supersede_retires_the_old_document_in_favour_of_the_new_one(ledger, index):
    old = ledger.commit("policy-v3", policy_v1())
    apply_plan(index, old)
    new = ledger.commit("policy-v4", policy_v2())
    apply_plan(index, new)

    plan = ledger.supersede("policy-v3", "policy-v4")
    apply_plan(index, plan)

    assert plan.doc_id == "policy-v3"
    assert plan.tombstone is True
    assert plan.add == ()
    assert set(plan.remove) == {posting.id for posting in old.add}
    assert ledger.current("policy-v3") == []
    assert set(index.list_ids()) == {posting.id for posting in new.add}

    head = ledger.head("policy-v3")
    assert head.is_deleted is True
    assert head.superseded_by == "policy-v4"
    assert ledger.revisions("policy-v3")[-1].action == "supersede"
    assert ledger.revisions("policy-v3")[-1].metadata == {"superseded_by": "policy-v4"}


def test_supersede_keeps_the_old_history_readable(ledger):
    old = ledger.commit("policy-v3", policy_v1())
    ledger.commit("policy-v4", policy_v2())
    ledger.supersede("policy-v3", "policy-v4")

    assert ids_by_key(ledger.as_of("policy-v3", rev=1)) == ids_by_key(old.add)


def test_supersede_is_idempotent(ledger):
    ledger.commit("policy-v3", policy_v1())
    ledger.commit("policy-v4", policy_v2())
    ledger.supersede("policy-v3", "policy-v4")
    again = ledger.supersede("policy-v3", "policy-v4")

    assert again.is_noop
    assert again.rev == 2


def test_supersede_needs_both_documents_and_two_distinct_ids(ledger):
    ledger.commit("policy-v3", [TextChunk("old", key="k")])

    with pytest.raises(DocumentNotFoundError, match="policy-v4"):
        ledger.supersede("policy-v3", "policy-v4")
    with pytest.raises(DocumentNotFoundError, match="missing"):
        ledger.supersede("missing", "policy-v3")
    with pytest.raises(ValueError, match="itself"):
        ledger.supersede("policy-v3", "policy-v3")


def test_deleting_one_document_leaves_the_others_alone(ledger, index):
    apply_plan(index, ledger.commit("policy", policy_v1()))
    handbook = ledger.commit("handbook", [TextChunk("Welcome.", key="intro")])
    apply_plan(index, handbook)

    apply_plan(index, ledger.delete("policy"))

    assert index.list_ids() == [handbook.add[0].id]
    assert len(ledger.current("handbook")) == 1
    assert ledger.reconcile(index).is_clean
