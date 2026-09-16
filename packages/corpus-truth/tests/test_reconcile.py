"""reconcile(): drift between the ledger's current view and a live index."""

from conftest import policy_v1, policy_v2

from corpus_truth import IndexAdapter, InMemoryIndex, TextChunk, apply_plan


def test_a_correctly_applied_plan_reconciles_clean(ledger, index):
    apply_plan(index, ledger.commit("policy", policy_v1()))
    apply_plan(index, ledger.commit("policy", policy_v2()))

    report = ledger.reconcile(index)

    assert report.is_clean
    assert report.to_dict() == {
        "orphans": [],
        "missing": [],
        "unknown": [],
        "duplicate_hashes": [],
        "stale": [],
        "is_clean": True,
    }


def test_skipping_the_removals_leaves_orphans(ledger, index):
    first = ledger.commit("policy", policy_v1())
    apply_plan(index, first)
    plan = ledger.commit("policy", policy_v2())
    index.add(plan.add)  # upsert applied, delete forgotten

    report = ledger.reconcile(index)

    assert set(report.orphans) == set(plan.remove) == {posting.id for posting in first.add}
    assert report.missing == ()
    assert not report.is_clean


def test_skipping_the_upserts_leaves_missing_postings(ledger, index):
    apply_plan(index, ledger.commit("policy", policy_v1()))
    plan = ledger.commit("policy", policy_v2())
    index.remove(plan.remove)  # delete applied, upsert forgotten

    report = ledger.reconcile(index)

    assert set(report.missing) == {posting.id for posting in plan.add}
    assert report.orphans == ()


def test_ids_the_ledger_has_never_seen_are_reported_as_unknown(ledger, index):
    apply_plan(index, ledger.commit("policy", policy_v1()))
    index.records["stray-uuid"] = {"id": "stray-uuid", "text": "indexed outside the ledger"}

    report = ledger.reconcile(index)

    assert report.unknown == ("stray-uuid",)
    assert report.orphans == ()


def test_reconcile_can_be_scoped_to_one_document(ledger, index):
    policy = ledger.commit("policy", policy_v1())
    apply_plan(index, policy)
    apply_plan(index, ledger.commit("handbook", [TextChunk("Welcome.", key="intro")]))
    ledger.delete("policy")  # removals not applied

    scoped = ledger.reconcile(index, doc_id="policy")
    assert set(scoped.orphans) == {posting.id for posting in policy.add}
    assert scoped.missing == ()
    assert scoped.unknown == ()  # other documents in the same index are not our business

    whole = ledger.reconcile(index)
    assert set(whole.orphans) == {posting.id for posting in policy.add}
    assert whole.missing == ()


def test_duplicate_payload_hashes_inside_one_document_are_reported(ledger, index):
    plan = ledger.commit(
        "policy",
        [TextChunk("Same paragraph.", key="a"), TextChunk("Same paragraph.", key="b")],
    )
    apply_plan(index, plan)

    report = ledger.reconcile(index)

    assert report.orphans == () and report.missing == ()
    assert len(report.duplicate_hashes) == 1
    entry = report.duplicate_hashes[0]
    assert entry["doc_id"] == "policy"
    assert entry["ids"] == sorted(posting.id for posting in plan.add)
    assert not report.is_clean


def test_identical_text_in_two_documents_is_not_a_duplicate(ledger, index):
    apply_plan(index, ledger.commit("policy", [TextChunk("Shared boilerplate.", key="a")]))
    apply_plan(index, ledger.commit("handbook", [TextChunk("Shared boilerplate.", key="a")]))

    assert ledger.reconcile(index).duplicate_hashes == ()


def test_stale_metadata_in_the_index_is_reported(ledger, index):
    plan = ledger.commit("policy", policy_v1())
    apply_plan(index, plan)
    target = plan.add[0].id
    index.records[target]["is_current"] = False
    index.records[target]["rev"] = 99

    report = ledger.reconcile(index)

    assert [entry["field"] for entry in report.stale] == ["is_current", "rev"]
    assert all(entry["id"] == target for entry in report.stale)
    assert report.stale[1]["index"] == 99
    assert report.stale[1]["ledger"] == 1


def test_reconcile_works_with_an_adapter_that_has_no_get(ledger):
    class BareIndex:
        def __init__(self):
            self.ids = []

        def add(self, postings):
            self.ids.extend(posting.id for posting in postings)

        def remove(self, posting_ids):
            self.ids = [held for held in self.ids if held not in set(posting_ids)]

        def list_ids(self):
            return list(self.ids)

    bare = BareIndex()
    assert isinstance(bare, IndexAdapter)
    apply_plan(bare, ledger.commit("policy", policy_v1()))

    report = ledger.reconcile(bare)

    assert report.is_clean
    assert report.stale == ()


def test_an_empty_index_reports_every_current_posting_as_missing(ledger):
    plan = ledger.commit("policy", policy_v1())

    report = ledger.reconcile(InMemoryIndex())

    assert set(report.missing) == {posting.id for posting in plan.add}
