# corpus_truth

Versioned corpus truth for RAG indexes. You keep parsing, chunking, embedding,
retrieving and generating. `corpus_truth` sits between chunk and index: it
assigns revisions, tells you exactly what to add and remove, keeps history, and
hands generation a changelog an LLM can read.

It is a **sidecar**, not a new row format. Keep `id + vector + metadata` and keep
your UUIDs. The ledger records which of *your* ids is current, which id replaced
which (`uuid-x -> uuid-y`), what each payload hashed to, and both sides of every
change.

```bash
pip install ./packages/corpus-truth     # no required dependencies, Python 3.10+
```

## Plug it into an existing embed + upsert loop

```python
from corpus_truth import ImageChunk, Ledger, TableChunk, TextChunk

ledger = Ledger.open("./.corpus")                 # SQLite, opened once per process

plan = ledger.commit("policy", [                  # your stable doc id + stable chunk keys
    TextChunk("Refunds take 14 days.", key="refund"),
    TableChunk([["item", "price"], ["A", "12"]], key="pricing"),
    ImageChunk("bbb...", key="fig1", uri="s3://new.png",
               description="Wiring diagram with 5 connectors."),
])

index.upsert(                                     # embed only what changed
    ids=[posting.id for posting in plan.add],
    vectors=embed([posting.embed_text() for posting in plan.add]),
    metadata=[posting.to_index_record() for posting in plan.add],
)
index.delete(plan.remove)                         # add before remove, always

prompt += ledger.diff_prompt("policy")            # later, when a user asks what changed
```

Second commit of unchanged content costs nothing: matching payload hashes keep
their id and vector and never appear in `plan.add`. `plan.remap` is the
`old_id -> new_id` map for the slots that did change.

```text
Document policy is now rev 2 (was 1).
- text/refund: "30 days." → "14 days."
- table/pricing cell B2: 9 → 12
- image/fig1: "Wiring diagram with 3 connectors." → "Wiring diagram with 5 connectors."
  old: s3://old.png
  new: s3://new.png
2 chunks unchanged. Current revision is 2.
```

## API

| Call | Returns |
|---|---|
| `commit(doc_id, chunks, metadata=None)` | `IndexMutationPlan` -- `add`, `remove`, `remap`, `unchanged` |
| `delete(doc_id)` | plan that removes every current id; history stays queryable |
| `supersede(old_doc_id, new_doc_id)` | plan retiring an older document (commit the replacement first) |
| `current(doc_id)` / `as_of(doc_id, rev=...)` | `list[Posting]` for now, or for any past revision |
| `lineage(doc_id, chunk_key)` | every posting that has held one slot, oldest first |
| `diff(doc_id, from_rev=None, to_rev=None)` | kind-aware `added` / `removed` / `changed` |
| `diff_prompt(doc_id, ...)` | the same changelog as short markdown for a prompt |
| `reconcile(index)` | orphans, missing postings, duplicate hashes, stale metadata |
| `head(doc_id)` / `revisions(doc_id)` / `documents()` | document state and timeline |

Postings carry `doc_id`, `key`, `kind`, `rev`, `is_current`, `valid_from`,
`valid_to`, `parent_rev` and `replaces`, so a retriever hit can filter and cite
them. Default behaviour is current-only: you delete `plan.remove` from the index.
Keep the old rows instead if you want historical kNN, and filter
`is_current == true` at query time.

## Required vs optional chunk fields

Versioning works with hashes alone; diffs get readable when you send payloads.

| Kind | Minimum | For a useful changelog |
|---|---|---|
| `TextChunk` | `text` | -- (unified patch always available) |
| `TableChunk` | `rows` **or** `hash` | `rows` gives cell diffs (`B2: 9 → 12`) |
| `ImageChunk` | `sha256` **or** `data` | `description` for a text LLM, `uri` for a vision model |
| `OtherChunk` | `mime` + (`hash` or `data`) | `metadata` for a readable metadata diff |

Without `rows` a table change reads "table changed"; without `description` an
image change reads "image replaced". We never open a PDF or PNG, never run OCR or
a captioner, and never store your bytes -- your parser fills these in.

## Adapters

An adapter is three methods: `add(postings)`, `remove(ids)`, `list_ids()`.
`InMemoryIndex` is the reference implementation; `corpus_truth.faiss_example`
is a copy-paste `IndexIDMap2` template whose imports are lazy, so nothing heavy
is installed with this package.

```python
from corpus_truth import apply_plan
apply_plan(index, plan)     # add first, then remove
```

## Tests

```bash
cd packages/corpus-truth && python -m pytest tests -q
```

MIT licensed, same as the repository.
