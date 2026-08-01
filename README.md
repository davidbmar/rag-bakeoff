# rag-bakeoff

Settles which retrieval architecture to use by measuring it, rather than by
architecture taste.

The portfolio contains six independent "upload documents, ask questions"
implementations sharing no code. That happened because none was ever measured
against another: with no way to tell whether a new one beat an old one,
building a seventh always looked as reasonable as improving the third. This is
the missing measurement.

## What it measures

**Required-concept recall.** For each question, the concepts you must have in
front of you to answer correctly are written down in advance and verified to
exist in the corpus (`validate_questions.py` refuses to run a key containing a
concept the source material lacks). An engine is scored on how many of those
its retrieved passages actually contain.

Retrieval is scored rather than final answers because an LLM judge conflates
two different failures — not finding the material, and mishandling material it
did find. Only the first decides whether an architecture can work at all. A
model cannot reason from a passage it was never shown.

Two secondary numbers carry as much weight:

- **critical recall** — concepts whose absence makes the answer *actively
  wrong* rather than merely thin. Missing the passive-activity rule on the
  rental question does not produce a vaguer answer; it produces a confident
  and incorrect one.
- **discrimination** — the score gap between answerable and unanswerable
  questions. An engine whose scores do not separate the two cannot abstain,
  whatever is built on top of it.

## Corpus and questions

Five IRS publications (p527 rental, p925 passive activity, p946 depreciation,
p523 home sale, 1040 instructions) — 1.57M characters, ~393k tokens. Public,
stable, and full of worked examples, so ground truth is free.

Thirteen questions across five types: strategy, vocabulary-gap (the caller's
words never appear in the corpus), multi-hop, factual lookup, and two
questions the corpus cannot answer at all. Every scored question is
cross-document by construction.

## Running it

```bash
python3 validate_questions.py          # check the key against the corpus first
python3 run_bakeoff.py --ingest        # load the corpus into each engine
python3 run_bakeoff.py --k 5           # score
python3 analyze.py                     # per-question comparison + hybrid ceiling
```

Engines are declared in `adapters/engines.py`. Adding one is a class with
`health`, `ingest`, and `retrieve`.

## What it found

Two names worth pinning down first, because the literature uses them constantly
and they sound more exotic than they are:

- **Sparse retrieval** (BM25, keyword search) represents a passage as a vector
  with one slot per word in the vocabulary. A passage uses a few hundred of
  perhaps 50,000 words, so almost every slot is zero — the vector is *sparse*.
- **Dense retrieval** (semantic/vector/embedding search) uses a neural model to
  compress meaning into a short vector — 768 numbers here — where essentially
  every number is non-zero and carries part of the meaning. No empty slots, so
  *dense*.

“Dense” therefore just means embedding-based semantic search. The arrow in
`HyDE → semantic search` means wrapping: HyDE is a strategy sitting in front of
a retriever, not a retriever itself.


| strategy | recall | critical | answerable | passages | latency |
|---|---|---|---|---|---|
| keyword search (BM25/FTS5) | 54% | 47% | 4 / 11 | 5 | ~0s |
| semantic search (nomic-embed-v1.5) | 71% | 64% | 6 / 11 | 5 | ~0s |
| semantic search, returning 17 instead of 5 | 85% | 86% | 9 / 11 | 17 | ~0s |
| **HyDE → semantic search** | **93%** | **95%** | **10 / 11** | **5** | 4s |
| **iterative loop (3 rounds) → semantic search** | **100%** | **100%** | **11 / 11** | 17 | 6s |
| full context (p527+p925, ~80k tok) | 100% | 100% | 11 / 11 | — | ~0s |

### Plain retrieval is not enough, and more of it does not fix it

The two retrievers fail on *different* questions, so fusion helps — their union
reaches 7/11 — but that is a ceiling assuming perfect fusion. Four questions
(`rental-below-zero`, `time-spent-managing`, `converted-home`,
`depreciate-then-sell`) are reached by neither. They are exactly the strategy,
vocabulary-gap and multi-hop cases. Every factual lookup passed everywhere.

### Query-side strategies close the gap, and they are cheap

Both wrappers attack the same root cause: the caller's words are not the
corpus's words, so a query built from the question cannot find the governing
rule. Watch the loop bridge it on the rental question — round 1 searches the
caller's phrasing, then:

```
round 2: passive activity loss limitation rental real estate
         at-risk rules rental property losses
         rental loss deduction $25000 special allowance
round 3: passive activity loss limitation deduction ordering rules
         rental real estate active participation requirements definition
```

None of those terms appear in the question. HyDE reaches the same vocabulary in
one shot by writing the passage it expects to exist, then searching with that.

**HyDE is the efficiency result**: 10/11 on the *same five passages* as plain
semantic search, for one extra model call. It beats a single semantic search
returning 17 passages (9/11)
while putting a third as much in the context window.

**The loop is the completeness result**: 11/11, matching full context on 17
passages instead of 80k tokens.

### The fairness check that matters

A loop sees more passages than a single shot, so beating single-shot k=5 proves
nothing on its own. At the loop's own budget, a single semantic search scores
9/11 —
so of the naive 6→11 improvement, +3 is simply seeing more and +2 is the loop
itself. The loop's advantage is real but smaller than the headline suggests,
which is why `passages_seen` is recorded for every strategy.

## The conclusion

For a bounded corpus, curated long context is unbeatable on quality and
trivial to build — 80k tokens answers everything here. But it does not scale
past the window, and it requires knowing which documents to load.

**The finding worth acting on is that the gap is query-side, not index-side.**
The engine you already run is not the problem; asking it one question built
from the caller's vocabulary is. HyDE is one model call in front of an existing
retriever and recovers most of the loss. The loop recovers the rest when the
question genuinely spans several rules.

## Excluded, and why

Three further systems were available but could not take this corpus on equal
terms, and are listed rather than quietly dropped: one indexes source
repositories through an API with no path for an external PDF; one runs entirely
in the browser over prebuilt conversation packs with no server; one can ingest
but only through its own bespoke evidence schema. A fourth — the most
interesting architecturally, combining strong embeddings with cross-encoder
reranking and explicit abstention — was not available to test and remains the
obvious next candidate.

## Adding your own engine

A class with `health`, `ingest` and `retrieve`. See `adapters/base.py` for the
contract and `adapters/engines.py` for three worked examples. Strategies like
HyDE and the loop wrap an engine rather than replacing it — see
`adapters/strategies.py`.
