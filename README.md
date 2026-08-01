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


| approach | recall | critical | answerable | passages | latency |
|---|---|---|---|---|---|
| keyword search (BM25) | 61% | 60% | 14 / 28 | 5 | ~0s |
| semantic search (nomic-embed-v1.5) | 63% | 66% | 16 / 28 | 5 | ~0s |
| **HyDE → semantic** | **82%** | **80%** | **20 / 28** | **5** | 3.9s |
| **iterative loop (3 rounds) → semantic** | **95%** | **96%** | **27 / 28** | 19 | 6.8s |
| full context, curated 2 pubs (~80k tok) | 95% | 93% | 24 / 28 | — | ~0s |
| full context, all 5 pubs (~393k tok) | 100% | 100% | 28 / 28 | — | ~0s |

### By question type — this is where the ranking actually lives

| | keyword | semantic | HyDE | loop | full (all) |
|---|---|---|---|---|---|
| factual lookup | 4/5 | 5/5 | 5/5 | 5/5 | 5/5 |
| strategy | 2/3 | 2/3 | 2/3 | 3/3 | 3/3 |
| **vocabulary-gap** | **4/9** | **4/9** | 7/9 | **9/9** | 9/9 |
| **misleading** | 2/5 | 3/5 | 3/5 | 4/5 | 5/5 |
| **multi-hop** | 2/6 | 2/6 | 3/6 | **6/6** | 6/6 |

**Semantic search is no better than keyword search on the vocabulary gap —
4/9 each.** That is the sharpest confirmation available that the problem is not
retrieval quality. Both indexes held the answer; neither was asked for it in
terms it could match.

### Plain retrieval is not enough, and more of it does not fix it

The two retrievers fail on *different* questions, so fusion helps — their union
reaches 7/11 — but that is a ceiling assuming perfect fusion. Four questions
(`rental-below-zero`, `time-spent-managing`, `converted-home`,
`depreciate-then-sell`) are reached by neither. They are exactly the strategy,
vocabulary-gap and multi-hop cases. Every factual lookup passed everywhere.

### Query-side strategies close the gap, and one of them is cheap

Both wrappers attack the same root cause: the caller's words are not the
corpus's words. Watch the loop bridge it on a deliberately misleading question —
*"I fixed the roof on my rental. Can I deduct the whole cost right now?"* —
where "fixed"/"repair" points at the wrong rule and the real answer is
capitalisation:

```
round 2: repair vs improvement safe harbor election
         de minimis safe harbor rental property
         routine maintenance safe harbor rental property
round 3: rental property roof repair versus improvement betterment restoration
         rental property repair regulations section 1.162-4 capitalization
```

None of those terms are in the question. **HyDE is the efficiency result** —
20/28 on the same five passages as plain semantic, one extra model call.
**The loop is the completeness result** — 27/28.

### It narrates, because six seconds of silence is not acceptable

The same model call that picks the next queries is also asked for one short
spoken sentence. Free, no extra round trip:

```
[ 2.8s] "Checking if there are special rules that let you deduct certain
         improvements immediately instead of depreciating."
[ 8.0s] "Checking the specific rules that distinguish a deductible repair
         from a capital improvement for roofs."
```

Pass `on_progress=` to `IterativeLoop`. In a voice agent that callback feeds the
stream the speech path already consumes.

### The fairness checks

A loop sees more passages than a single shot, so `passages_seen` is recorded for
every strategy — the loop's 27/28 costs 19 passages against semantic's 5.

And the no-retrieval control is run twice, because the difference is a result.
Curated to the two publications a human judged relevant, it scores 24/28 — and
the four it misses are precisely the questions whose answers live in the
publications that curation excluded. **Curation is a guess about what will be
asked.** Given all five publications it scores 28/28, at 393k tokens per
question against the loop's ~19 passages: roughly 78× the context for one more
question.

### The question nothing retrieved

`deduct-my-own-time` — *"Can I deduct the value of my time managing my
rentals?"* — was answered only by full context. The answer is that no such
deduction exists, and **you cannot retrieve a negative.** No passage says "there
is no deduction for your own labour"; the absence is the answer. Any system
built on retrieval alone will fail this class of question, and it will fail it
confidently.

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

## The API key

HyDE and the loop need one model call per round. Provide a key by any of:

```bash
export ANTHROPIC_API_KEY=sk-ant-...
# or
export BAKEOFF_ENV_FILE=/path/to/some/.env     # a file containing ANTHROPIC_API_KEY=
# or drop a .env beside this README
```

Without one, those two strategies report an error rather than quietly
degrading to a single search — a loop that cannot make its decision call is
not a loop, and reporting single-shot numbers under a loop's name once made a
broken run look like a finding.
