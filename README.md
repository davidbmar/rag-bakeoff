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

| engine | retrieval | mean recall | critical | fully answerable |
|---|---|---|---|---|
| intelligence-platform | BM25 / FTS5 | 54% | 47% | 4 / 11 |
| voice-optimal-RAG | dense, nomic-embed-v1.5 | 71% | 64% | 6 / 11 |
| full context (p527+p925, ~80k tokens) | none | **100%** | **100%** | **11 / 11** |

Raising k from 5 to 10 moves the retrievers to 67% and 77% — better, not
close.

The two retrievers fail on *different* questions, so fusing them helps: the
union reaches 7/11. That is a ceiling assuming perfect fusion, and it is still
four questions short of simply putting the right two publications in the
prompt.

Four questions are reached by no retrieval strategy tested — `rental-below-zero`,
`time-spent-managing`, `converted-home`, `depreciate-then-sell`. They are
exactly the strategy, vocabulary-gap and multi-hop cases. Every factual lookup
passed everywhere.

## The conclusion

For a bounded, stable corpus, **curated long context beats retrieval outright**
— not marginally, and most decisively on precisely the questions worth asking.
Retrieval earns its place when the corpus outgrows the window, and hybrid beats
either channel alone, but neither closes the gap on reasoning questions.

## Excluded, and why

`github-portfolio-search` ingests GitHub repos through the API with no path for
an external PDF. `browser-RAG` (FSM/Iris) runs in the browser over prebuilt
JSONL packs with no server. `persona-rag` can ingest but only via its own
evidence-item JSON schema and has been idle since May — includable with about a
day of adapter work. `evidence-qa` is private and not cloned; on paper it is
the closest architecture to right (BGE + cross-encoder + abstention) and is the
one worth cloning next.
