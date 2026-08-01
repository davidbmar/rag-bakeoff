"""Run the same questions through every engine and score what came back.

The metric is **required-concept recall**: of the things you must have in front
of you to answer a question correctly, how many did this engine actually
retrieve? It is objective, cheap, and it isolates the failure that decides
whether an architecture can work at all — a model cannot reason from a passage
it was never shown.

Two secondary numbers matter as much as the headline:

  critical recall   concepts whose absence makes the answer actively wrong
                    rather than merely thin. Missing the passive-activity rule
                    on the rental question does not produce a vaguer answer,
                    it produces a confidently incorrect one.

  discrimination    the gap between top scores on answerable and unanswerable
                    questions. An engine whose scores do not separate the two
                    cannot abstain, no matter what is built on top of it — and
                    for tax advice, answering confidently from nothing is the
                    worst available behaviour.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from adapters.engines import FullContext, IntelligencePlatform, VoiceOptimalRAG  # noqa: E402
from adapters.strategies import HyDE, IterativeLoop  # noqa: E402

ROOT = Path(__file__).parent


def score_question(question: dict, text: str) -> dict:
    """Which required concepts made it into the retrieved context."""

    found, missing, critical_missing = [], [], []
    for concept in question["required_concepts"]:
        if re.search(concept["pattern"], text, re.IGNORECASE):
            found.append(concept["name"])
        else:
            missing.append(concept["name"])
            if concept.get("critical"):
                critical_missing.append(concept["name"])

    total = len(question["required_concepts"])
    criticals = [c for c in question["required_concepts"] if c.get("critical")]
    return {
        "found": found,
        "missing": missing,
        "critical_missing": critical_missing,
        "recall": len(found) / total if total else None,
        "critical_recall": (
            (len(criticals) - len(critical_missing)) / len(criticals) if criticals else None
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=5, help="passages per question")
    parser.add_argument("--ingest", action="store_true", help="load the corpus first")
    parser.add_argument("--only", help="run a single engine by name substring")
    args = parser.parse_args()

    spec = json.loads((ROOT / "questions.json").read_text())
    dense = VoiceOptimalRAG()
    engines = [
        IntelligencePlatform(base_url="http://127.0.0.1:8124"),
        dense,
        # Query-side strategies over the strongest single retriever. Both are
        # wrappers, not new engines — the point being that neither requires
        # replacing an index you already have.
        HyDE(dense),
        IterativeLoop(dense, rounds=3, per_round=3),
        # The realistic form of the no-retrieval option: the two publications
        # that actually govern these questions, ~80k tokens.
        FullContext(subset=["p527", "p925"]),
    ]
    if args.only:
        engines = [e for e in engines if args.only.lower() in e.name.lower()]

    report: dict = {"k": args.k, "engines": {}}

    for engine in engines:
        print(f"\n{'=' * 66}\n{engine.name}  —  {engine.retrieval}\n{'=' * 66}")
        problem = engine.health()
        if problem:
            print(f"  SKIPPED: {problem}")
            report["engines"][engine.name] = {"skipped": problem}
            continue

        if args.ingest:
            print("  ingesting corpus...")
            started = time.time()
            try:
                stats = engine.ingest(ROOT / "corpus")
            except Exception as exc:  # noqa: BLE001
                print(f"  INGEST FAILED: {type(exc).__name__}: {exc}")
                report["engines"][engine.name] = {"skipped": f"ingest failed: {exc}"}
                continue
            for doc in stats.get("documents", []):
                print(f"    {doc['doc']:<10} {doc.get('segments')} segments")
            print(f"  ingested in {time.time() - started:.0f}s")

        results, answerable_scores, abstain_scores = [], [], []
        for question in spec["questions"]:
            got = engine.retrieve(question["question"], args.k)
            if got.error:
                print(f"  ! {question['id']}: {got.error}")
                results.append({"id": question["id"], "error": got.error})
                continue

            if question["type"] == "should-abstain":
                abstain_scores.append(got.top_score)
                print(
                    f"  {question['id']:<24} top_score={got.top_score:.3f} "
                    f"({len(got.chunks)} passages returned)"
                )
                results.append(
                    {
                        "id": question["id"],
                        "type": question["type"],
                        "top_score": got.top_score,
                        "chunks": len(got.chunks),
                    }
                )
                continue

            answerable_scores.append(got.top_score)
            scored = score_question(question, got.text)
            flag = "✓" if not scored["critical_missing"] else "✗"
            detail = (
                f"missing critical: {', '.join(scored['critical_missing'])}"
                if scored["critical_missing"]
                else ""
            )
            print(
                f"  {flag} {question['id']:<24} recall={scored['recall']:.0%} "
                f"critical={scored['critical_recall']:.0%}  {detail}"
            )
            results.append({"id": question["id"], "type": question["type"], **scored,
                            "top_score": got.top_score, "latency_ms": got.latency_ms,
                            # A multi-round strategy sees more passages than a
                            # single shot; without this the comparison flatters it.
                            "passages_seen": len(got.chunks)})

        scored_only = [r for r in results if r.get("recall") is not None]
        if scored_only:
            summary = {
                "mean_recall": statistics.mean(r["recall"] for r in scored_only),
                "mean_critical_recall": statistics.mean(
                    r["critical_recall"] for r in scored_only
                ),
                "questions_with_all_criticals": sum(
                    1 for r in scored_only if not r["critical_missing"]
                ),
                "questions_scored": len(scored_only),
                "mean_passages_seen": statistics.mean(
                    r.get("passages_seen", 0) for r in scored_only
                ),
                "mean_latency_ms": statistics.mean(
                    r.get("latency_ms", 0) for r in scored_only
                ),
            }
            if answerable_scores and abstain_scores:
                summary["discrimination"] = (
                    statistics.mean(answerable_scores) - statistics.mean(abstain_scores)
                )
            report["engines"][engine.name] = {
                "retrieval": engine.retrieval,
                "summary": summary,
                "results": results,
            }
            print(
                f"\n  MEAN RECALL {summary['mean_recall']:.0%}  |  "
                f"CRITICAL {summary['mean_critical_recall']:.0%}  |  "
                f"fully-answerable {summary['questions_with_all_criticals']}"
                f"/{summary['questions_scored']}  |  "
                f"{summary['mean_passages_seen']:.0f} passages, "
                f"{summary['mean_latency_ms'] / 1000:.1f}s per question"
            )
            if "discrimination" in summary:
                print(f"  DISCRIMINATION (answerable − unanswerable score): "
                      f"{summary['discrimination']:+.3f}")

    out = ROOT / "results" / f"bakeoff-k{args.k}.json"
    out.write_text(json.dumps(report, indent=2))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
