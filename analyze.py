"""Compare the engines question by question, and price the hybrid.

The headline averages hide the finding that matters: whether two engines fail
on the *same* questions or different ones. If their failures overlap, hybrid
retrieval buys little and the ceiling is low. If they are complementary, the
union is the argument for fusing them — and it can be computed exactly from
runs already made, rather than guessed at.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).parent
LEXICAL = "intelligence-platform"
DENSE = "voice-optimal-RAG"


def load(k: int) -> dict:
    return json.loads((ROOT / "results" / f"bakeoff-k{k}.json").read_text())


def by_id(report: dict, engine: str) -> dict:
    block = report["engines"].get(engine, {})
    return {r["id"]: r for r in block.get("results", []) if "recall" in r}


def main() -> None:
    spec = json.loads((ROOT / "questions.json").read_text())
    types = {q["id"]: q["type"] for q in spec["questions"]}
    report = load(5)
    lex, dense = by_id(report, LEXICAL), by_id(report, DENSE)

    print(f"{'question':<24} {'type':<15} {'BM25':>6} {'dense':>6} {'union':>6}  winner")
    print("-" * 74)

    union_full = lex_full = dense_full = 0
    for qid in lex:
        if qid not in dense:
            continue
        question = next(q for q in spec["questions"] if q["id"] == qid)
        required = {c["name"] for c in question["required_concepts"]}
        criticals = {c["name"] for c in question["required_concepts"] if c.get("critical")}

        found_lex = set(lex[qid]["found"])
        found_dense = set(dense[qid]["found"])
        union = found_lex | found_dense

        # A question counts as answerable only when every critical concept is
        # present — a partial answer to a tax question is not a partial credit.
        lex_ok = criticals <= found_lex
        dense_ok = criticals <= found_dense
        union_ok = criticals <= union
        lex_full += lex_ok
        dense_full += dense_ok
        union_full += union_ok

        if lex_ok and not dense_ok:
            winner = "BM25 only"
        elif dense_ok and not lex_ok:
            winner = "dense only"
        elif union_ok:
            winner = "both"
        else:
            winner = "NEITHER"

        print(
            f"{qid:<24} {types[qid]:<15} "
            f"{len(found_lex) / len(required):>5.0%} "
            f"{len(found_dense) / len(required):>5.0%} "
            f"{len(union) / len(required):>5.0%}  {winner}"
        )

    total = len(lex)
    print("-" * 74)
    print(
        f"questions fully answerable (all critical concepts retrieved), of {total}:\n"
        f"    BM25 alone   {lex_full}\n"
        f"    dense alone  {dense_full}\n"
        f"    union        {union_full}   <- the ceiling a hybrid could reach\n"
        f"    full context {total}   <- no retrieval at all"
    )

    neither = [
        qid for qid in lex
        if qid in dense
        and not ({c["name"] for c in next(q for q in spec["questions"] if q["id"] == qid)
                  ["required_concepts"] if c.get("critical")}
                 <= set(lex[qid]["found"]) | set(dense[qid]["found"]))
    ]
    if neither:
        print(f"\nno retrieval strategy tested reaches these: {', '.join(neither)}")


if __name__ == "__main__":
    main()
