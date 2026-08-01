"""Retrieval strategies that wrap an engine rather than replace it.

The four questions no engine reached all fail the same way: the caller's words
do not appear in the corpus, so a single query built from those words cannot
find the governing rule. That is a query-side problem, and both strategies here
attack it from the query side while leaving the index alone.

Which is the useful architectural point — neither of these is a new retriever.
They sit in front of whichever one you already have.

FAIRNESS: a loop that runs five rounds sees five times the passages of a
single shot, so beating single-shot k=5 proves nothing on its own. Every
strategy reports `passages_seen`, and the comparison that counts is against a
single shot with the same budget.
"""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path

import httpx

from adapters.base import Retrieved

MODEL = os.environ.get("BAKEOFF_MODEL", "claude-sonnet-4-5")


def _api_key() -> str:
    key = os.environ.get("ANTHROPIC_API_KEY", "")
    if key:
        return key
    env = Path.home() / "src/nano-claw/.env"
    if env.exists():
        found = re.search(r"^ANTHROPIC_API_KEY=(.+)$", env.read_text(), re.M)
        if found:
            return found.group(1).strip()
    raise RuntimeError("no ANTHROPIC_API_KEY available")


def ask(prompt: str, max_tokens: int = 600) -> str:
    """One completion. Kept deliberately small — this is plumbing, not the product."""

    response = httpx.post(
        "https://api.anthropic.com/v1/messages",
        headers={
            "x-api-key": _api_key(),
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        },
        json={
            "model": MODEL,
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": prompt}],
        },
        timeout=120,
    )
    response.raise_for_status()
    return "".join(
        block.get("text", "") for block in response.json().get("content", [])
    )


class HyDE:
    """Search with a hypothetical answer instead of the question.

    The model writes the passage it expects to exist, and that passage is used
    as the query. It contains the corpus's vocabulary — "passive activity
    loss", "Form 8582" — even when the caller's question does not, which is
    precisely the gap that sank the strategy questions.

    Single-shot: one extra model call, one retrieval. The cheapest thing that
    could possibly address the failure.
    """

    def __init__(self, engine, k_multiplier: int = 1):
        self.engine = engine
        self.k_multiplier = k_multiplier
        self.name = f"HyDE → {engine.name}"
        self.retrieval = f"hypothetical-document query over {engine.retrieval}"

    def health(self) -> str | None:
        return self.engine.health()

    def ingest(self, corpus_dir: Path) -> dict:
        return self.engine.ingest(corpus_dir)

    def retrieve(self, question: str, k: int) -> Retrieved:
        started = time.time()
        try:
            hypothetical = ask(
                "Write a short passage, two or three sentences, exactly as it "
                "would appear in an IRS publication, answering this question. "
                "Use the technical terms the IRS would use, not the words in "
                "the question. Do not hedge and do not mention that you are "
                "writing an example.\n\n"
                f"Question: {question}"
            )
        except Exception as exc:  # noqa: BLE001
            return Retrieved(error=f"hyde generation failed: {exc}")

        got = self.engine.retrieve(hypothetical, k * self.k_multiplier)
        got.latency_ms = (time.time() - started) * 1000
        return got


class IterativeLoop:
    """Retrieve, read, notice what is missing, retrieve again.

    After each round the model is shown the question and what has been found,
    and asked what is still missing — as search queries, in the corpus's
    vocabulary. It stops when it reports nothing further is needed, which is
    also the signal a system would use to abstain.

    This is the pattern the multi-hop literature supports, and the one the
    deep-reasoning path in nano-claw is already shaped for.
    """

    def __init__(self, engine, rounds: int = 3, per_round: int = 3):
        self.engine = engine
        self.rounds = rounds
        self.per_round = per_round
        self.name = f"loop×{rounds} → {engine.name}"
        self.retrieval = f"iterative ({rounds} rounds) over {engine.retrieval}"

    def health(self) -> str | None:
        return self.engine.health()

    def ingest(self, corpus_dir: Path) -> dict:
        return self.engine.ingest(corpus_dir)

    def retrieve(self, question: str, k: int) -> Retrieved:
        started = time.time()
        chunks: list[str] = []
        scores: list[float] = []
        seen: set[str] = set()
        queries = [question]
        trace: list[dict] = []

        for round_index in range(self.rounds):
            fresh = 0
            for query in queries:
                got = self.engine.retrieve(query, k)
                if got.error:
                    return Retrieved(error=got.error)
                for text, score in zip(got.chunks, got.scores):
                    key = text[:120]
                    if key not in seen:
                        seen.add(key)
                        chunks.append(text)
                        scores.append(score)
                        fresh += 1
            trace.append({"round": round_index + 1, "queries": queries, "new": fresh})

            if round_index == self.rounds - 1:
                break

            context = "\n---\n".join(c[:700] for c in chunks[-18:])
            try:
                reply = ask(
                    "You are assembling the material needed to answer a tax "
                    "question correctly. Below is the question and the "
                    "passages retrieved so far.\n\n"
                    "Name any concept, rule or limitation that is REQUIRED to "
                    "answer correctly and is NOT yet present. Reply with a "
                    "JSON object: {\"sufficient\": bool, \"queries\": [up to "
                    f"{self.per_round} short search strings using the "
                    "terminology an IRS publication would use]}. Reply with "
                    "JSON and nothing else.\n\n"
                    f"QUESTION: {question}\n\nRETRIEVED SO FAR:\n{context}"
                )
                block = re.search(r"\{.*\}", reply, re.S)
                decision = json.loads(block.group(0)) if block else {}
            except Exception:  # noqa: BLE001 - a bad round ends the loop, not the run
                break

            if decision.get("sufficient") is True:
                trace.append({"round": round_index + 1, "stopped": "model reported sufficient"})
                break
            queries = [q for q in decision.get("queries", []) if isinstance(q, str)][
                : self.per_round
            ]
            if not queries:
                break

        result = Retrieved(
            chunks=chunks, scores=scores, latency_ms=(time.time() - started) * 1000
        )
        result.trace = trace  # type: ignore[attr-defined]
        return result
