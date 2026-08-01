"""The contract every engine in the bakeoff implements.

Deliberately narrow: ingest the same corpus, answer the same question, hand
back the text you would have put in front of the model. Everything that
differs between engines — chunk size, embedding model, lexical vs dense — is
inside the box and is exactly what the comparison is measuring.

Retrieval is scored rather than answers. An LLM judge on final answers is
noisy, expensive, and conflates two failures: not finding the material, and
mishandling material you did find. Concept recall isolates the first, which is
the one that decides whether an architecture can work at all — a model cannot
reason from a passage it was never shown.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol


@dataclass
class Retrieved:
    """What an engine hands back for one question."""

    chunks: list[str] = field(default_factory=list)
    scores: list[float] = field(default_factory=list)
    latency_ms: float = 0.0
    error: str | None = None

    @property
    def text(self) -> str:
        return "\n\n".join(self.chunks)

    @property
    def top_score(self) -> float:
        return self.scores[0] if self.scores else 0.0


class Engine(Protocol):
    """An engine under test."""

    name: str
    retrieval: str  # "lexical" | "dense" | "hybrid" | "none (full context)"

    def health(self) -> str | None:
        """Return None if ready, or a human-readable reason it is not."""

    def ingest(self, corpus_dir: Path) -> dict:
        """Load the corpus. Idempotent where the engine allows it."""

    def retrieve(self, question: str, k: int) -> Retrieved:
        """Return the top-k passages this engine would give the model."""
