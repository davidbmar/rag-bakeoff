"""Adapters for the engines that can actually take this corpus.

Of the six "upload documents, ask questions" implementations in the portfolio,
only three can ingest arbitrary PDFs and be queried over HTTP with the same
corpus. The others are excluded for stated reasons rather than quietly:

  github-portfolio-search  indexes GitHub repos through the GitHub API; there
                           is no path for an external PDF.
  browser-RAG (FSM/Iris)   runs entirely in the browser over prebuilt JSONL
                           conversation packs; no server, no document upload.
  persona-rag              can ingest, but only PDFs converted into its own
                           evidence-item JSON schema, and it has been idle
                           since May. Includable with a day of adapter work;
                           excluded here to keep the comparison honest about
                           what was actually run.
  evidence-qa              private and not cloned locally. On paper it is the
                           closest architecture to right (BGE + cross-encoder
                           + abstention) and is the one worth cloning next.
"""

from __future__ import annotations

import time
from pathlib import Path

import httpx

from adapters.base import Retrieved


def _score_of(evidence: dict) -> float:
    """Pull a comparable relevance number out of an evidence item."""

    score = evidence.get("score")
    if isinstance(score, dict):
        # raw_score before normalized_score, deliberately. The normalized value
        # is scaled per query so the top hit is always 1.0 — fine for ordering
        # within one result set, meaningless for comparing across questions,
        # which is exactly what an abstention decision has to do.
        for key in ("raw_score", "normalized_score"):
            if isinstance(score.get(key), (int, float)):
                return float(score[key])
        return 0.0
    return float(score) if isinstance(score, (int, float)) else 0.0


class IntelligencePlatform:
    """The engine live in production today: SQLite FTS5 + BM25, lexical only."""

    name = "intelligence-platform"
    retrieval = "lexical (BM25/FTS5)"

    def __init__(self, base_url: str = "http://127.0.0.1:8000", tenant: str = "bakeoff"):
        self.base_url = base_url.rstrip("/")
        self.tenant = tenant
        self.collection = "bakeoff-irs"

    def health(self) -> str | None:
        try:
            r = httpx.get(f"{self.base_url}/healthz", timeout=5)
            return None if r.status_code == 200 else f"healthz returned {r.status_code}"
        except Exception as exc:  # noqa: BLE001
            return f"unreachable at {self.base_url} ({type(exc).__name__})"

    def _policy(self, permission: str) -> dict:
        return {
            "tenant_id": self.tenant,
            "principal_id": "bakeoff",
            "permissions": [permission],
        }

    def ingest(self, corpus_dir: Path) -> dict:
        ingested = []
        for path in sorted(corpus_dir.glob("*.txt")):
            body = {
                "tenant_id": self.tenant,
                "policy": self._policy("knowledge:ingest"),
                "source_ref": f"bakeoff://{path.name}",
                "document_key": f"bakeoff:{path.stem}",
                "title": path.stem,
                "text": path.read_text(),
                "collection_ids": [self.collection],
                "source_kind": "bakeoff",
                "metadata": {},
            }
            # Ingestion here is synchronous and segments the whole document
            # before answering; the large publications legitimately take minutes.
            r = httpx.post(f"{self.base_url}/v1/ingest/text", json=body, timeout=900)
            r.raise_for_status()
            ingested.append({"doc": path.stem, "segments": len(r.json()["segment_ids"])})
        return {"documents": ingested}

    def retrieve(self, question: str, k: int) -> Retrieved:
        started = time.time()
        try:
            r = httpx.post(
                f"{self.base_url}/v1/retrieve",
                json={
                    "query_id": "bakeoff",
                    "text": question,
                    "policy": self._policy("knowledge:retrieve"),
                    "scope": {
                        "tenant_id": self.tenant,
                        "collection_ids": [self.collection],
                    },
                    "limit": k,
                    "candidate_pool": max(40, k * 8),
                },
                timeout=60,
            )
            r.raise_for_status()
            evidence = r.json().get("evidence", [])
            # `score` is a structured object here — method, raw_score and a
            # normalized_score — not a bare float. The normalized value is the
            # comparable one across engines.
            return Retrieved(
                chunks=[e["text"] for e in evidence],
                scores=[_score_of(e) for e in evidence],
                latency_ms=(time.time() - started) * 1000,
            )
        except Exception as exc:  # noqa: BLE001
            return Retrieved(error=f"{type(exc).__name__}: {exc}")


class VoiceOptimalRAG:
    """Dense vectors: nomic-embed-text-v1.5 (768d) in LanceDB."""

    name = "voice-optimal-RAG"
    retrieval = "dense (nomic-embed-text-v1.5)"

    def __init__(self, base_url: str = "http://127.0.0.1:8100"):
        self.base_url = base_url.rstrip("/")

    def health(self) -> str | None:
        try:
            r = httpx.get(f"{self.base_url}/health", timeout=5)
            return None if r.status_code == 200 else f"health returned {r.status_code}"
        except Exception as exc:  # noqa: BLE001
            return f"unreachable at {self.base_url} ({type(exc).__name__})"

    def ingest(self, corpus_dir: Path) -> dict:
        ingested = []
        for path in sorted(corpus_dir.glob("*.pdf")):
            # Its own PDF parser is part of what is being compared, so it gets
            # the PDF rather than the text the other engine was handed.
            with path.open("rb") as handle:
                r = httpx.post(
                    f"{self.base_url}/upload",
                    files={"files": (path.name, handle, "application/pdf")},
                    timeout=1800,
                )
            r.raise_for_status()
            for doc in r.json().get("documents", []):
                ingested.append({"doc": doc.get("filename"), "segments": doc.get("chunks")})
        return {"documents": ingested}

    def retrieve(self, question: str, k: int) -> Retrieved:
        started = time.time()
        try:
            r = httpx.post(
                f"{self.base_url}/query",
                json={"query": question, "top_k": k},
                timeout=120,
            )
            r.raise_for_status()
            results = r.json().get("results", [])
            return Retrieved(
                chunks=[x["text"] for x in results],
                scores=[float(x.get("score", 0.0)) for x in results],
                latency_ms=(time.time() - started) * 1000,
            )
        except Exception as exc:  # noqa: BLE001
            return Retrieved(error=f"{type(exc).__name__}: {exc}")


class FullContext:
    """The control: no retrieval at all — hand over the whole corpus.

    Included because it is the option most likely to win on a bounded corpus,
    and because it makes the others' numbers legible: its concept recall is
    100% by construction, so any engine scoring below it is losing information
    that was available. What it costs is tokens and latency, which is the
    honest trade and is reported alongside.
    """

    name = "full-context (control)"
    retrieval = "none — entire corpus in the prompt"

    def __init__(self, subset: list[str] | None = None):
        # A curated per-topic subset is the realistic form of this option:
        # p527 + p925 is ~80k tokens and answers every rental question here.
        self.subset = subset
        self._text = ""

    def health(self) -> str | None:
        return None

    def __post_init__(self) -> None:  # pragma: no cover - dataclass parity
        pass

    def _load(self, corpus_dir: Path) -> dict:
        parts, total = [], []
        for path in sorted(corpus_dir.glob("*.txt")):
            if self.subset and path.stem not in self.subset:
                continue
            parts.append(path.read_text())
            total.append({"doc": path.stem, "segments": 1})
        self._text = "\n\n".join(parts)
        return {"documents": total, "chars": len(self._text)}

    def ingest(self, corpus_dir: Path) -> dict:
        return self._load(corpus_dir)

    def retrieve(self, question: str, k: int) -> Retrieved:
        # Load on demand. Reporting 0% recall merely because --ingest was not
        # passed would make the control look like the worst engine in the
        # bakeoff, which is the exact opposite of the truth.
        if not self._text:
            self._load(Path(__file__).resolve().parent.parent / "corpus")
        return Retrieved(chunks=[self._text], scores=[1.0], latency_ms=0.0)
