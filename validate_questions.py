"""Check the answer key against the corpus before trusting any result.

A required concept that does not appear anywhere in the source material is not
a retrieval failure waiting to be measured — it is a bug in the test. Running
the bakeoff without this check would produce confident numbers about nothing.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).parent
CORPUS = ROOT / "corpus"


def main() -> int:
    spec = json.loads((ROOT / "questions.json").read_text())
    texts = {p.stem: p.read_text() for p in CORPUS.glob("*.txt")}
    if not texts:
        print("no extracted corpus text found — run the extraction step first")
        return 1

    print(f"corpus: {', '.join(sorted(texts))}\n")
    problems = 0

    for question in spec["questions"]:
        concepts = question["required_concepts"]
        if not concepts:
            print(f"  {question['id']:<24} (abstain question — no concepts required)")
            continue

        found_in: list[str] = []
        for concept in concepts:
            pattern = re.compile(concept["pattern"], re.IGNORECASE)
            hits = [name for name, text in texts.items() if pattern.search(text)]
            if not hits:
                print(
                    f"  ✗ {question['id']}: '{concept['name']}' "
                    f"(/{concept['pattern']}/) appears in NO corpus document"
                )
                problems += 1
            else:
                found_in.append(f"{concept['name']}→{'+'.join(sorted(hits))}")

        if len(found_in) == len(concepts):
            spread = {d for entry in found_in for d in entry.split("→")[1].split("+")}
            marker = "  (cross-document)" if len(spread) > 1 else ""
            print(f"  ✓ {question['id']:<24} {len(concepts)} concepts{marker}")

    print()
    if problems:
        print(f"{problems} concept(s) not in the corpus — fix the key before running")
        return 1
    print("answer key is grounded: every required concept exists in the corpus")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
