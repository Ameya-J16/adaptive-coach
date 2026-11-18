"""Regenerate the golden set's retrieved contexts from the real RAG index.

Run with: python -m scripts.refresh_eval_contexts

The golden set's `retrieved_context` fields must reflect what the planner agent
would actually retrieve, otherwise the eval measures the critic against evidence
the system never sees. This runs the real retriever for each context slot and
caches the result to evals/retrieved_contexts.json, so eval runs stay
reproducible (and free) until the index changes.

Re-run this after any change to the corpus or rag/ingest.py.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from rag.retriever import retrieve

OUT_PATH = Path(__file__).parent.parent / "evals" / "retrieved_contexts.json"

# Queries mirror what progression_planner asks for in each fatigue state,
# phrased the way that agent phrases them in live runs.
QUERIES = {
    "KB_INCREASE": "progressive overload volume increase for intermediate lifters hypertrophy",
    "KB_DELOAD": "deload protocol training load reduction after overreaching",
    "KB_AMBER": "managing accumulated fatigue maintain training load avoid intensity increases",
    "KB_BEGINNER": "beginner resistance training starting loads linear progression novice",
    "KB_ENDURANCE": "endurance training intensity distribution polarized concurrent strength training",
    "KB_NUTRITION": "protein intake nutrient timing carbohydrate periodisation athletes",
}


def main() -> None:
    contexts = {}
    for name, query in QUERIES.items():
        text = retrieve(query, k=4)
        contexts[name] = text
        first_line = text.splitlines()[0] if text else "(empty)"
        print(f"{name:16s} {len(text):6,d} chars  {first_line[:72]}")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(contexts, indent=2), encoding="utf-8")
    print(f"\nWrote {OUT_PATH}")


if __name__ == "__main__":
    main()
