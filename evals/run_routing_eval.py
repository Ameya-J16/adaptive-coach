"""Supervisor routing evaluation.

Offline (free):  python -m evals.run_routing_eval --offline
Live (calls the supervisor's model):  python -m evals.run_routing_eval

Offline checks the golden set itself and the cases decided in code (no plan
-> plan). Live sends every request through the real supervisor and reports
routing accuracy, a confusion matrix and the wrong-path cases. Results are
written to evals/results/routing_eval_<timestamp>.json.

The costly mistake is routing a change request to "plan" (a full rewrite of a
week mid-progression) or a pain report to "explain" (nothing changes), so both
are counted separately.
"""
import argparse
import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from evals.routing_set import CASES, ROUTES  # noqa: E402

RESULTS_DIR = Path(__file__).parent / "results"


def validate(cases: list[dict]) -> list[str]:
    problems = []
    names = [c["name"] for c in cases]
    if len(names) != len(set(names)):
        problems.append("duplicate case names")
    for c in cases:
        if c["expected"] not in ROUTES:
            problems.append(f"{c['name']}: unknown route {c['expected']!r}")
        if not c.get("plan_exists", True) and c["expected"] != "plan":
            problems.append(f"{c['name']}: without a plan the only valid route is 'plan'")
    for route in ROUTES:
        if not any(c["expected"] == route for c in cases):
            problems.append(f"no case expects {route!r}")
    return problems


def route_case(case: dict) -> dict:
    """Run one case through the real supervisor node."""
    import graph.nodes.supervisor as sup

    plan_exists = case.get("plan_exists", True)
    original = sup.has_plan
    sup.has_plan = lambda user_id: plan_exists  # the eval controls the athlete's state
    try:
        out = sup.supervisor({
            "user_id": "eval",
            "user_request": case["request"],
            "user_profile": {"goals": "strength", "experience_level": "intermediate"},
            "current_week_log": [],
            "fatigue_report": {"fatigue_level": "amber"},
        })
    finally:
        sup.has_plan = original
    return out


def run_live(cases: list[dict]) -> dict:
    rows = []
    for case in cases:
        out = route_case(case)
        got = out.get("route")
        rows.append({
            "name": case["name"], "request": case["request"], "expected": case["expected"],
            "got": got, "correct": got == case["expected"], "reason": out.get("route_reason", ""),
        })
        mark = "ok " if got == case["expected"] else "XX "
        print(f"  {mark}{case['name']:14s} expected {case['expected']:7s} got {got}")

    confusion = Counter(f"{r['expected']}->{r['got']}" for r in rows)
    from config.models import describe as describe_llm

    return {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "llm_backend": describe_llm(),
        "n_cases": len(rows),
        "metrics": {
            "accuracy": round(sum(r["correct"] for r in rows) / len(rows), 3),
            "change_request_sent_to_plan": confusion["adapt->plan"],
            "pain_or_disruption_sent_to_explain": confusion["adapt->explain"],
            "question_that_changed_the_plan": confusion["explain->adapt"] + confusion["explain->plan"],
        },
        "confusion": dict(confusion),
        "cases": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the supervisor routing eval")
    parser.add_argument("--offline", action="store_true", help="validate the set only (no API)")
    args = parser.parse_args()

    problems = validate(CASES)
    if problems:
        print("ROUTING SET INVALID:")
        for p in problems:
            print(f"  - {p}")
        sys.exit(1)
    print(f"Routing set valid: {len(CASES)} cases.")
    if args.offline:
        return

    from config.models import describe as describe_llm
    print(f"LLM backend: {describe_llm()}\n")
    report = run_live(CASES)

    RESULTS_DIR.mkdir(exist_ok=True)
    out = RESULTS_DIR / f"routing_eval_{report['timestamp'].replace(':', '-')}.json"
    with open(out, "w") as f:
        json.dump(report, f, indent=2)
    print("\nMETRICS")
    for k, v in report["metrics"].items():
        print(f"  {k:36s}: {v}")
    print(f"\nFull report: {out}")


if __name__ == "__main__":
    main()
