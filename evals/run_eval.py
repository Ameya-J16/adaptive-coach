"""Critic evaluation runner.

Offline mode (free, no API):   python -m evals.run_eval --offline
Live mode (calls the critic):  python -m evals.run_eval

Offline validates the deterministic layer against the golden labels:
- GOOD cases must trigger zero violations
- MECHANICAL-BAD cases must trigger their expected violation

Live runs every case through the full critic (rule engine + LLM judge agent)
and reports:
- verdict accuracy / false-pass / false-fail on verdict-labeled cases
  (false-pass = a bad plan passing = the dangerous direction)
- per-dimension detection on judge-only cases (flawed dimension <= 0.6)
- score separation between good and flawed plans

Results are written to evals/results/critic_eval_<timestamp>.json.
"""
import argparse
import json
import statistics
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from evals.golden_set import DIMENSION_LOW_THRESHOLD, PASS_THRESHOLD, build_cases
from graph.plan_checks import check_plan

RESULTS_DIR = Path(__file__).parent / "results"


def validate_offline(cases: list[dict]) -> list[str]:
    """Check the deterministic layer against golden labels. Returns problems."""
    problems = []
    for case in cases:
        s = case["state"]
        checks = check_plan(
            s["weekly_plan"], s["fatigue_report"], s["progression_decision"],
            s["workout_history"],
        )
        if case["expected_verdict"] == "pass" and checks["violations"]:
            problems.append(
                f"{case['name']}: labeled GOOD but triggered violations: {checks['violations']}"
            )
        for needle in case["expect_violations"]:
            if not any(needle.lower() in v.lower() for v in checks["violations"]):
                problems.append(
                    f"{case['name']}: expected a violation mentioning '{needle}', "
                    f"got: {checks['violations'] or 'none'}"
                )
        if not case["expected_verdict"] and checks["violations"]:
            problems.append(
                f"{case['name']}: judge-only case must be mechanically clean, "
                f"but triggered: {checks['violations']}"
            )
    return problems


def run_live(cases: list[dict]) -> dict:
    from graph.nodes.critic import evaluate_plan

    rows = []
    for i, case in enumerate(cases, 1):
        print(f"\n[{i}/{len(cases)}] {case['name']} — {case['description']}")
        s = case["state"]
        evaluation = evaluate_plan(
            weekly_plan=s["weekly_plan"],
            fatigue_report=s["fatigue_report"],
            progression_decision=s["progression_decision"],
            user_profile=s["user_profile"],
            retrieved_context=s["retrieved_context"],
            workout_history=s["workout_history"],
        )
        verdict = "pass" if evaluation["overall"] >= PASS_THRESHOLD else "fail"
        row = {
            "name": case["name"],
            "description": case["description"],
            "expected_verdict": case["expected_verdict"],
            "expect_low_dimensions": case["expect_low_dimensions"],
            "overall": evaluation["overall"],
            "verdict": verdict,
            "scores": evaluation["scores"],
            "violations": evaluation["violations"],
            "issues_found": evaluation["issues_found"],
        }
        if case["expected_verdict"]:
            row["verdict_correct"] = verdict == case["expected_verdict"]
        if case["expect_low_dimensions"]:
            row["dimension_hits"] = {
                d: evaluation["scores"][d] <= DIMENSION_LOW_THRESHOLD
                for d in case["expect_low_dimensions"]
            }
        rows.append(row)
        print(f"    overall={evaluation['overall']:.3f} verdict={verdict} "
              f"scores={evaluation['scores']}")

    # ---- metrics ----
    labeled = [r for r in rows if r["expected_verdict"]]
    correct = [r for r in labeled if r["verdict_correct"]]
    false_pass = [r for r in labeled
                  if r["expected_verdict"] == "fail" and r["verdict"] == "pass"]
    false_fail = [r for r in labeled
                  if r["expected_verdict"] == "pass" and r["verdict"] == "fail"]

    judge_rows = [r for r in rows if r["expect_low_dimensions"]]
    dim_checks = [hit for r in judge_rows for hit in r["dimension_hits"].values()]

    good_scores = [r["overall"] for r in rows if r["expected_verdict"] == "pass"]
    flawed_scores = [r["overall"] for r in rows
                     if r["expected_verdict"] == "fail" or r["expect_low_dimensions"]]

    metrics = {
        "verdict_accuracy": len(correct) / len(labeled) if labeled else None,
        "false_pass_rate": len(false_pass) / len(labeled) if labeled else None,
        "false_fail_rate": len(false_fail) / len(labeled) if labeled else None,
        "dimension_detection_rate": (sum(dim_checks) / len(dim_checks)) if dim_checks else None,
        "mean_good_score": round(statistics.mean(good_scores), 3) if good_scores else None,
        "mean_flawed_score": round(statistics.mean(flawed_scores), 3) if flawed_scores else None,
        "min_good_score": min(good_scores) if good_scores else None,
        "max_flawed_score": max(flawed_scores) if flawed_scores else None,
    }
    if metrics["mean_good_score"] is not None and metrics["mean_flawed_score"] is not None:
        metrics["score_separation"] = round(
            metrics["mean_good_score"] - metrics["mean_flawed_score"], 3
        )

    from config.models import describe as describe_llm
    from rag.embeddings import describe as describe_embeddings

    return {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "llm_backend": describe_llm(),
        "embedding_backend": describe_embeddings(),
        "pass_threshold": PASS_THRESHOLD,
        "dimension_low_threshold": DIMENSION_LOW_THRESHOLD,
        "n_cases": len(rows),
        "metrics": metrics,
        "cases": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the critic golden-set eval")
    parser.add_argument("--offline", action="store_true",
                        help="only validate the deterministic layer (no API calls)")
    parser.add_argument("--only", help="run a single case by name")
    args = parser.parse_args()

    cases = build_cases()
    if args.only:
        cases = [c for c in cases if c["name"] == args.only]
        if not cases:
            sys.exit(f"no case named {args.only!r}")

    problems = validate_offline(cases)
    if problems:
        print("OFFLINE VALIDATION FAILED:")
        for p in problems:
            print(f"  - {p}")
        sys.exit(1)
    print(f"Offline validation passed: {len(cases)} cases, deterministic layer "
          "matches all golden labels.")
    if args.offline:
        return

    from config.models import describe as describe_llm
    print(f"LLM backend: {describe_llm()}")

    report = run_live(cases)

    RESULTS_DIR.mkdir(exist_ok=True)
    out = RESULTS_DIR / f"critic_eval_{report['timestamp'].replace(':', '-')}.json"
    with open(out, "w") as f:
        json.dump(report, f, indent=2)

    print("\n" + "=" * 64)
    print("METRICS")
    print("=" * 64)
    for k, v in report["metrics"].items():
        print(f"  {k:26s}: {v}")
    print(f"\nFull report: {out}")


if __name__ == "__main__":
    main()
