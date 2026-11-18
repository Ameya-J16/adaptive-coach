"""Guard: the deterministic critic layer must always match the golden labels.

This is the free half of the eval harness (no API calls) — it keeps the rule
engine honest against the hand-labeled dataset on every test run.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from evals.golden_set import build_cases
from evals.run_eval import validate_offline
from graph.plan_checks import apply_caps


def test_golden_set_offline_validation():
    problems = validate_offline(build_cases())
    assert not problems, "\n".join(problems)


def test_any_violation_forces_score_below_replan_threshold():
    """A plan with a mechanical violation must never average its way past 0.75,
    even if the judge hands out perfect scores everywhere else."""
    perfect = {"safety": 1.0, "coherence": 1.0, "groundedness": 1.0, "goal_alignment": 1.0}
    checks = {"safety_cap": 0.0, "coherence_cap": 1.0, "violations": ["SAFETY: something"]}
    _, overall = apply_caps(perfect, checks)
    assert overall < 0.75


def test_no_violations_leaves_judge_scores_untouched():
    judge = {"safety": 0.9, "coherence": 0.8, "groundedness": 0.7, "goal_alignment": 1.0}
    checks = {"safety_cap": 1.0, "coherence_cap": 1.0, "violations": []}
    scores, overall = apply_caps(judge, checks)
    assert scores == judge
    assert overall == 0.85
