"""Tests for critic loop routing and the deterministic plan-check rule engine."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from datetime import datetime, timedelta

from graph.graph import should_replan
from graph.plan_checks import check_plan, plan_volume


class TestCriticLoopLogic:
    def test_low_score_triggers_replan(self):
        state = {"critic_score": 0.60, "loop_count": 1}
        assert should_replan(state) == "replan"

    def test_high_score_exits_loop(self):
        state = {"critic_score": 0.80, "loop_count": 0}
        assert should_replan(state) == "end"

    def test_exact_threshold_exits(self):
        state = {"critic_score": 0.75, "loop_count": 0}
        assert should_replan(state) == "end"

    def test_loop_exits_after_3_iterations(self):
        state = {"critic_score": 0.40, "loop_count": 3}
        assert should_replan(state) == "end"

    def test_loop_count_boundary(self):
        state = {"critic_score": 0.50, "loop_count": 2}
        assert should_replan(state) == "replan"

    def test_loop_count_4_exits(self):
        state = {"critic_score": 0.10, "loop_count": 4}
        assert should_replan(state) == "end"


def _day(exercises: list[dict], session_type: str = "strength") -> dict:
    return {"session_type": session_type, "exercises": exercises, "notes": ""}


def _rest() -> dict:
    return _day([], "rest")


def _ex(name: str, sets: int, reps: int, weight: float, rpe: float, muscles: list[str]) -> dict:
    return {
        "name": name, "sets": sets, "reps": reps, "weight_kg": weight,
        "rpe_target": rpe, "muscle_groups": muscles,
    }


def _plan(**days) -> dict:
    base = {d: _rest() for d in
            ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]}
    base.update(days)
    return {"week_start_date": "2026-08-10", "days": base, "overall_notes": ""}


def _history_weeks(weekly_volume: float, weeks: int = 4) -> list[dict]:
    """One session per prior calendar week carrying the full weekly volume."""
    today = datetime.now().date()
    this_monday = today - timedelta(days=today.weekday())
    return [
        {
            "date": (this_monday - timedelta(weeks=w + 1)).isoformat(),
            "exercises": [{"sets": 1, "reps": 1, "weight_kg": weekly_volume}],
        }
        for w in range(weeks)
    ]


GREEN = {"fatigue_level": "green", "acwr": 1.0, "acwr_status": "computed"}
RED = {"fatigue_level": "red", "acwr": 1.7, "acwr_status": "computed"}


class TestPlanChecks:
    def test_consecutive_heavy_same_muscles_zeroes_safety(self):
        heavy_squat = _day([_ex("Squat", 5, 5, 100, 9.0, ["quads", "glutes"])])
        plan = _plan(monday=heavy_squat, tuesday=heavy_squat)
        checks = check_plan(plan, GREEN, {"action": "maintain"}, [])
        assert checks["safety_cap"] == 0.0
        assert any("consecutive" in v.lower() for v in checks["violations"])

    def test_alternating_muscle_groups_is_safe(self):
        lower = _day([_ex("Squat", 5, 5, 100, 8.0, ["quads"])])
        upper = _day([_ex("Bench Press", 5, 5, 80, 8.0, ["chest"])])
        plan = _plan(monday=lower, tuesday=upper)
        checks = check_plan(plan, GREEN, {"action": "maintain"}, [])
        assert checks["safety_cap"] == 1.0

    def test_red_fatigue_with_four_hard_days_capped(self):
        hard = _day([_ex("Squat", 3, 5, 100, 8.0, ["quads"])])
        hard2 = _day([_ex("Bench Press", 3, 5, 80, 8.0, ["chest"])])
        hard3 = _day([_ex("Deadlift", 3, 5, 120, 8.0, ["hamstrings"])])
        hard4 = _day([_ex("Overhead Press", 3, 5, 50, 8.0, ["shoulders"])])
        plan = _plan(monday=hard, wednesday=hard2, friday=hard3, sunday=hard4)
        checks = check_plan(plan, RED, {"action": "deload"}, [])
        assert checks["safety_cap"] <= 0.7

    def test_deload_with_rpe_above_nine_capped(self):
        plan = _plan(monday=_day([_ex("Squat", 3, 5, 120, 9.5, ["quads"])]))
        checks = check_plan(plan, RED, {"action": "deload"}, [])
        assert checks["safety_cap"] <= 0.8
        assert any("9.5" in v for v in checks["violations"])

    def test_deload_without_volume_reduction_zeroes_coherence(self):
        history = _history_weeks(5000)
        # planned volume == baseline: not a deload at all
        plan = _plan(monday=_day([_ex("Squat", 5, 10, 100, 6.0, ["quads"])]))
        assert plan_volume(plan) == 5000
        checks = check_plan(plan, RED, {"action": "deload"}, history)
        assert checks["coherence_cap"] == 0.0

    def test_real_deload_passes_coherence(self):
        history = _history_weeks(5000)
        plan = _plan(monday=_day([_ex("Squat", 5, 6, 100, 6.0, ["quads"])]))  # 3000kg = 60%
        checks = check_plan(plan, RED, {"action": "deload"}, history)
        assert checks["coherence_cap"] == 1.0

    def test_increase_without_increase_capped(self):
        history = _history_weeks(5000)
        plan = _plan(monday=_day([_ex("Squat", 5, 8, 100, 7.0, ["quads"])]))  # 4000kg
        checks = check_plan(plan, GREEN, {"action": "increase"}, history)
        assert checks["coherence_cap"] <= 0.4

    def test_swapped_exercise_reappearing_capped(self):
        plan = _plan(monday=_day([_ex("Barbell Squat", 3, 8, 80, 7.0, ["quads"])]))
        decision = {"action": "maintain", "exercises_to_swap": ["Squat"]}
        checks = check_plan(plan, GREEN, decision, [])
        assert checks["coherence_cap"] <= 0.5

    def test_no_history_skips_volume_checks(self):
        plan = _plan(monday=_day([_ex("Squat", 3, 8, 40, 6.0, ["quads"])]))
        checks = check_plan(plan, {"fatigue_level": "amber", "acwr": None}, {"action": "maintain"}, [])
        assert checks["coherence_cap"] == 1.0
        assert checks["safety_cap"] == 1.0
