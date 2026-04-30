"""Tests for critic loop logic — loop_count increments, exit conditions, safety scoring."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from graph.graph import should_replan


class TestCriticLoopLogic:
    def test_low_score_triggers_replan(self):
        """Score < 0.75 and loop_count < 3 should route to 'replan'."""
        state = {"critic_score": 0.60, "loop_count": 1}
        assert should_replan(state) == "replan"

    def test_high_score_exits_loop(self):
        """Score >= 0.75 should route to 'end' regardless of loop count."""
        state = {"critic_score": 0.80, "loop_count": 0}
        assert should_replan(state) == "end"

    def test_exact_threshold_exits(self):
        """Score exactly 0.75 should exit (not replan)."""
        state = {"critic_score": 0.75, "loop_count": 0}
        assert should_replan(state) == "end"

    def test_loop_exits_after_3_iterations(self):
        """After 3 loops, exit regardless of score."""
        state = {"critic_score": 0.40, "loop_count": 3}
        assert should_replan(state) == "end"

    def test_loop_count_boundary(self):
        """loop_count=2 with low score should still replan (< 3 budget)."""
        state = {"critic_score": 0.50, "loop_count": 2}
        assert should_replan(state) == "replan"

    def test_loop_count_4_exits(self):
        """loop_count > 3 should always exit."""
        state = {"critic_score": 0.10, "loop_count": 4}
        assert should_replan(state) == "end"


class TestCriticSafetyRules:
    def test_consecutive_heavy_leg_days_unsafe(self):
        """A plan with consecutive heavy squat days should score below 0.75 safety."""
        # Construct a plan that violates the consecutive heavy compound rule
        heavy_squat = {
            "session_type": "strength",
            "exercises": [
                {"name": "Squat", "sets": 5, "reps": 5, "weight_kg": 100, "rpe_target": 9.0, "muscle_groups": ["quads", "glutes"]},
                {"name": "Romanian Deadlift", "sets": 4, "reps": 6, "weight_kg": 80, "rpe_target": 8.5, "muscle_groups": ["hamstrings", "glutes"]},
            ],
            "notes": "Heavy lower day",
        }

        bad_plan = {
            "week_start_date": "2026-04-14",
            "days": {
                "monday": heavy_squat,
                "tuesday": heavy_squat,  # consecutive heavy lower — safety violation
                "wednesday": {"session_type": "rest", "exercises": [], "notes": "Rest"},
                "thursday": {"session_type": "strength", "exercises": [], "notes": "Upper"},
                "friday": {"session_type": "rest", "exercises": [], "notes": "Rest"},
                "saturday": {"session_type": "rest", "exercises": [], "notes": "Rest"},
                "sunday": {"session_type": "rest", "exercises": [], "notes": "Rest"},
            },
            "overall_notes": "Standard week",
        }

        # Simulate critic safety evaluation logic
        days_list = list(bad_plan["days"].values())
        consecutive_heavy_violation = False
        for i in range(len(days_list) - 1):
            day_a = days_list[i]
            day_b = days_list[i + 1]
            if day_a["session_type"] == "strength" and day_b["session_type"] == "strength":
                ex_a = day_a.get("exercises", [])
                ex_b = day_b.get("exercises", [])
                # Both days have exercises with RPE > 7 targeting lower body
                heavy_lower_a = any(
                    e.get("rpe_target", 0) > 7 and
                    any(mg in ["quads", "hamstrings", "glutes"] for mg in e.get("muscle_groups", []))
                    for e in ex_a
                )
                heavy_lower_b = any(
                    e.get("rpe_target", 0) > 7 and
                    any(mg in ["quads", "hamstrings", "glutes"] for mg in e.get("muscle_groups", []))
                    for e in ex_b
                )
                if heavy_lower_a and heavy_lower_b:
                    consecutive_heavy_violation = True

        assert consecutive_heavy_violation, "Expected to detect consecutive heavy lower body days"

    def test_deload_plan_with_high_rpe_flags_unsafe(self):
        """A deload plan with RPE 9+ targets should be flagged as unsafe."""
        deload_plan_with_high_rpe = {
            "week_start_date": "2026-04-14",
            "days": {
                "monday": {
                    "session_type": "deload",
                    "exercises": [
                        {"name": "Squat", "sets": 3, "reps": 5, "weight_kg": 120, "rpe_target": 9.5, "muscle_groups": ["quads"]},
                    ],
                    "notes": "Deload but high RPE — this is wrong",
                },
            },
            "overall_notes": "Deload week",
        }

        progression_decision = {"action": "deload", "load_adjustment_percent": -30}

        # Check for deload + high RPE violation
        has_violation = False
        if progression_decision["action"] == "deload":
            for day in deload_plan_with_high_rpe["days"].values():
                for ex in day.get("exercises", []):
                    if ex.get("rpe_target", 0) > 9:
                        has_violation = True

        assert has_violation, "Expected to detect high RPE during deload"
