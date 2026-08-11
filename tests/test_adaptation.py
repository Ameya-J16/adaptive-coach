"""Tests for the adaptation agent's write tools and its safety gate.

No API calls: the tools are pure Python, and the gate's semantics (reject only
violations the agent *introduced*) is the part most worth pinning down, because
getting it wrong silently discards correct edits.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from graph.adaptation_tools import build_adaptation_tools
from graph.plan_checks import check_plan


def _ex(name, sets=4, reps=8, weight=60.0, rpe=7.5, muscles=("quads", "glutes")):
    return {"name": name, "sets": sets, "reps": reps, "weight_kg": weight,
            "rpe_target": rpe, "muscle_groups": list(muscles)}


def _plan():
    rest = {"session_type": "rest", "exercises": [], "notes": ""}
    return {
        "week_start_date": "2026-08-10",
        "days": {
            "monday": {"session_type": "strength",
                       "exercises": [_ex("Back Squat"), _ex("Leg Press")], "notes": "Lower"},
            "tuesday": rest.copy(),
            "wednesday": {"session_type": "strength",
                          "exercises": [_ex("Bench Press", muscles=("chest",))], "notes": "Upper"},
            "thursday": rest.copy(),
            "friday": rest.copy(),
            "saturday": rest.copy(),
            "sunday": rest.copy(),
        },
        "overall_notes": "",
    }


class TestWriteTools:
    def _tools(self):
        tools, state = build_adaptation_tools(_plan())
        return {t.name: t for t in tools}, state

    def test_swap_exercise_replaces_and_logs(self):
        tools, state = self._tools()
        out = tools["swap_exercise"].invoke({
            "day": "monday", "old_exercise": "Back Squat", "new_exercise": "Goblet Squat",
            "sets": 3, "reps": 10, "weight_kg": 24.0, "rpe_target": 7.0,
            "muscle_groups": "quads,glutes", "reason": "knee pain",
        })
        names = [e["name"] for e in state["plan"]["days"]["monday"]["exercises"]]
        assert "Goblet Squat" in names and "Back Squat" not in names
        assert "Done" in out
        assert len(state["changes"]) == 1

    def test_swap_unknown_exercise_is_a_no_op(self):
        tools, state = self._tools()
        out = tools["swap_exercise"].invoke({
            "day": "monday", "old_exercise": "Deadlift", "new_exercise": "Hip Thrust",
            "sets": 3, "reps": 10, "weight_kg": 60.0, "rpe_target": 7.0,
            "muscle_groups": "glutes", "reason": "x",
        })
        assert "No exercise matching" in out
        assert state["changes"] == []

    def test_set_day_to_rest(self):
        tools, state = self._tools()
        tools["set_day_to_rest"].invoke({"day": "monday", "reason": "illness"})
        assert state["plan"]["days"]["monday"]["exercises"] == []
        assert state["plan"]["days"]["monday"]["session_type"] == "rest"

    def test_move_session_clears_origin(self):
        tools, state = self._tools()
        tools["move_session"].invoke({"from_day": "monday", "to_day": "thursday",
                                      "reason": "gym closed"})
        days = state["plan"]["days"]
        assert days["thursday"]["exercises"] and not days["monday"]["exercises"]

    def test_move_onto_occupied_day_is_refused(self):
        tools, state = self._tools()
        out = tools["move_session"].invoke({"from_day": "monday", "to_day": "wednesday",
                                            "reason": "x"})
        assert "already has a session" in out
        assert state["plan"]["days"]["monday"]["exercises"], "origin must be untouched"

    def test_adjust_load(self):
        tools, state = self._tools()
        tools["adjust_exercise_load"].invoke({
            "day": "monday", "exercise": "Leg Press", "weight_kg": 80.0,
            "rpe_target": 6.0, "reason": "bad sleep",
        })
        leg = [e for e in state["plan"]["days"]["monday"]["exercises"]
               if e["name"] == "Leg Press"][0]
        assert leg["weight_kg"] == 80.0 and leg["rpe_target"] == 6.0

    def test_invalid_day_rejected(self):
        tools, state = self._tools()
        assert "not a day" in tools["set_day_to_rest"].invoke({"day": "someday",
                                                               "reason": "x"})
        assert state["changes"] == []

    def test_original_plan_is_not_mutated(self):
        original = _plan()
        tools, state = build_adaptation_tools(original)
        {t.name: t for t in tools}["set_day_to_rest"].invoke({"day": "monday", "reason": "x"})
        assert original["days"]["monday"]["exercises"], "tools must edit a copy"


class TestSafetyGateDiff:
    """The gate rejects only violations the agent introduced."""

    GREEN = {"fatigue_level": "green", "acwr": 1.0, "acwr_status": "computed"}

    def test_preexisting_violation_is_not_blamed_on_the_agent(self):
        # A plan that already violates coherence: 'deload' with no volume cut.
        plan = _plan()
        decision = {"action": "deload", "exercises_to_swap": []}
        history = [{"date": "2026-07-20",
                    "exercises": [{"sets": 1, "reps": 1, "weight_kg": 4000}]}]

        before = set(check_plan(plan, self.GREEN, decision, history)["violations"])
        assert before, "fixture should already be in violation"

        tools, state = build_adaptation_tools(plan)
        {t.name: t for t in tools}["swap_exercise"].invoke({
            "day": "monday", "old_exercise": "Back Squat", "new_exercise": "Goblet Squat",
            "sets": 4, "reps": 8, "weight_kg": 60.0, "rpe_target": 7.5,
            "muscle_groups": "quads,glutes", "reason": "knee pain",
        })
        after = check_plan(state["plan"], self.GREEN, decision, history)["violations"]
        introduced = [v for v in after if v not in before]
        assert introduced == [], "a like-for-like swap introduces nothing new"

    def test_agent_introducing_consecutive_heavy_days_is_caught(self):
        plan = _plan()
        decision = {"action": "maintain", "exercises_to_swap": []}
        before = set(check_plan(plan, self.GREEN, decision, [])["violations"])

        tools, state = build_adaptation_tools(plan)
        # Move the heavy lower session onto the day after another heavy day.
        state["plan"]["days"]["tuesday"] = {
            "session_type": "strength",
            "exercises": [_ex("Front Squat", rpe=8.5)],
            "notes": "",
        }
        after = check_plan(state["plan"], self.GREEN, decision, [])["violations"]
        introduced = [v for v in after if v not in before]
        assert any("consecutive" in v.lower() for v in introduced)
