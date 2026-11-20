"""End-to-end: the real LangGraph pipeline, real storage (in a temp dir), real
rule engine — with a scripted model standing in for every agent.

Unit tests cover each piece; this catches what only shows up when they run
together (a library API change in the context loader once failed every
request at the first node while all unit tests stayed green).
"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from langchain_core.messages import AIMessage

DEMO_PERSONA = "red_overreach"


def _day(session_type, exercises=(), notes=""):
    return {"session_type": session_type, "exercises": list(exercises), "notes": notes}


def _ex(name, sets, reps, kg, rpe, muscles):
    return {"name": name, "sets": sets, "reps": reps, "weight_kg": kg,
            "rpe_target": rpe, "muscle_groups": muscles}


# A sensible deload week for the overreached persona: low volume, RPE <= 7,
# no back-to-back heavy days — so the rule engine has nothing to flag.
DELOAD_PLAN = {
    "week_start_date": "2026-11-16",
    "overall_notes": "Deload: volume cut, RPE capped at 7.",
    "days": {
        "monday": _day("strength", [_ex("Squat", 2, 5, 60, 6.5, ["quads", "glutes"]),
                                    _ex("Bench Press", 2, 5, 45, 6.5, ["chest", "triceps"])]),
        "tuesday": _day("rest"),
        "wednesday": _day("cardio", notes="30 min easy"),
        "thursday": _day("strength", [_ex("Romanian Deadlift", 2, 6, 50, 6.5, ["hamstrings"])]),
        "friday": _day("rest"),
        "saturday": _day("rest"),
        "sunday": _day("rest"),
    },
}

REPLIES = {
    "fatigue_analyst": {"fatigue_level": "red", "dominant_stress_pattern": "RPE drift",
                        "recommendation_hint": "deload"},
    "progression_planner": {"action": "deload", "load_adjustment_percent": -40,
                            "rationale": "Overreached; cut volume.", "exercises_to_swap": []},
    "plan_writer": DELOAD_PLAN,
    "critic": {"safety": 0.9, "coherence": 0.9, "groundedness": 0.85, "goal_alignment": 0.85,
               "feedback": "Solid deload.", "issues_found": []},
    "explainer": {"explanation": "You're deloading because your load spiked.",
                  "sources_used": ["Bell 2023"]},
    "adaptation_agent": {"summary": "Nothing needed changing.", "changes_made": []},
}


class NodeLLM:
    """Answers as whichever agent asked for it, without calling tools."""

    def __init__(self, node, route):
        self.node, self.route = node, route

    def bind_tools(self, tools):
        return self

    def invoke(self, messages, config=None):
        if self.node == "supervisor":
            return AIMessage(content=json.dumps({"route": self.route, "reason": "test"}))
        return AIMessage(content=json.dumps(REPLIES[self.node]))


@pytest.fixture
def pipeline(tmp_path, monkeypatch):
    """The compiled graph wired to temp storage and the scripted model."""
    import memory.plan_store as plan_store
    import memory.session_memory as session_memory
    import memory.workout_store as workout_store
    from scripts import seed_demo_data

    monkeypatch.setattr(workout_store, "DATA_DIR", tmp_path / "workouts")
    monkeypatch.setattr(seed_demo_data, "DATA_DIR", tmp_path / "workouts")
    monkeypatch.setattr(plan_store, "DATA_DIR", tmp_path / "plans")
    monkeypatch.setattr(session_memory, "DB_PATH", tmp_path / "sessions.db")
    monkeypatch.chdir(tmp_path)  # profiles are read from ./data/profiles
    seed_demo_data.seed()

    route = {"value": "plan"}
    import graph.nodes as nodes_pkg

    for mod_name in ("fatigue_analyst", "supervisor", "progression_planner", "plan_writer",
                     "critic", "adaptation_agent", "explainer"):
        mod = sys.modules.get(f"graph.nodes.{mod_name}")
        if mod is None:
            mod = __import__(f"graph.nodes.{mod_name}", fromlist=["x"])
        if hasattr(mod, "get_llm"):
            monkeypatch.setattr(mod, "get_llm",
                                lambda node=None, *a, **k: NodeLLM(node, route["value"]))
    assert nodes_pkg  # imported for its side effect of loading the node modules

    from graph.graph import graph
    from main import _load_profile

    def run(request="", chosen_route="plan"):
        route["value"] = chosen_route
        state = {
            "user_id": DEMO_PERSONA, "user_profile": _load_profile(DEMO_PERSONA),
            "workout_history": [], "current_week_log": [], "fatigue_report": {},
            "retrieved_context": "", "progression_decision": {}, "nutrition_targets": {},
            "weekly_plan": {}, "critic_score": 0.0, "critic_feedback": "", "loop_count": 0,
            "messages": [], "user_request": request,
        }
        result, visited = dict(state), []
        for update in graph.stream(state, stream_mode="updates"):
            for node, patch in update.items():
                visited.append(node)
                result.update(patch or {})
        return result, visited

    return run, plan_store


def test_plan_path_runs_end_to_end_and_saves_the_plan(pipeline):
    run, plan_store = pipeline
    result, visited = run()
    assert visited[:3] == ["context_loader", "fatigue_analyst", "supervisor"]
    assert {"progression_planner", "nutrition_advisor", "plan_writer", "critic"} <= set(visited)
    assert result["route"] == "plan"
    assert result["fatigue_report"]["acwr"] > 1.5  # computed in code from seeded history
    assert result["nutrition_targets"]["training_day"]["calories"] > 0
    assert result["critic_score"] >= 0.75, result["critic_feedback"]
    assert plan_store.load_plan(DEMO_PERSONA)["days"]["monday"]["exercises"]


def test_explain_path_answers_without_changing_the_plan(pipeline):
    run, plan_store = pipeline
    run()  # a plan must exist before the supervisor consults the model
    before = plan_store.load_plan(DEMO_PERSONA)
    result, visited = run("Why am I deloading?", chosen_route="explain")
    assert visited[-1] == "explainer"
    assert "deloading" in result["explanation"]
    assert plan_store.load_plan(DEMO_PERSONA) == before


def test_adapt_path_reaches_the_adaptation_agent(pipeline):
    run, _ = pipeline
    run()
    result, visited = run("My knee hurts", chosen_route="adapt")
    assert visited[-1] == "adaptation_agent"
    assert result["adaptation_result"]["summary"]
