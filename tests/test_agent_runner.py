"""The tool-calling agent loop and the supervisor, driven by a scripted fake
model — no API key, no network. Covers the decision cycle (call tools, feed
results back, stop on a direct answer), its failure handling, and routing."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import tool

from graph.agent_runner import run_tool_agent


class ScriptedLLM:
    """Replays a fixed list of replies; records what it was sent."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []
        self.bound_tools = None

    def bind_tools(self, tools):
        self.bound_tools = [t.name for t in tools]
        return self

    def invoke(self, messages, config=None):
        self.calls.append(list(messages))
        if not self.replies:
            raise AssertionError("model called more times than scripted")
        return self.replies.pop(0)


def answer(text):
    return AIMessage(content=text)


def tool_call(name, args, call_id="c1"):
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": call_id}])


@tool
def lookup(exercise: str) -> str:
    """Return the logged history for one exercise."""
    return f"{exercise}: 3x5 @ 100kg, RPE 8"


@tool
def explode(x: int) -> str:
    """A tool that always fails."""
    raise RuntimeError("boom")


# ── The loop ────────────────────────────────────────────────────────────────

def test_direct_answer_needs_no_tools():
    llm = ScriptedLLM([answer('{"route": "plan"}')])
    assert run_tool_agent(llm, [], "sys", "user") == {"route": "plan"}
    assert len(llm.calls) == 1


def test_tool_result_is_fed_back_before_the_answer():
    llm = ScriptedLLM([
        tool_call("lookup", {"exercise": "squat"}),
        answer('{"load": "hold"}'),
    ])
    result = run_tool_agent(llm, [lookup], "sys", "user")
    assert result == {"load": "hold"}
    assert llm.bound_tools == ["lookup"]
    second_turn = llm.calls[1]
    tool_msgs = [m for m in second_turn if isinstance(m, ToolMessage)]
    assert tool_msgs and "3x5 @ 100kg" in tool_msgs[0].content
    assert tool_msgs[0].tool_call_id == "c1"


def test_unknown_tool_is_reported_not_raised():
    llm = ScriptedLLM([tool_call("nope", {}), answer('{"ok": true}')])
    assert run_tool_agent(llm, [lookup], "sys", "user") == {"ok": True}
    tool_msgs = [m for m in llm.calls[1] if isinstance(m, ToolMessage)]
    assert "Unknown tool" in tool_msgs[0].content


def test_tool_exception_is_returned_to_the_model():
    llm = ScriptedLLM([tool_call("explode", {"x": 1}), answer('{"ok": true}')])
    assert run_tool_agent(llm, [explode], "sys", "user") == {"ok": True}
    tool_msgs = [m for m in llm.calls[1] if isinstance(m, ToolMessage)]
    assert "Tool error" in tool_msgs[0].content


def test_round_budget_forces_a_final_answer():
    llm = ScriptedLLM([
        tool_call("lookup", {"exercise": "squat"}, "a"),
        tool_call("lookup", {"exercise": "bench"}, "b"),
        answer('{"done": true}'),  # the forced, tool-free final answer
    ])
    assert run_tool_agent(llm, [lookup], "sys", "user", max_rounds=2) == {"done": True}
    assert "budget exhausted" in llm.calls[-1][-1].content


def test_json_wrapped_in_a_code_fence_is_accepted():
    llm = ScriptedLLM([answer('```json\n{"route": "adapt"}\n```')])
    assert run_tool_agent(llm, [], "sys", "user") == {"route": "adapt"}


def test_malformed_json_gets_one_retry():
    llm = ScriptedLLM([answer("Sure! Here's my answer: route=adapt"), answer('{"route": "adapt"}')])
    assert run_tool_agent(llm, [], "sys", "user") == {"route": "adapt"}
    assert "single JSON object" in llm.calls[1][-1].content


def test_non_object_json_gets_one_retry():
    llm = ScriptedLLM([answer("0.8"), answer('{"overall": 0.8}')])
    assert run_tool_agent(llm, [], "sys", "user") == {"overall": 0.8}


def test_persistent_garbage_fails_soft_with_empty_dict():
    llm = ScriptedLLM([answer("not json"), answer("still not json")])
    assert run_tool_agent(llm, [], "sys", "user") == {}


def test_malformed_answer_after_budget_also_retries():
    llm = ScriptedLLM([
        tool_call("lookup", {"exercise": "squat"}),
        answer("I think we should hold"),
        answer('{"load": "hold"}'),
    ])
    assert run_tool_agent(llm, [lookup], "sys", "user", max_rounds=1) == {"load": "hold"}


# ── The supervisor ──────────────────────────────────────────────────────────

@pytest.fixture
def supervisor_with(monkeypatch):
    """Run the supervisor with a scripted model and a controllable has_plan."""
    import graph.nodes.supervisor as sup

    def run(replies, plan_exists=True, request="my knee hurts"):
        llm = ScriptedLLM(replies)
        monkeypatch.setattr(sup, "get_llm", lambda *a, **k: llm)
        monkeypatch.setattr(sup, "has_plan", lambda user_id: plan_exists)
        state = {
            "user_id": "u", "user_request": request, "user_profile": {},
            "current_week_log": [], "fatigue_report": {"fatigue_level": "amber"},
        }
        return sup.supervisor(state), llm

    return run


def test_supervisor_without_a_plan_always_plans_and_skips_the_model(supervisor_with):
    out, llm = supervisor_with([], plan_exists=False, request="why am I deloading?")
    assert out["route"] == "plan"
    assert llm.calls == []


def test_supervisor_without_a_request_plans_and_skips_the_model(supervisor_with):
    out, llm = supervisor_with([], plan_exists=True, request="  ")
    assert out["route"] == "plan"
    assert llm.calls == []


@pytest.mark.parametrize("route", ["plan", "adapt", "explain"])
def test_supervisor_follows_a_valid_model_route(supervisor_with, route):
    out, _ = supervisor_with([answer(f'{{"route": "{route}", "reason": "because"}}')])
    assert out == {"route": route, "route_reason": "because"}


def test_supervisor_normalises_case_and_whitespace(supervisor_with):
    out, _ = supervisor_with([answer('{"route": " ADAPT ", "reason": "knee"}')])
    assert out["route"] == "adapt"


def test_supervisor_falls_back_to_explain_on_an_invalid_route(supervisor_with):
    out, _ = supervisor_with([answer('{"route": "replan_everything", "reason": "?"}')])
    assert out["route"] == "explain"  # the only path that changes nothing


def test_supervisor_falls_back_to_explain_on_garbage(supervisor_with):
    out, _ = supervisor_with([answer("adapt please"), answer("still prose")])
    assert out["route"] == "explain"
