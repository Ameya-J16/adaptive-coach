"""Model selection: which settings reach the chat client for each model family."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest

from config.models import get_llm, is_reasoning_model, model_for


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for var in ("LLM_PROVIDER", "LLM_MODEL", "LLM_MODEL_CRITIC", "LLM_REASONING_EFFORT"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-not-a-real-key")


@pytest.mark.parametrize("model,expected", [
    ("gpt-5.4", True), ("gpt-5-mini", True), ("o3", True), ("o4-mini", True),
    ("openai/gpt-5.4", True),
    ("gpt-4o", False), ("gpt-4.1", False), ("openai/gpt-oss-120b", False),
])
def test_reasoning_model_detection(model, expected):
    assert is_reasoning_model(model) is expected


def test_default_model_keeps_temperature():
    llm = get_llm("critic", temperature=0.0)
    assert llm.model_name == "gpt-4o"
    assert llm.temperature == 0.0


def test_reasoning_model_omits_temperature(monkeypatch):
    monkeypatch.setenv("LLM_MODEL", "gpt-5.4")
    llm = get_llm("critic", temperature=0.0)
    assert llm.model_name == "gpt-5.4"
    assert llm.temperature is None
    assert llm.reasoning_effort is None


def _payload(llm, with_tools=True):
    from langchain_core.messages import HumanMessage
    from langchain_core.tools import tool

    @tool
    def lookup(exercise: str) -> str:
        """Look up an exercise."""
        return ""

    tools = llm.bind_tools([lookup]).kwargs["tools"] if with_tools else None
    kwargs = {"tools": tools} if tools else {}
    return llm._get_request_payload([HumanMessage("hi")], **kwargs)


def test_reasoning_effort_uses_the_responses_api_with_tools(monkeypatch):
    # Chat Completions rejects tools + reasoning_effort on newer models
    # (HTTP 400 seen live on gpt-5.4-mini); the Responses API accepts both.
    monkeypatch.setenv("LLM_MODEL", "gpt-5.4-mini")
    monkeypatch.setenv("LLM_REASONING_EFFORT", "low")
    payload = _payload(get_llm("critic"))
    assert "input" in payload, "should target the Responses API"
    assert payload["reasoning"] == {"effort": "low"}
    assert "reasoning_effort" not in payload
    assert payload["tools"]


def test_reasoning_effort_none_stays_on_chat_completions(monkeypatch):
    monkeypatch.setenv("LLM_MODEL", "gpt-5.4-mini")
    monkeypatch.setenv("LLM_REASONING_EFFORT", "none")
    payload = _payload(get_llm("critic"))
    assert "messages" in payload and payload["reasoning_effort"] == "none"


def test_no_effort_sends_no_reasoning_settings(monkeypatch):
    monkeypatch.setenv("LLM_MODEL", "gpt-5.4-mini")
    payload = _payload(get_llm("critic"))
    assert "messages" in payload
    assert "reasoning" not in payload and "reasoning_effort" not in payload


def test_per_node_override_beats_global(monkeypatch):
    monkeypatch.setenv("LLM_MODEL", "gpt-5.4")
    monkeypatch.setenv("LLM_MODEL_CRITIC", "gpt-4o")
    assert model_for("critic") == "gpt-4o"
    assert model_for("plan_writer") == "gpt-5.4"
    assert get_llm("critic", temperature=0.0).temperature == 0.0
