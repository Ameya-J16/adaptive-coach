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


def test_reasoning_effort_is_passed_through(monkeypatch):
    monkeypatch.setenv("LLM_MODEL", "gpt-5.4")
    monkeypatch.setenv("LLM_REASONING_EFFORT", "low")
    assert get_llm("critic").reasoning_effort == "low"


def test_per_node_override_beats_global(monkeypatch):
    monkeypatch.setenv("LLM_MODEL", "gpt-5.4")
    monkeypatch.setenv("LLM_MODEL_CRITIC", "gpt-4o")
    assert model_for("critic") == "gpt-4o"
    assert model_for("plan_writer") == "gpt-5.4"
    assert get_llm("critic", temperature=0.0).temperature == 0.0
