"""The routing golden set is well-formed, and the cases decided in code
(no plan -> plan) route correctly without any model call."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from evals.routing_set import CASES
from evals.run_routing_eval import route_case, validate


def test_routing_set_is_valid():
    assert validate(CASES) == []


def test_short_circuit_cases_route_without_a_model(monkeypatch):
    import graph.nodes.supervisor as sup

    def no_model(*a, **k):
        raise AssertionError("short-circuit cases must not call the model")

    monkeypatch.setattr(sup, "get_llm", no_model)
    for case in [c for c in CASES if not c.get("plan_exists", True)]:
        assert route_case(case)["route"] == case["expected"]
