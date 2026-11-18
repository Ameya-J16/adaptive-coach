"""Supervisor: the LLM decides which specialists run.

Before this node existed the graph had one hardcoded path — every request, from
"plan my week" to "why did you deload me?", walked all six nodes. The routing
decision now belongs to a model reading the athlete's words and state, which is
the difference between a workflow (predefined code path) and an agentic system
(the LLM directs its own process).

Deliberately not tool-calling: routing is a single judgement over information
already in hand, and a tool loop here would add latency and failure modes for
nothing. Agency is about who decides, not about how many tools get called.
"""
import json

from config.models import get_llm
from graph.agent_runner import run_tool_agent
from graph.state import AgentState
from memory.plan_store import has_plan
from prompts.supervisor import SUPERVISOR_SYSTEM, SUPERVISOR_USER
from tracing.langfuse_config import trace_node

VALID_ROUTES = {"plan", "adapt", "explain"}


def supervisor(state: AgentState) -> dict:
    with trace_node("supervisor"):
        user_id = state["user_id"]
        request = (state.get("user_request") or "").strip()
        plan_exists = has_plan(user_id)

        # Two deterministic short-circuits. Not a hedge against the model: with
        # no plan there is nothing to adapt or explain, and with no request
        # there is nothing to interpret, so an LLM call would burn latency to
        # reach a forced conclusion.
        if not plan_exists:
            return {
                "route": "plan",
                "route_reason": "No plan exists yet, so I'm building your first week.",
            }
        if not request:
            return {
                "route": "plan",
                "route_reason": "No specific request, so I'm generating a fresh weekly plan.",
            }

        llm = get_llm("supervisor", temperature=0.0)
        result = run_tool_agent(
            llm,
            tools=[],
            system_prompt=SUPERVISOR_SYSTEM,
            user_prompt=SUPERVISOR_USER.format(
                user_profile=json.dumps(state.get("user_profile", {})),
                has_plan="yes",
                recent_sessions=len(state.get("current_week_log", [])),
                fatigue_level=state.get("fatigue_report", {}).get("fatigue_level", "unknown"),
                user_request=request,
            ),
            agent_name="supervisor",
        )

        route = str(result.get("route", "")).strip().lower()
        if route not in VALID_ROUTES:
            # Unparseable route: fall back to the path that changes nothing.
            return {
                "route": "explain",
                "route_reason": "I wasn't sure what you needed, so I'm explaining "
                                "your current plan rather than changing it.",
            }

        return {"route": route, "route_reason": result.get("reason", "")}


def route_from_supervisor(state: AgentState) -> str:
    """Conditional edge: hand control to whichever path the supervisor chose."""
    return state.get("route", "plan")
