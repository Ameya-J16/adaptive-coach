"""Explainer: answers questions about the plan without changing anything.

The read-only path. It exists because "why did you deload me?" previously ran
the entire planning pipeline and produced a brand-new plan — an expensive
non-answer to a question. Routing that to a node with no write tools is both
cheaper and safer.
"""
import json

from config.models import get_llm
from graph.agent_runner import run_tool_agent
from graph.state import AgentState
from graph.tools import build_history_tools, build_search_tool
from memory.plan_store import get_changes, load_plan
from prompts.explainer import EXPLAINER_SYSTEM, EXPLAINER_USER
from tracing.langfuse_config import trace_node


def explainer(state: AgentState) -> dict:
    with trace_node("explainer"):
        user_id = state["user_id"]
        plan = state.get("weekly_plan") or load_plan(user_id) or {}

        lines = []
        for day, session in plan.get("days", {}).items():
            names = ", ".join(e.get("name", "") for e in session.get("exercises", []))
            lines.append(f"{day}: {session.get('session_type', 'rest')} {names}".rstrip())
        plan_text = "\n".join(lines) or "(no plan stored)"

        changes = get_changes(user_id)
        changes_text = "\n".join(f"- {c['change']}" for c in changes[-5:]) or "(none)"

        retrieved: list[str] = []
        tools = build_history_tools(
            state.get("workout_history", []), state.get("current_week_log", [])
        ) + [build_search_tool(retrieved)]

        result = run_tool_agent(
            get_llm("explainer", temperature=0.1),
            tools,
            system_prompt=EXPLAINER_SYSTEM,
            user_prompt=EXPLAINER_USER.format(
                user_profile=json.dumps(state.get("user_profile", {})),
                fatigue_report=json.dumps(state.get("fatigue_report", {})),
                progression_decision=json.dumps(state.get("progression_decision", {})),
                current_plan=plan_text,
                recent_changes=changes_text,
                user_request=state.get("user_request", ""),
            ),
            agent_name="explainer",
        )

        explanation = result.get("explanation", "")
        sources = result.get("sources_used") or []
        if sources:
            explanation += "\n\nBased on: " + "; ".join(str(s) for s in sources)

        return {"explanation": explanation}
