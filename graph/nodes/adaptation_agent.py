"""The mid-week adaptation agent.

This is the one part of the system that is agentic in the strict sense: the
input space is open-ended ("my knee hurts", "gym is shut", "I missed Tuesday"),
so there is no predefined code path. The model chooses which tools to call, how
many times, and when the week is coherent again — and its tool calls have
consequences, because they rewrite the stored plan.

Safety is enforced after the fact rather than by constraining the agent: the
revised plan runs through the same deterministic checks as a generated one
(graph/plan_checks.py). If the agent's edits introduce a violation, the plan is
rejected and the original is kept, so an agent with write access can never
persist an unsafe week.
"""
import json
from datetime import datetime

from config.models import get_llm
from graph.adaptation_tools import build_adaptation_tools
from graph.agent_runner import run_tool_agent
from graph.plan_checks import check_plan
from graph.state import AgentState
from graph.tools import build_history_tools, build_search_tool
from memory.plan_store import load_context, load_plan, update_plan
from prompts.adaptation_agent import ADAPTATION_AGENT_SYSTEM, ADAPTATION_AGENT_USER
from tracing.langfuse_config import trace_node


def _plan_summary(plan: dict) -> str:
    """Compact rendering — the full plan JSON is mostly noise for this task."""
    lines = []
    for day, session in plan.get("days", {}).items():
        exercises = session.get("exercises", [])
        if not exercises:
            lines.append(f"{day}: {session.get('session_type', 'rest')}")
            continue
        names = ", ".join(
            f"{e.get('name')} {e.get('sets')}x{e.get('reps')}@{e.get('weight_kg')}kg"
            f" RPE{e.get('rpe_target')}"
            for e in exercises
        )
        lines.append(f"{day}: {session.get('session_type')} — {names}")
    return "\n".join(lines)


def adaptation_agent(state: AgentState) -> dict:
    with trace_node("adaptation_agent"):
        user_id = state["user_id"]
        disruption = state.get("user_request", "")
        plan = state.get("weekly_plan") or load_plan(user_id)

        # The adapt path skips progression_planner, so recover the decision the
        # plan was actually written under rather than judging it against an
        # empty one (which would read every week as "maintain").
        stored_fatigue, stored_decision = load_context(user_id)
        fatigue_report = state.get("fatigue_report") or stored_fatigue
        progression_decision = state.get("progression_decision") or stored_decision

        if not plan:
            return {
                "adaptation_result": {
                    "summary": "There is no active plan to adapt yet — generate one first.",
                    "changes_made": [],
                    "needs_user_input": "",
                    "escalate": True,
                },
            }

        write_tools, tool_state = build_adaptation_tools(plan)
        retrieved: list[str] = []
        tools = (
            write_tools
            + build_history_tools(
                state.get("workout_history", []), state.get("current_week_log", [])
            )
            + [build_search_tool(retrieved)]
        )

        today = datetime.now().date()
        llm = get_llm("adaptation_agent", temperature=0.2)

        result = run_tool_agent(
            llm,
            tools,
            system_prompt=ADAPTATION_AGENT_SYSTEM,
            user_prompt=ADAPTATION_AGENT_USER.format(
                user_profile=json.dumps(state.get("user_profile", {})),
                fatigue_report=json.dumps(fatigue_report),
                progression_decision=json.dumps(progression_decision),
                today=today.isoformat(),
                weekday=today.strftime("%A"),
                current_plan=_plan_summary(plan),
                disruption=disruption,
            ),
            max_rounds=10,  # open-ended task: more room to investigate than a pipeline node
            agent_name="adaptation_agent",
        )

        revised = tool_state["plan"]
        applied = tool_state["changes"]

        # Post-hoc safety gate. The agent writes freely, then the revision is
        # checked — but against the violations the plan ALREADY had, not against
        # an absolute standard. An agent is accountable for what it changed, not
        # for what it inherited; judging the whole plan would reject a correct
        # injury substitution because the week it was editing already ran hot.
        rejected = []
        if applied:
            history = state.get("workout_history", [])
            before = set(
                check_plan(plan, fatigue_report, progression_decision, history)["violations"]
            )
            after = check_plan(
                revised, fatigue_report, progression_decision, history
            )["violations"]
            introduced = [v for v in after if v not in before]

            if introduced:
                rejected = introduced
            else:
                update_plan(user_id, revised, "; ".join(applied))

        result["changes_made"] = applied
        result["rejected_violations"] = rejected
        if rejected:
            result["summary"] = (
                "I tried to adjust your week but the revision would have introduced a "
                "new problem: " + "; ".join(rejected)
                + " Your original plan is unchanged — ask me again and I'll try a "
                "different approach."
            )

        return {
            "adaptation_result": result,
            "weekly_plan": plan if rejected else revised,
        }
