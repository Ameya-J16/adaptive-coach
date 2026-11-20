import json

from config.models import get_llm
from graph.state import AgentState
from graph.agent_runner import run_tool_agent
from graph.plan_checks import apply_caps, check_plan
from graph.tools import build_search_tool
from memory.plan_store import save_plan
from prompts.critic import CRITIC_SYSTEM, CRITIC_USER
from tracing.langfuse_config import trace_node


def evaluate_plan(
    weekly_plan: dict,
    fatigue_report: dict,
    progression_decision: dict,
    user_profile: dict,
    retrieved_context: str,
    workout_history: list[dict],
) -> dict:
    """The critic's core, callable outside the graph (used by the eval harness).

    Mechanical rubric items are enforced in Python; the LLM judge adds
    qualitative deductions on top and can never lift a capped score. Any
    mechanical violation forces the overall below the replan threshold.
    """
    checks = check_plan(weekly_plan, fatigue_report, progression_decision, workout_history)
    violations_text = "\n".join(checks["violations"]) or "(none)"

    llm = get_llm("critic", temperature=0.0)
    verification_chunks: list[str] = []
    tools = [build_search_tool(verification_chunks)]

    result = run_tool_agent(
        llm,
        tools,
        system_prompt=CRITIC_SYSTEM,
        user_prompt=CRITIC_USER.format(
            user_profile=json.dumps(user_profile),
            fatigue_report=json.dumps(fatigue_report),
            progression_decision=json.dumps(progression_decision),
            rule_violations=violations_text,
            retrieved_context=retrieved_context,
            weekly_plan=json.dumps(weekly_plan),
        ),
        agent_name="critic",
    )

    scores, overall = apply_caps(result, checks)

    feedback_parts = []
    if checks["violations"]:
        feedback_parts.append(
            "Rule violations (must fix):\n- " + "\n- ".join(checks["violations"])
        )
    if result.get("feedback"):
        feedback_parts.append(result["feedback"])

    return {
        "scores": scores,
        "overall": overall,
        "feedback": "\n\n".join(feedback_parts),
        "violations": checks["violations"],
        "issues_found": result.get("issues_found", []),
    }


def critic(state: AgentState) -> dict:
    with trace_node("critic"):
        weekly_plan = state.get("weekly_plan", {})
        evaluation = evaluate_plan(
            weekly_plan=weekly_plan,
            fatigue_report=state.get("fatigue_report", {}),
            progression_decision=state.get("progression_decision", {}),
            user_profile=state.get("user_profile", {}),
            retrieved_context=state.get("retrieved_context", ""),
            workout_history=state.get("workout_history", []),
        )
        loop_count = state.get("loop_count", 0) + 1

        # Persist once the loop is finished — accepted, or out of revisions.
        # Mirrors should_replan; a plan on disk is what the adaptation agent
        # and explainer later act on.
        finished = not (evaluation["overall"] < 0.75 and loop_count < 3)
        if finished and weekly_plan:
            save_plan(
                state["user_id"],
                weekly_plan,
                source="planner",
                fatigue_report=state.get("fatigue_report", {}),
                progression_decision=state.get("progression_decision", {}),
            )

        return {
            "critic_score": evaluation["overall"],
            "critic_feedback": evaluation["feedback"],
            "loop_count": loop_count,
        }
