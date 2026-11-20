import json

from config.models import get_llm
from graph.state import AgentState
from graph.agent_runner import run_tool_agent
from graph.tools import build_history_tools
from prompts.plan_writer import PLAN_WRITER_SYSTEM, PLAN_WRITER_USER
from tracing.langfuse_config import trace_node


def plan_writer(state: AgentState) -> dict:
    with trace_node("plan_writer"):
        llm = get_llm("plan_writer", temperature=0.3)
        tools = build_history_tools(
            state.get("workout_history", []),
            state.get("current_week_log", []),
        )

        result = run_tool_agent(
            llm,
            tools,
            system_prompt=PLAN_WRITER_SYSTEM,
            user_prompt=PLAN_WRITER_USER.format(
                user_profile=json.dumps(state.get("user_profile", {})),
                fatigue_report=json.dumps(state.get("fatigue_report", {})),
                progression_decision=json.dumps(state.get("progression_decision", {})),
                nutrition_targets=json.dumps(state.get("nutrition_targets", {})),
                retrieved_context=state.get("retrieved_context", ""),
                critic_feedback=state.get("critic_feedback", "") or "(none)",
            ),
            agent_name="plan_writer",
        )

        return {
            "weekly_plan": result,
            "loop_count": state.get("loop_count", 0),
        }
