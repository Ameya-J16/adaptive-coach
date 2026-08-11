import json

from config.models import get_llm
from graph.state import AgentState
from graph.agent_runner import run_tool_agent
from graph.tools import build_history_tools, build_search_tool
from prompts.progression_planner import PROGRESSION_PLANNER_SYSTEM, PROGRESSION_PLANNER_USER
from tracing.langfuse_config import trace_node


def progression_planner(state: AgentState) -> dict:
    with trace_node("progression_planner"):
        fatigue_report = state.get("fatigue_report", {})
        user_profile = state.get("user_profile", {})
        workout_history = state.get("workout_history", [])
        current_week_log = state.get("current_week_log", [])

        # Everything the agent retrieves lands here, so downstream nodes
        # (plan_writer grounding, critic verification) see the same evidence.
        collected_chunks: list[str] = []
        tools = build_history_tools(workout_history, current_week_log) + [
            build_search_tool(collected_chunks)
        ]

        llm = get_llm("progression_planner", temperature=0.2)

        result = run_tool_agent(
            llm,
            tools,
            system_prompt=PROGRESSION_PLANNER_SYSTEM,
            user_prompt=PROGRESSION_PLANNER_USER.format(
                fatigue_report=json.dumps(fatigue_report),
                user_profile=json.dumps(user_profile),
            ),
            agent_name="progression_planner",
        )

        # Dedup chunks while preserving retrieval order.
        retrieved_context = "\n\n---\n\n".join(dict.fromkeys(collected_chunks))
        if not retrieved_context:
            retrieved_context = "(the planner agent did not retrieve any knowledge-base context)"

        return {
            "progression_decision": result,
            "retrieved_context": retrieved_context,
        }
