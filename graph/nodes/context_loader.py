from memory.workout_store import get_history, get_current_week
from memory.session_memory import get_messages_as_dicts
from graph.state import AgentState
from tracing.langfuse_config import trace_node


def context_loader(state: AgentState) -> dict:
    """Load workout history, current week log, and session messages from persistent stores."""
    with trace_node("context_loader"):
        user_id = state["user_id"]

        # 5 rolling weeks, not 4: the ACWR chronic window covers the 4 calendar
        # weeks before this one, whose oldest Monday can be up to 4 weeks + 6
        # days ago — a 4-week rolling cutoff truncates it.
        workout_history = get_history(user_id, weeks=5)
        current_week_log = get_current_week(user_id)
        messages = get_messages_as_dicts(user_id)

        return {
            "workout_history": workout_history,
            "current_week_log": current_week_log,
            "messages": messages,
            "loop_count": state.get("loop_count", 0),
            "critic_feedback": state.get("critic_feedback", ""),
        }
