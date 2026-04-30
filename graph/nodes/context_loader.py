from memory.workout_store import get_history, get_current_week
from memory.session_memory import get_messages_as_dicts
from graph.state import AgentState
from tracing.langfuse_config import trace_node


def context_loader(state: AgentState) -> dict:
    """Load workout history, current week log, and session messages from persistent stores."""
    with trace_node("context_loader"):
        user_id = state["user_id"]

        workout_history = get_history(user_id, weeks=4)
        current_week_log = get_current_week(user_id)
        messages = get_messages_as_dicts(user_id)

        return {
            "workout_history": workout_history,
            "current_week_log": current_week_log,
            "messages": messages,
            "loop_count": state.get("loop_count", 0),
            "critic_feedback": state.get("critic_feedback", ""),
        }
