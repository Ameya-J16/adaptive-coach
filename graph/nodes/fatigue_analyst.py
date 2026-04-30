import json
import re

from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import JsonOutputParser

from graph.state import AgentState
from prompts.fatigue_analyst import FATIGUE_ANALYST_PROMPT
from tracing.langfuse_config import trace_node, get_callbacks


def _exercise_volume(ex: dict) -> float:
    """Compute volume for one exercise, handling both per-set and averaged formats."""
    if "sets_data" in ex and ex["sets_data"]:
        return sum(s.get("reps", 0) * s.get("weight_kg", 0) for s in ex["sets_data"])
    return ex.get("sets", 0) * ex.get("reps", 0) * ex.get("weight_kg", 0)


def _compute_acwr(workout_history: list[dict], current_week_log: list[dict]) -> float:
    """Compute ACWR: acute (current week) / chronic (4-week rolling average)."""
    def week_volume(sessions: list[dict]) -> float:
        total = 0.0
        for s in sessions:
            for ex in s.get("exercises", []):
                total += _exercise_volume(ex)
        return total

    acute = week_volume(current_week_log)

    if not workout_history:
        return 1.0  # no history — assume neutral

    # Partition history into up to 4 weekly buckets
    from datetime import datetime, timedelta
    today = datetime.now().date()
    week_volumes = []
    for week_offset in range(4):
        week_start = (today - timedelta(weeks=week_offset + 1)).isoformat()
        week_end = (today - timedelta(weeks=week_offset)).isoformat()
        sessions = [
            s for s in workout_history
            if week_start <= s.get("date", "") < week_end
        ]
        week_volumes.append(week_volume(sessions))

    chronic = sum(week_volumes) / len(week_volumes) if week_volumes else 1.0
    if chronic == 0:
        return 1.0
    return round(acute / chronic, 3)


def fatigue_analyst(state: AgentState) -> dict:
    with trace_node("fatigue_analyst"):
        workout_history = state.get("workout_history", [])
        current_week_log = state.get("current_week_log", [])
        user_profile = state.get("user_profile", {})

        acwr = _compute_acwr(workout_history, current_week_log)

        llm = ChatOpenAI(model="gpt-4o", temperature=0.1)
        chain = FATIGUE_ANALYST_PROMPT | llm | JsonOutputParser()

        result = chain.invoke(
            {
                "user_profile": json.dumps(user_profile, indent=2),
                "workout_history": json.dumps(workout_history, indent=2),
                "current_week_log": json.dumps(current_week_log, indent=2),
            },
            config={"callbacks": get_callbacks()},
        )

        # Override the ACWR with our computed value for accuracy
        result["acwr"] = acwr

        # Enforce ACWR-based fatigue level rules
        if acwr > 1.5:
            result["fatigue_level"] = "red"
        elif acwr > 1.3:
            result["fatigue_level"] = "amber"
        else:
            result["fatigue_level"] = result.get("fatigue_level", "green")

        return {"fatigue_report": result}
