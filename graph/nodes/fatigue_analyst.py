import json

from config.models import get_llm
from graph.state import AgentState
from graph.agent_runner import run_tool_agent
from graph.tools import build_history_tools
from prompts.fatigue_analyst import FATIGUE_ANALYST_SYSTEM, FATIGUE_ANALYST_USER
from tracing.langfuse_config import trace_node


# Rolling window lengths, in days. 7/28 is the standard formulation in the
# training-load literature.
ACUTE_DAYS = 7
CHRONIC_DAYS = 28


def _exercise_volume(ex: dict) -> float:
    """Compute volume for one exercise, handling both per-set and averaged formats."""
    if "sets_data" in ex and ex["sets_data"]:
        return sum(s.get("reps", 0) * s.get("weight_kg", 0) for s in ex["sets_data"])
    return ex.get("sets", 0) * ex.get("reps", 0) * ex.get("weight_kg", 0)


def _compute_acwr(workout_history: list[dict], current_week_log: list[dict]) -> float | None:
    """Compute ACWR with rolling windows: last 7 days over the average week of
    the 28 days before that.

    Rolling rather than calendar-based on purpose. A calendar acute window
    (Monday..today) compares a *partial* week against *complete* weeks, so the
    same training week reads ACWR 0.51 on Monday and 1.22 on Friday — which
    would tell an athlete checking in early in the week that they are
    under-trained. The two windows are adjacent and non-overlapping, so no
    session is ever counted as both acute and chronic load.

    Returns None when there is no chronic baseline to compare against —
    callers must treat that as "unknown", not as a neutral reading.
    """
    from datetime import datetime, timedelta

    def volume(sessions: list[dict]) -> float:
        total = 0.0
        for s in sessions:
            for ex in s.get("exercises", []):
                total += _exercise_volume(ex)
        return total

    # history and current-week are read from the same store, so the same
    # session can appear in both lists — dedupe by value before slicing.
    seen: dict[str, dict] = {}
    for s in list(workout_history) + list(current_week_log or []):
        seen.setdefault(json.dumps(s, sort_keys=True, default=str), s)
    sessions = list(seen.values())

    today = datetime.now().date()
    acute_start = today - timedelta(days=ACUTE_DAYS - 1)
    chronic_end = acute_start - timedelta(days=1)
    chronic_start = chronic_end - timedelta(days=CHRONIC_DAYS - 1)

    def in_window(s: dict, start, end) -> bool:
        d = s.get("date", "")
        return bool(d) and start.isoformat() <= d <= end.isoformat()

    acute = volume([s for s in sessions if in_window(s, acute_start, today)])
    chronic_total = volume(
        [s for s in sessions if in_window(s, chronic_start, chronic_end)]
    )

    if chronic_total == 0:
        return None

    chronic_weekly = chronic_total / (CHRONIC_DAYS / ACUTE_DAYS)
    return round(acute / chronic_weekly, 3)


def _classify_fatigue(acwr: float | None) -> tuple[str, str]:
    """Map ACWR to (fatigue_level, acwr_status).

    No baseline means we cannot rule out overreach, so default to amber
    (caution / no load increases) rather than green.
    """
    if acwr is None:
        return "amber", "no_baseline"
    if acwr > 1.5:
        return "red", "computed"
    if acwr > 1.3:
        return "amber", "computed"
    return "green", "computed"


def fatigue_analyst(state: AgentState) -> dict:
    with trace_node("fatigue_analyst"):
        workout_history = state.get("workout_history", [])
        current_week_log = state.get("current_week_log", [])
        user_profile = state.get("user_profile", {})

        acwr = _compute_acwr(workout_history, current_week_log)
        level, status = _classify_fatigue(acwr)

        llm = get_llm("fatigue_analyst", temperature=0.1)
        tools = build_history_tools(workout_history, current_week_log)

        result = run_tool_agent(
            llm,
            tools,
            system_prompt=FATIGUE_ANALYST_SYSTEM,
            user_prompt=FATIGUE_ANALYST_USER.format(
                user_profile=json.dumps(user_profile),
                acwr=acwr,
                acwr_status=status,
                floor_level=level,
                current_week_count=len(current_week_log),
            ),
            agent_name="fatigue_analyst",
        )

        # The computed ACWR is authoritative; the agent only does qualitative analysis.
        result["acwr"] = acwr
        result["acwr_status"] = status

        # Enforce the deterministic floor: red/amber/no-baseline always win,
        # but in the green zone the agent may still raise the level from RPE trends.
        if status == "no_baseline" or level in ("red", "amber"):
            result["fatigue_level"] = level
        else:
            result["fatigue_level"] = result.get("fatigue_level", "green")

        return {"fatigue_report": result}
