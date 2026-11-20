"""Tools the agents can call to query workout data and the knowledge base.

Built as factories returning closures over one run's data, so a tool call
always sees the requesting user's history and nothing else. This replaces
the old approach of serialising the full history into every prompt.
"""
from datetime import datetime, timedelta

from langchain_core.tools import tool


def _exercise_volume(ex: dict) -> float:
    if "sets_data" in ex and ex["sets_data"]:
        return sum(s.get("reps", 0) * s.get("weight_kg", 0) for s in ex["sets_data"])
    return ex.get("sets", 0) * ex.get("reps", 0) * ex.get("weight_kg", 0)


def _session_volume(session: dict) -> float:
    return sum(_exercise_volume(ex) for ex in session.get("exercises", []))


def _session_line(s: dict) -> str:
    names = ", ".join(ex.get("name", "?") for ex in s.get("exercises", []))
    return (
        f"{s.get('date', '?')} | {s.get('session_type', 'strength')} | "
        f"volume {_session_volume(s):.0f}kg | avg RPE {s.get('avg_rpe', '?')} | {names or 'no exercises'}"
    )


def build_history_tools(workout_history: list[dict], current_week_log: list[dict]) -> list:
    """Tools for querying this user's training history."""
    # history and current-week are loaded separately from the same store, so
    # the same session shows up in both lists as distinct objects — dedup by value.
    import json as _json
    seen: dict[str, dict] = {}
    for s in workout_history + current_week_log:
        seen.setdefault(_json.dumps(s, sort_keys=True, default=str), s)
    all_sessions = sorted(seen.values(), key=lambda s: s.get("date", ""))

    @tool
    def get_weekly_volume_summary() -> str:
        """Per-calendar-week training summary: total volume (kg), session count and
        average RPE for the current week and the 4 weeks before it. Use this first
        to understand the overall load trend."""
        today = datetime.now().date()
        this_monday = today - timedelta(days=today.weekday())
        lines = []
        for offset in range(4, -1, -1):
            start = this_monday - timedelta(weeks=offset)
            end = start + timedelta(weeks=1)
            sessions = [
                s for s in all_sessions
                if start.isoformat() <= s.get("date", "") < end.isoformat()
            ]
            vol = sum(_session_volume(s) for s in sessions)
            rpes = [s["avg_rpe"] for s in sessions if s.get("avg_rpe") is not None]
            avg_rpe = f"{sum(rpes) / len(rpes):.1f}" if rpes else "n/a"
            label = "current week (partial)" if offset == 0 else f"{offset} week(s) ago"
            lines.append(
                f"week of {start.isoformat()} ({label}): "
                f"{len(sessions)} sessions, volume {vol:.0f}kg, avg RPE {avg_rpe}"
            )
        return "\n".join(lines) if lines else "No sessions logged."

    @tool
    def get_recent_sessions(n: int = 6) -> str:
        """The n most recent logged sessions, one line each: date, session type,
        total volume, average RPE, and exercises performed."""
        recent = all_sessions[-n:]
        if not recent:
            return "No sessions logged."
        return "\n".join(_session_line(s) for s in recent)

    @tool
    def get_exercise_history(exercise_name: str) -> str:
        """Every logged occurrence of one exercise (case-insensitive substring
        match), with date, sets x reps x weight and RPE. Use this to spot
        stagnation or degradation on a specific lift."""
        needle = exercise_name.lower().strip()
        lines = []
        for s in all_sessions:
            for ex in s.get("exercises", []):
                if needle in ex.get("name", "").lower():
                    if ex.get("sets_data"):
                        detail = "; ".join(
                            f"{st.get('reps')}x{st.get('weight_kg')}kg" for st in ex["sets_data"]
                        )
                    else:
                        detail = f"{ex.get('sets')}x{ex.get('reps')} @ {ex.get('weight_kg')}kg"
                    rpe = ex.get("rpe_target") or s.get("avg_rpe", "?")
                    lines.append(f"{s.get('date', '?')}: {ex.get('name')} {detail} (RPE {rpe})")
        return "\n".join(lines) or f"No logged sessions contain '{exercise_name}'."

    @tool
    def get_muscle_group_volume() -> str:
        """Hard sets, stimulating reps and near-failure sets per muscle group over
        the last 7 days, with each compared to MEV/MAV landmarks. Counts effort
        rather than kilograms, so bodyweight work is included. Use this to judge
        whether a muscle group is under- or over-worked."""
        from graph.training_metrics import weekly_muscle_summary
        summary = weekly_muscle_summary(all_sessions)
        if not summary:
            return "No sets logged with RPE in the last 7 days."
        lines = []
        for muscle, s in sorted(summary.items()):
            lines.append(
                f"{muscle}: {s['hard_sets']} hard sets, {s['stimulating_reps']} "
                f"stimulating reps, {s['failure_sets']} near-failure sets — {s['status']}"
            )
        return "\n".join(lines)

    @tool
    def get_effort_trend() -> str:
        """Where the athlete's effort is drifting at IDENTICAL weight and reps,
        plus how much recent work was taken near failure. Rising RPE on unchanged
        work is accumulated fatigue that volume figures cannot show."""
        from graph.training_metrics import failure_exposure, rpe_drift
        drift = rpe_drift(all_sessions)
        exposure = failure_exposure(all_sessions)
        lines = [
            f"Near-failure sets in the last 7 days: {exposure['near_failure_sets']}"
            f"/{exposure['total_sets']} ({exposure['share']:.0%}, {exposure['flag']})"
        ]
        if not drift:
            lines.append("No matched-load comparisons available yet.")
        for d in drift[:6]:
            lines.append(
                f"{d['exercise']} {d['weight_kg']}kg x{d['reps']}: RPE "
                f"{d['rpe_change']:+} ({d['direction']}) between {d['from_date']} and {d['to_date']}"
            )
        return "\n".join(lines)

    return [
        get_weekly_volume_summary,
        get_recent_sessions,
        get_exercise_history,
        get_muscle_group_volume,
        get_effort_trend,
    ]


def build_search_tool(collected_chunks: list[str]):
    """Knowledge-base search tool. Every retrieved chunk is also appended to
    collected_chunks so the caller can persist what the agent looked at —
    that record is what the critic later verifies groundedness against."""
    from rag.retriever import retrieve

    @tool
    def search_knowledge_base(query: str) -> str:
        """Search the sports-science knowledge base (training load management,
        periodisation, recovery, nutrition). Phrase the query around the concept
        you need evidence for, e.g. 'deload protocol after overreaching'."""
        result = retrieve(query, k=4)
        collected_chunks.append(result)
        return result

    return search_knowledge_base
