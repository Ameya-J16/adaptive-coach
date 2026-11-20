"""Write-capable tools for the adaptation agent.

This is the line between the planning pipeline and a real agent. Every tool in
graph/tools.py answers a question; the tools here **change the plan**. The agent
decides which to call, with what arguments, and when it is finished — nothing in
Python decides that for it.

The tools mutate a working copy held by the factory closure. The caller reads
`state["plan"]` afterwards and persists it, so a crashed or rejected run cannot
leave a half-edited plan on disk.
"""
import copy

from langchain_core.tools import tool

DAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]


def build_adaptation_tools(plan: dict) -> tuple[list, dict]:
    """Return (tools, state). `state["plan"]` is the working copy the tools edit
    and `state["changes"]` is the running list of what the agent did."""
    state = {"plan": copy.deepcopy(plan), "changes": []}

    def _days() -> dict:
        return state["plan"].setdefault("days", {})

    def _valid_day(day: str) -> str | None:
        d = (day or "").strip().lower()
        return d if d in DAYS else None

    @tool
    def view_day(day: str) -> str:
        """Show the current contents of one day: session type, every exercise
        with sets/reps/weight/RPE, and the day's notes."""
        d = _valid_day(day)
        if not d:
            return f"'{day}' is not a day of the week."
        session = _days().get(d)
        if not session:
            return f"{d}: nothing scheduled."
        lines = [f"{d}: {session.get('session_type', 'rest')}"]
        for ex in session.get("exercises", []):
            lines.append(
                f"  - {ex.get('name')}: {ex.get('sets')}x{ex.get('reps')} @ "
                f"{ex.get('weight_kg')}kg RPE {ex.get('rpe_target')} "
                f"[{', '.join(ex.get('muscle_groups', []))}]"
            )
        if session.get("notes"):
            lines.append(f"  notes: {session['notes']}")
        return "\n".join(lines)

    @tool
    def set_day_to_rest(day: str, reason: str) -> str:
        """Convert a day to a rest day. Use when the athlete cannot train that
        day at all, or when recovery is the right call."""
        d = _valid_day(day)
        if not d:
            return f"'{day}' is not a day of the week."
        _days()[d] = {"session_type": "rest", "exercises": [], "notes": reason}
        change = f"{d} -> rest ({reason})"
        state["changes"].append(change)
        return f"Done: {change}"

    @tool
    def move_session(from_day: str, to_day: str, reason: str) -> str:
        """Move a whole session to another day. The origin day becomes rest.
        Use for scheduling conflicts like travel or a closed gym."""
        src, dst = _valid_day(from_day), _valid_day(to_day)
        if not src or not dst:
            return "Both from_day and to_day must be days of the week."
        days = _days()
        session = days.get(src)
        if not session or not session.get("exercises"):
            return f"{src} has no session to move."
        if days.get(dst, {}).get("exercises"):
            return (
                f"{dst} already has a session — moving here would create a "
                "double day. Pick an empty day or merge deliberately."
            )
        days[dst] = session
        days[src] = {"session_type": "rest", "exercises": [], "notes": reason}
        change = f"moved {src} session to {dst} ({reason})"
        state["changes"].append(change)
        return f"Done: {change}"

    @tool
    def swap_exercise(
        day: str,
        old_exercise: str,
        new_exercise: str,
        sets: int,
        reps: int,
        weight_kg: float,
        rpe_target: float,
        muscle_groups: str,
        reason: str,
    ) -> str:
        """Replace one exercise with another on a given day. Use for injuries,
        missing equipment, or a lift that is stalling. muscle_groups is a
        comma-separated list, e.g. 'quads,glutes'."""
        d = _valid_day(day)
        if not d:
            return f"'{day}' is not a day of the week."
        session = _days().get(d, {})
        needle = old_exercise.lower().strip()
        for ex in session.get("exercises", []):
            if needle in ex.get("name", "").lower():
                previous = ex.get("name")
                ex.update({
                    "name": new_exercise,
                    "sets": sets,
                    "reps": reps,
                    "weight_kg": weight_kg,
                    "rpe_target": rpe_target,
                    "muscle_groups": [m.strip() for m in muscle_groups.split(",") if m.strip()],
                })
                change = f"{d}: {previous} -> {new_exercise} ({reason})"
                state["changes"].append(change)
                return f"Done: {change}"
        return f"No exercise matching '{old_exercise}' on {d}."

    @tool
    def adjust_exercise_load(
        day: str, exercise: str, weight_kg: float, rpe_target: float, reason: str
    ) -> str:
        """Change the weight and RPE target of one exercise without replacing
        it. Use to reduce intensity after a poor night's sleep or a niggle."""
        d = _valid_day(day)
        if not d:
            return f"'{day}' is not a day of the week."
        needle = exercise.lower().strip()
        for ex in _days().get(d, {}).get("exercises", []):
            if needle in ex.get("name", "").lower():
                before = f"{ex.get('weight_kg')}kg RPE {ex.get('rpe_target')}"
                ex["weight_kg"] = weight_kg
                ex["rpe_target"] = rpe_target
                change = (
                    f"{d}: {ex.get('name')} {before} -> {weight_kg}kg "
                    f"RPE {rpe_target} ({reason})"
                )
                state["changes"].append(change)
                return f"Done: {change}"
        return f"No exercise matching '{exercise}' on {d}."

    @tool
    def add_note_to_day(day: str, note: str) -> str:
        """Append coaching guidance to a day's notes without changing the work."""
        d = _valid_day(day)
        if not d:
            return f"'{day}' is not a day of the week."
        session = _days().setdefault(
            d, {"session_type": "rest", "exercises": [], "notes": ""}
        )
        session["notes"] = (session.get("notes", "") + " " + note).strip()
        state["changes"].append(f"{d}: note added")
        return f"Done: note added to {d}."

    tools = [
        view_day,
        set_day_to_rest,
        move_session,
        swap_exercise,
        adjust_exercise_load,
        add_note_to_day,
    ]
    return tools, state
