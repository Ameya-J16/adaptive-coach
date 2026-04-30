import json
import os
from datetime import datetime, timedelta
from pathlib import Path


DATA_DIR = Path(__file__).parent.parent / "data" / "workouts"


def _user_file(user_id: str) -> Path:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    return DATA_DIR / f"{user_id}.json"


def _load(user_id: str) -> list[dict]:
    path = _user_file(user_id)
    if not path.exists():
        return []
    with open(path) as f:
        return json.load(f)


def _save(user_id: str, sessions: list[dict]) -> None:
    with open(_user_file(user_id), "w") as f:
        json.dump(sessions, f, indent=2, default=str)


def _compute_volume(exercises: list[dict]) -> float:
    """Compute total volume handling both per-set (sets_data) and averaged formats."""
    total = 0.0
    for ex in exercises:
        if "sets_data" in ex and ex["sets_data"]:
            total += sum(s.get("reps", 0) * s.get("weight_kg", 0) for s in ex["sets_data"])
        else:
            total += ex.get("sets", 0) * ex.get("reps", 0) * ex.get("weight_kg", 0)
    return round(total, 2)


def log_workout(user_id: str, session: dict) -> None:
    """Append a workout session for the given user."""
    sessions = _load(user_id)
    if "date" not in session:
        session["date"] = datetime.now().date().isoformat()
    session["total_volume"] = _compute_volume(session.get("exercises", []))
    sessions.append(session)
    _save(user_id, sessions)


def get_history(user_id: str, weeks: int = 4) -> list[dict]:
    """Return workout sessions from the last N weeks."""
    sessions = _load(user_id)
    cutoff = (datetime.now() - timedelta(weeks=weeks)).date().isoformat()
    return [s for s in sessions if s.get("date", "") >= cutoff]


def get_current_week(user_id: str) -> list[dict]:
    """Return workout sessions from the current week (Mon–today)."""
    sessions = _load(user_id)
    today = datetime.now().date()
    week_start = (today - timedelta(days=today.weekday())).isoformat()
    return [s for s in sessions if s.get("date", "") >= week_start]


def get_all_sessions(user_id: str) -> list[dict]:
    return _load(user_id)
