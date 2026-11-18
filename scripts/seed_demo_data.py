"""Seed demo personas with realistic multi-day training weeks.

Run with: python -m scripts.seed_demo_data

Each persona targets an ACWR zone under the corrected computation
(chronic = the 4 calendar weeks before this one, acute = Monday..today).
Chronic weeks get three sessions (Mon/Wed/Fri); the current week spreads
its load across Monday..today. After writing, the script recomputes every
persona's ACWR through the real store + fatigue code and prints the zone,
so a bad seed is loud, not silent.
"""
import json
import sys
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from graph.nodes.fatigue_analyst import _classify_fatigue, _compute_acwr
from memory.workout_store import get_current_week, get_history

DATA_DIR = Path(__file__).parent.parent / "data" / "workouts"

TODAY = date.today()
THIS_MONDAY = TODAY - timedelta(days=TODAY.weekday())

# name, sets, reps, relative weight share, muscle groups
STRENGTH_TEMPLATE = [
    ("Squat", 4, 5, 1.00, ["quads", "glutes"]),
    ("Bench Press", 4, 6, 0.75, ["chest", "triceps"]),
    ("Romanian Deadlift", 3, 8, 0.80, ["hamstrings", "glutes"]),
    ("Barbell Row", 3, 10, 0.60, ["back", "biceps"]),
]
DUMBBELL_TEMPLATE = [
    ("Goblet Squat", 4, 10, 0.60, ["quads", "glutes"]),
    ("Dumbbell Bench Press", 4, 10, 0.55, ["chest", "triceps"]),
    ("Lat Pulldown", 3, 12, 0.65, ["back", "biceps"]),
    ("Cable Row", 3, 12, 0.60, ["back", "biceps"]),
]
LIGHT_TEMPLATE = [
    ("Goblet Squat", 3, 12, 0.50, ["quads", "glutes"]),
    ("Walking Lunge", 3, 12, 0.35, ["quads", "glutes"]),
    ("Dumbbell Press", 3, 10, 0.40, ["shoulders", "triceps"]),
]


def _strength_session(day: date, template: list, target_volume: float, rpe: float) -> dict:
    unit = sum(sets * reps * share for _, sets, reps, share, _ in template)
    scale = target_volume / unit
    exercises = []
    for name, sets, reps, share, muscles in template:
        weight = round(share * scale * 2) / 2  # nearest 0.5 kg
        exercises.append({
            "name": name,
            "sets_data": [
                {"set_number": i + 1, "reps": reps, "weight_kg": weight, "rpe": rpe}
                for i in range(sets)
            ],
            "muscle_groups": muscles,
            "notes": "",
        })
    total = sum(
        s["reps"] * s["weight_kg"] for ex in exercises for s in ex["sets_data"]
    )
    return {
        "date": day.isoformat(),
        "exercises": exercises,
        "total_volume": round(total, 2),
        "avg_rpe": rpe,
        "session_type": "strength",
        "notes": "",
    }


def _cardio_session(day: date, kind: str, rpe: float, note: str) -> dict:
    return {
        "date": day.isoformat(),
        "exercises": [],
        "total_volume": 0.0,
        "avg_rpe": rpe,
        "session_type": kind,
        "notes": note,
    }


def _build_persona(
    chronic_weekly_volume: float,
    target_acwr: float,
    template: list,
    chronic_rpes: list[float],
    acute_rpe: float,
    cardio: bool = False,
) -> list[dict]:
    """Training laid out against the rolling ACWR windows.

    Chronic window is days 7..34 back: four weekly blocks of 3 sessions.
    Acute window is the last 7 days: 3 sessions carrying target_acwr x the
    chronic weekly average. Dates are relative to today, not to Monday, so
    re-seeding on any weekday produces the same ACWR.
    """
    sessions = []
    # Chronic blocks centred at 10, 17, 24 and 31 days back — all comfortably
    # inside the 7..34 day chronic window regardless of weekday.
    for block_centre, rpe in zip((31, 24, 17, 10), chronic_rpes):
        per_session = chronic_weekly_volume / 3
        for offset in (0, 2, 4):
            day = TODAY - timedelta(days=block_centre + 2 - offset)
            sessions.append(_strength_session(day, template, per_session, rpe))
            if cardio and offset == 2:
                sessions.append(
                    _cardio_session(day + timedelta(days=1), "cardio", rpe - 0.5,
                                    "60min zone-2 run")
                )

    acute_total = target_acwr * chronic_weekly_volume
    acute_days = [TODAY - timedelta(days=d) for d in (0, 2, 4)]
    per_session = acute_total / len(acute_days)
    for day in acute_days:
        sessions.append(_strength_session(day, template, per_session, acute_rpe))
    if cardio:
        sessions.append(_cardio_session(TODAY, "HIIT", acute_rpe, "8x400m intervals"))

    return sorted(sessions, key=lambda s: s["date"])


PERSONAS = {
    # chronic weekly volume, target ACWR, template, chronic RPE by week, acute RPE, cardio
    "red_overreach": (7500, 1.65, STRENGTH_TEMPLATE, [7.5, 8.0, 8.4, 8.8], 9.2, False),
    "amber_caution": (7500, 1.40, STRENGTH_TEMPLATE, [7.5, 7.6, 7.9, 8.1], 8.3, False),
    # Green-zone personas keep flat RPEs: a rising trend makes the fatigue
    # agent (correctly) raise the level and the persona stops demoing "increase".
    "green_progress": (9000, 1.05, STRENGTH_TEMPLATE, [7.2, 7.2, 7.2, 7.2], 7.2, False),
    "fat_loss_cut": (6000, 1.10, DUMBBELL_TEMPLATE, [7.0, 7.0, 7.0, 7.0], 7.0, False),
    "endurance_runner": (3000, 1.05, LIGHT_TEMPLATE, [6.8, 6.8, 6.8, 6.8], 6.8, True),
    "default_user": (7000, 1.05, STRENGTH_TEMPLATE, [7.2, 7.2, 7.2, 7.2], 7.2, False),
    "fresh_user": None,
}


def seed() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Seeding relative to week of {THIS_MONDAY.isoformat()} (today: {TODAY.isoformat()})\n")

    for user_id, spec in PERSONAS.items():
        sessions = [] if spec is None else _build_persona(*spec)
        with open(DATA_DIR / f"{user_id}.json", "w") as f:
            json.dump(sessions, f, indent=2)

        # Verify through the real store + fatigue code, not our own assumptions.
        acwr = _compute_acwr(get_history(user_id, weeks=5), get_current_week(user_id))
        level, status = _classify_fatigue(acwr)
        acwr_str = f"{acwr:.3f}" if acwr is not None else "None"
        print(
            f"{user_id:18s} sessions={len(sessions):2d} "
            f"ACWR={acwr_str:6s} status={status:11s} level={level.upper()}"
        )


if __name__ == "__main__":
    seed()
