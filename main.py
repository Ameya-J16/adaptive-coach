"""
CLI entry point for AdaptiveCoach.

Usage:
  python main.py --user-id <id> --action plan
  python main.py --user-id <id> --action log
  python main.py --user-id <id> --action history
"""
import argparse
import json
import sys
from datetime import datetime

from dotenv import load_dotenv

load_dotenv()


def run_plan(user_id: str) -> None:
    from graph.graph import graph
    from memory.workout_store import get_history, get_current_week

    print(f"\n[AdaptiveCoach] Generating adaptive plan for user: {user_id}")
    print("Running multi-agent pipeline...\n")

    initial_state = {
        "user_id": user_id,
        "user_profile": _load_profile(user_id),
        "workout_history": [],
        "current_week_log": [],
        "fatigue_report": {},
        "retrieved_context": "",
        "progression_decision": {},
        "nutrition_targets": {},
        "weekly_plan": {},
        "critic_score": 0.0,
        "critic_feedback": "",
        "loop_count": 0,
        "messages": [],
    }

    result = graph.invoke(initial_state)

    print("=" * 60)
    print("FATIGUE REPORT")
    print("=" * 60)
    fr = result.get("fatigue_report", {})
    print(f"  Level  : {fr.get('fatigue_level', 'N/A').upper()}")
    print(f"  ACWR   : {fr.get('acwr', 'N/A')}")
    print(f"  Pattern: {fr.get('dominant_stress_pattern', 'N/A')}")
    print(f"  Hint   : {fr.get('recommendation_hint', 'N/A')}")

    print("\n" + "=" * 60)
    print("PROGRESSION DECISION")
    print("=" * 60)
    pd = result.get("progression_decision", {})
    print(f"  Action      : {pd.get('action', 'N/A')}")
    print(f"  Load adjust : {pd.get('load_adjustment_percent', 0):+.1f}%")
    print(f"  Rationale   : {pd.get('rationale', 'N/A')}")

    print("\n" + "=" * 60)
    print("WEEKLY PLAN")
    print("=" * 60)
    plan = result.get("weekly_plan", {})
    days = plan.get("days", {})
    for day, content in days.items():
        print(f"\n  {day.upper()} — {content.get('session_type', 'rest').upper()}")
        for ex in content.get("exercises", []):
            print(f"    • {ex.get('name')}: {ex.get('sets')}×{ex.get('reps')} @ {ex.get('weight_kg')}kg (RPE {ex.get('rpe_target')})")
        if content.get("notes"):
            print(f"    Note: {content['notes']}")

    print("\n" + "=" * 60)
    print(f"CRITIC SCORE: {result.get('critic_score', 0):.2f} | Loops: {result.get('loop_count', 0)}")
    print("=" * 60)

    if plan.get("overall_notes"):
        print(f"\nOverall notes: {plan['overall_notes']}")


def run_log(user_id: str) -> None:
    from memory.workout_store import log_workout

    print(f"\n[AdaptiveCoach] Log a workout for user: {user_id}")
    session_type = input("Session type (strength/cardio/HIIT/rest/deload): ").strip() or "strength"
    avg_rpe = float(input("Average RPE for this session (1-10): ").strip() or "7")
    notes = input("Notes (optional): ").strip()

    exercises = []
    while True:
        name = input("\nExercise name (or press Enter to finish): ").strip()
        if not name:
            break
        sets = int(input("  Sets: ").strip() or "3")
        reps = int(input("  Reps: ").strip() or "8")
        weight_kg = float(input("  Weight (kg): ").strip() or "0")
        rpe = float(input("  RPE: ").strip() or str(avg_rpe))
        exercises.append({
            "name": name,
            "sets": sets,
            "reps": reps,
            "weight_kg": weight_kg,
            "rpe_target": rpe,
            "muscle_groups": [],
        })

    total_volume = sum(e["sets"] * e["reps"] * e["weight_kg"] for e in exercises)
    session = {
        "date": datetime.now().date().isoformat(),
        "exercises": exercises,
        "total_volume": total_volume,
        "avg_rpe": avg_rpe,
        "session_type": session_type,
        "notes": notes,
    }

    log_workout(user_id, session)
    print(f"\nWorkout logged! Total volume: {total_volume:.1f}kg")


def run_history(user_id: str) -> None:
    from memory.workout_store import get_all_sessions

    sessions = get_all_sessions(user_id)
    if not sessions:
        print(f"No workout history found for user: {user_id}")
        return

    print(f"\n[AdaptiveCoach] Workout history for {user_id} ({len(sessions)} sessions)\n")
    for s in sessions[-10:]:
        print(f"  {s.get('date')} | {s.get('session_type', 'N/A'):10s} | RPE {s.get('avg_rpe', 0):.1f} | Vol: {s.get('total_volume', 0):.0f}kg")


def _load_profile(user_id: str) -> dict:
    """Load user profile from data/profiles/{user_id}.json or return defaults."""
    import os
    from pathlib import Path
    profile_path = Path("data") / "profiles" / f"{user_id}.json"
    if profile_path.exists():
        with open(profile_path) as f:
            return json.load(f)
    return {
        "name": user_id,
        "goals": "strength",
        "experience_level": "intermediate",
        "available_days": 4,
        "equipment": ["barbell", "dumbbells", "rack"],
        "weight_kg": 80,
        "height_cm": 175,
        "age": 28,
    }


def main():
    parser = argparse.ArgumentParser(description="AdaptiveCoach CLI")
    parser.add_argument("--user-id", required=True, help="User identifier")
    parser.add_argument("--action", choices=["plan", "log", "history"], default="plan")
    args = parser.parse_args()

    actions = {"plan": run_plan, "log": run_log, "history": run_history}
    actions[args.action](args.user_id)


if __name__ == "__main__":
    main()
