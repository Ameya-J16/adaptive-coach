"""Golden evaluation set for the critic.

14 hand-labeled cases in three groups:
- GOOD: well-constructed plans that must pass (score >= 0.75, zero violations)
- MECHANICAL-BAD: plans with rule violations the deterministic layer must
  catch (guaranteed fail via the violation score ceiling)
- JUDGE-BAD: plans that are mechanically clean but qualitatively flawed —
  only the LLM judge can catch these. Labeled per-dimension (the flawed
  dimension must score <= 0.6) rather than by overall verdict, because a
  single soft flaw does not deterministically decide pass/fail under
  4-dimension averaging.

Retrieved-context excerpts quote rag/knowledge_base/sports_science.md so
groundedness checks run against the project's real evidence base.
"""
from datetime import datetime, timedelta

DIMENSION_LOW_THRESHOLD = 0.6
PASS_THRESHOLD = 0.75

# ---------------------------------------------------------------- helpers

DAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]


def ex(name, sets, reps, weight, rpe, muscles):
    return {"name": name, "sets": sets, "reps": reps, "weight_kg": weight,
            "rpe_target": rpe, "muscle_groups": muscles}


def day(session_type, exercises, notes=""):
    return {"session_type": session_type, "exercises": exercises, "notes": notes}


REST = day("rest", [], "Planned rest for recovery.")


def plan(overall_notes="", **named_days):
    days = {d: REST for d in DAYS}
    days.update(named_days)
    return {"week_start_date": "2026-08-10", "days": days, "overall_notes": overall_notes}


def plan_volume(p):
    return sum((e["sets"] * e["reps"] * e["weight_kg"])
               for d in p["days"].values() for e in d.get("exercises", []))


def scale_to_volume(p, target_volume):
    """Scale every weight so total plan volume hits target_volume exactly-ish."""
    current = plan_volume(p)
    factor = target_volume / current
    for d in p["days"].values():
        for e in d.get("exercises", []):
            e["weight_kg"] = round(e["weight_kg"] * factor, 1)
    return p


def history(weekly_volume, weeks=4):
    """One synthetic session per prior calendar week carrying the whole volume."""
    today = datetime.now().date()
    this_monday = today - timedelta(days=today.weekday())
    return [{"date": (this_monday - timedelta(weeks=w + 1)).isoformat(),
             "exercises": [{"sets": 1, "reps": 1, "weight_kg": weekly_volume}]}
            for w in range(weeks)]


# ------------------------------------------------- knowledge-base excerpts
#
# Contexts come from the real RAG index when available (regenerate with
# `python -m scripts.refresh_eval_contexts`), so the eval scores the critic
# against the evidence the live system would actually retrieve. The inline
# fallbacks below keep the eval runnable before the index is built.

import json as _json
from pathlib import Path as _Path

_CONTEXT_CACHE_PATH = _Path(__file__).parent / "retrieved_contexts.json"
try:
    _CACHED = _json.loads(_CONTEXT_CACHE_PATH.read_text(encoding="utf-8"))
except (OSError, ValueError):
    _CACHED = {}

KB_INCREASE = """[Source 1]
A conservative volume increase of 5-10% per week is sustainable for most intermediate lifters. \
Beginners can tolerate higher weekly increases (10-15%). Advanced athletes should limit volume \
increases to 3-5% per week to avoid rapid fatigue accumulation.

[Source 2]
Training each muscle group 2x per week outperforms 1x per week for hypertrophy. For strength, \
3-4 sessions per week on major movements is optimal."""

KB_DELOAD = """[Source 1]
ACWR > 1.5: High overreach risk. Injury risk increases significantly. Immediate load reduction \
is strongly recommended. A deload week should be implemented.

[Source 2]
When ACWR > 1.5, the training plan must include at least 2 full rest days in the upcoming week \
and reduce compound lift intensity by at least 20%.

[Source 3]
Volume deload (recommended for most): Reduce total sets per session by 40-50% while maintaining \
load. This preserves neuromuscular patterns and intensity adaptations."""

KB_AMBER = """[Source 1]
ACWR 1.3-1.5: Caution zone. Training load is increasing faster than the chronic baseline can \
accommodate. Monitor for fatigue signs and consider brief load reduction. When ACWR is in the \
amber zone (1.3-1.5), maintain current volume but avoid intensity increases.

[Source 2]
Staying below RPE 8 on compound movements during early training blocks reduces injury risk and \
allows sustainable volume increases."""

KB_BEGINNER = """[Source 1]
Linear progression: Add 2.5-5kg per session (beginners). Accumulation phases typically target \
RPE 6-7.

[Source 2]
Minimum Effective Volume (MEV): The minimum sets per muscle group per week needed to maintain \
adaptations. Approximately 8-10 sets per muscle group per week."""

KB_ENDURANCE = """[Source 1]
Training intensity distribution research supports a polarized model for endurance athletes: the \
majority of training volume at low intensity with a smaller portion at high intensity, rather \
than concentrating work at threshold. Concurrent strength training supports running economy \
without compromising endurance adaptation when sessions are adequately separated."""

# Prefer the real retriever's output; fall back to the inline excerpt above.
KB_INCREASE = _CACHED.get("KB_INCREASE", KB_INCREASE)
KB_DELOAD = _CACHED.get("KB_DELOAD", KB_DELOAD)
KB_AMBER = _CACHED.get("KB_AMBER", KB_AMBER)
KB_BEGINNER = _CACHED.get("KB_BEGINNER", KB_BEGINNER)
KB_ENDURANCE = _CACHED.get("KB_ENDURANCE", KB_ENDURANCE)

# ------------------------------------------------------- profiles/reports

GREEN_HYPERTROPHY = {
    "profile": {"goals": "hypertrophy", "experience_level": "intermediate", "available_days": 5,
                "weight_kg": 84, "height_cm": 179, "age": 24},
    "fatigue": {"fatigue_level": "green", "acwr": 1.05, "acwr_status": "computed",
                "dominant_stress_pattern": "stable RPE at stable loads",
                "recommendation_hint": "Ready for a modest volume increase."},
}
RED_STRENGTH = {
    "profile": {"goals": "strength", "experience_level": "intermediate", "available_days": 4,
                "weight_kg": 82, "height_cm": 178, "age": 29},
    "fatigue": {"fatigue_level": "red", "acwr": 1.66, "acwr_status": "computed",
                "dominant_stress_pattern": "acute load spike well above chronic baseline",
                "recommendation_hint": "Immediate load reduction required."},
}
AMBER_STRENGTH = {
    "profile": {"goals": "strength", "experience_level": "intermediate", "available_days": 4,
                "weight_kg": 78, "height_cm": 175, "age": 31},
    "fatigue": {"fatigue_level": "amber", "acwr": 1.41, "acwr_status": "computed",
                "dominant_stress_pattern": "load rising faster than baseline",
                "recommendation_hint": "Hold volume, avoid intensity increases."},
}
FRESH_BEGINNER = {
    "profile": {"goals": "strength", "experience_level": "beginner", "available_days": 3,
                "weight_kg": 70, "height_cm": 170, "age": 22},
    "fatigue": {"fatigue_level": "amber", "acwr": None, "acwr_status": "no_baseline",
                "dominant_stress_pattern": "no training history",
                "recommendation_hint": "Start conservatively to establish a baseline."},
}
GREEN_ENDURANCE = {
    "profile": {"goals": "endurance", "experience_level": "intermediate", "available_days": 5,
                "weight_kg": 66, "height_cm": 172, "age": 27},
    "fatigue": {"fatigue_level": "green", "acwr": 1.07, "acwr_status": "computed",
                "dominant_stress_pattern": "consistent aerobic load",
                "recommendation_hint": "Room for a small volume increase."},
}


def _case(name, description, persona, decision, retrieved, weekly_plan, hist,
          expected_verdict=None, expect_violations=(), expect_low_dimensions=()):
    return {
        "name": name,
        "description": description,
        "expected_verdict": expected_verdict,
        "expect_violations": list(expect_violations),
        "expect_low_dimensions": list(expect_low_dimensions),
        "state": {
            "weekly_plan": weekly_plan,
            "fatigue_report": persona["fatigue"],
            "progression_decision": decision,
            "user_profile": persona["profile"],
            "retrieved_context": retrieved,
            "workout_history": hist,
        },
    }


# ---------------------------------------------------------------- plans


def _upper(rpe, notes=""):
    return day("strength", [
        ex("Bench Press", 4, 8, 60, rpe, ["chest", "triceps"]),
        ex("Barbell Row", 4, 8, 55, rpe, ["back", "biceps"]),
        ex("Overhead Press", 3, 10, 35, rpe - 0.5, ["shoulders"]),
    ], notes)


def _lower(rpe, notes=""):
    return day("strength", [
        ex("Squat", 4, 6, 90, rpe, ["quads", "glutes"]),
        ex("Romanian Deadlift", 3, 8, 70, rpe - 0.5, ["hamstrings", "glutes"]),
        ex("Leg Press", 3, 10, 120, rpe - 1.0, ["quads"]),
    ], notes)


def build_cases() -> list[dict]:
    cases = []

    # ---------------- GOOD ----------------
    p = plan(
        "Volume increased ~7% per the 5-10% weekly guideline for intermediates; "
        "each muscle group trained 2x this week for hypertrophy.",
        monday=_upper(7.5, "Upper A - moderate RPE per accumulation guidance."),
        tuesday=_lower(7.5, "Lower A - quads/glutes focus."),
        thursday=_upper(7.0, "Upper B - second weekly exposure."),
        friday=_lower(7.0, "Lower B - second weekly exposure."),
    )
    scale_to_volume(p, 9600)  # 106.7% of 9000 baseline
    cases.append(_case(
        "good_green_increase",
        "Green athlete, increase action, volume +7%, 2x frequency, grounded notes",
        GREEN_HYPERTROPHY, {"action": "increase", "load_adjustment_percent": 7.0,
                            "exercises_to_swap": []},
        KB_INCREASE, p, history(9000), expected_verdict="pass",
    ))

    p = plan(
        "Deload: volume cut to ~60% of baseline with 4 full rest days per ACWR>1.5 protocol; "
        "compound intensity reduced over 20%.",
        monday=day("deload", [
            ex("Squat", 2, 5, 70, 6.0, ["quads", "glutes"]),
            ex("Bench Press", 2, 6, 45, 6.0, ["chest", "triceps"]),
        ], "Half normal set count, load kept moderate."),
        wednesday=day("deload", [
            ex("Deadlift", 2, 4, 90, 6.5, ["hamstrings", "back"]),
            ex("Overhead Press", 2, 8, 25, 6.0, ["shoulders"]),
        ], "Technique focus, long rests."),
        friday=day("deload", [
            ex("Front Squat", 2, 5, 55, 6.0, ["quads"]),
            ex("Barbell Row", 2, 8, 40, 6.0, ["back", "biceps"]),
        ], "Final light exposure before normal training resumes."),
    )
    scale_to_volume(p, 4500)  # 60% of 7500
    cases.append(_case(
        "good_red_deload",
        "Red athlete, deload action, volume 60% of baseline, RPE ~6, 4 rest days",
        RED_STRENGTH, {"action": "deload", "load_adjustment_percent": -35.0,
                       "exercises_to_swap": []},
        KB_DELOAD, p, history(7500), expected_verdict="pass",
    ))

    p = plan(
        "Amber zone: volume held at baseline, no intensity increases, compounds kept below RPE 8.",
        monday=_upper(7.5), wednesday=_lower(7.5), friday=_upper(7.0), saturday=_lower(7.0),
    )
    scale_to_volume(p, 7400)  # ~99% of 7500
    cases.append(_case(
        "good_amber_maintain",
        "Amber athlete, maintain action, volume ~99% of baseline, RPE <= 8",
        AMBER_STRENGTH, {"action": "maintain", "load_adjustment_percent": 0.0,
                         "exercises_to_swap": []},
        KB_AMBER, p, history(7500), expected_verdict="pass",
    ))

    p = plan(
        "First week: conservative dumbbell loads at RPE 6 to establish a baseline, "
        "linear progression to follow.",
        monday=day("strength", [
            ex("Goblet Squat", 3, 10, 12, 6.0, ["quads", "glutes"]),
            ex("Dumbbell Bench Press", 3, 10, 10, 6.0, ["chest", "triceps"]),
            ex("Dumbbell Row", 3, 10, 12, 6.0, ["back", "biceps"]),
        ], "Learn movement patterns; stop well short of failure."),
        wednesday=day("strength", [
            ex("Dumbbell Romanian Deadlift", 3, 10, 12, 6.0, ["hamstrings", "glutes"]),
            ex("Dumbbell Shoulder Press", 3, 10, 8, 6.0, ["shoulders"]),
            ex("Goblet Squat", 3, 10, 12, 6.0, ["quads", "glutes"]),
        ], "Repeat exposure at identical loads."),
        friday=day("strength", [
            ex("Dumbbell Bench Press", 3, 10, 10, 6.0, ["chest", "triceps"]),
            ex("Dumbbell Row", 3, 10, 12, 6.0, ["back", "biceps"]),
            ex("Dumbbell Lunge", 3, 10, 8, 6.0, ["quads", "glutes"]),
        ], "Third exposure; add 2.5kg next week if all sets complete."),
    )
    cases.append(_case(
        "good_fresh_start",
        "No-history beginner, conservative starter plan at RPE 6",
        FRESH_BEGINNER, {"action": "maintain", "load_adjustment_percent": 0.0,
                         "exercises_to_swap": []},
        KB_BEGINNER, p, [], expected_verdict="pass",
    ))

    p = plan(
        "Maintain with swaps honoured: barbell squat replaced by leg press due to "
        "reported knee discomfort; volume held at baseline.",
        monday=day("strength", [
            ex("Leg Press", 4, 10, 130, 7.0, ["quads", "glutes"]),
            ex("Romanian Deadlift", 3, 8, 65, 7.0, ["hamstrings", "glutes"]),
        ], "Squat pattern replaced per progression decision."),
        wednesday=_upper(7.0),
        friday=day("strength", [
            ex("Leg Curl", 3, 12, 40, 7.0, ["hamstrings"]),
            ex("Leg Extension", 3, 12, 45, 6.5, ["quads"]),
            ex("Calf Raise", 3, 15, 60, 6.5, ["calves"]),
        ], "Machine-based lower day."),
        saturday=_upper(6.5),
    )
    scale_to_volume(p, 7000)
    cases.append(_case(
        "good_swap_honoured",
        "Maintain with exercises_to_swap correctly substituted",
        AMBER_STRENGTH, {"action": "maintain", "load_adjustment_percent": 0.0,
                         "exercises_to_swap": ["Squat"]},
        KB_AMBER, p, history(7000), expected_verdict="pass",
    ))

    p = plan(
        "Polarized endurance block: one high-intensity interval session, all other running "
        "at low intensity with no threshold work, per training-intensity-distribution "
        "evidence. Strength support 2x for running economy, placed away from the quality "
        "run to protect concurrent-training adaptation.",
        monday=day("cardio", [], "Intervals: 6x800m at 5k pace, RPE 9 — the single "
                                 "high-intensity session of the week."),
        tuesday=day("strength", [
            ex("Goblet Squat", 3, 10, 20, 6.5, ["quads", "glutes"]),
            ex("Walking Lunge", 3, 12, 10, 6.5, ["quads", "glutes"]),
            ex("Calf Raise", 3, 15, 30, 6.0, ["calves"]),
        ], "Strength support for running economy, 24h after the quality session."),
        wednesday=day("cardio", [], "Easy run 45min, RPE 4 — strictly conversational."),
        thursday=day("cardio", [], "Easy run 40min, RPE 4. Deliberately not a tempo run: "
                                   "threshold work is what polarized distributions avoid."),
        friday=day("strength", [
            ex("Dumbbell Romanian Deadlift", 3, 10, 16, 6.5, ["hamstrings", "glutes"]),
            ex("Step-up", 3, 10, 10, 6.5, ["quads", "glutes"]),
            ex("Plank Row", 3, 10, 8, 6.0, ["core", "back"]),
        ], "Posterior chain support."),
        sunday=day("cardio", [], "Long run 90min easy, RPE 5."),
    )
    scale_to_volume(p, 3150)  # 105% of 3000
    cases.append(_case(
        "good_endurance_mix",
        "Endurance athlete: run quality + long run + 2 strength support days",
        GREEN_ENDURANCE, {"action": "increase", "load_adjustment_percent": 5.0,
                          "exercises_to_swap": []},
        KB_ENDURANCE, p, history(3000), expected_verdict="pass",
    ))

    # ------------- MECHANICAL-BAD (deterministic layer must catch) -------------
    def heavy_lower():
        return day("strength", [
            ex("Squat", 5, 5, 100, 9.0, ["quads", "glutes"]),
            ex("Romanian Deadlift", 4, 6, 80, 8.5, ["hamstrings", "glutes"]),
        ], "Heavy lower day.")
    p = plan("Back-to-back heavy lower sessions to maximise stimulus.",
             monday=heavy_lower(), tuesday=heavy_lower(),
             thursday=_upper(7.5), friday=REST)
    scale_to_volume(p, 9600)
    cases.append(_case(
        "bad_consecutive_heavy",
        "Same heavy lower-body session on consecutive days",
        GREEN_HYPERTROPHY, {"action": "increase", "load_adjustment_percent": 7.0,
                            "exercises_to_swap": []},
        KB_INCREASE, p, history(9000),
        expected_verdict="fail", expect_violations=["consecutive"],
    ))

    p = plan("Deload week focusing on recovery.",
             monday=_lower(7.0), wednesday=_upper(7.0), friday=_lower(6.5), saturday=_upper(6.5))
    scale_to_volume(p, 7400)  # ~99% of baseline: not a deload at all
    cases.append(_case(
        "bad_deload_no_reduction",
        "Deload action but planned volume ~= baseline",
        RED_STRENGTH, {"action": "deload", "load_adjustment_percent": -35.0,
                       "exercises_to_swap": []},
        KB_DELOAD, p, history(7500),
        expected_verdict="fail", expect_violations=["deload"],
    ))

    p = plan("Progressive overload week.",
             monday=_upper(7.5), wednesday=_lower(7.5), friday=_upper(7.0))
    scale_to_volume(p, 7200)  # 80% of 9000: an "increase" that decreases
    cases.append(_case(
        "bad_increase_decrease",
        "Increase action but planned volume is 80% of baseline",
        GREEN_HYPERTROPHY, {"action": "increase", "load_adjustment_percent": 7.0,
                            "exercises_to_swap": []},
        KB_INCREASE, p, history(9000),
        expected_verdict="fail", expect_violations=["increase"],
    ))

    p = plan("Maintain week with planned substitutions.",
             monday=day("strength", [
                 ex("Barbell Squat", 4, 6, 85, 7.5, ["quads", "glutes"]),
                 ex("Leg Curl", 3, 12, 40, 6.5, ["hamstrings"]),
             ], "Lower day."),
             wednesday=_upper(7.0), friday=_lower(7.0), saturday=_upper(6.5))
    scale_to_volume(p, 7000)
    cases.append(_case(
        "bad_swap_ignored",
        "Exercise marked for substitution still appears in the plan",
        AMBER_STRENGTH, {"action": "maintain", "load_adjustment_percent": 0.0,
                         "exercises_to_swap": ["Barbell Squat"]},
        KB_AMBER, p, history(7000),
        expected_verdict="fail", expect_violations=["substitution"],
    ))

    # ------------- JUDGE-BAD (mechanically clean; LLM must catch) -------------
    p = plan(
        "Research shows 25-30% weekly volume jumps are standard for intermediates, and "
        "deloads are unnecessary for athletes under 30, so this block pushes volume "
        "aggressively per the retrieved evidence.",
        monday=_upper(7.5), tuesday=_lower(7.5), thursday=_upper(7.0), friday=_lower(7.0),
    )
    scale_to_volume(p, 9600)  # volume itself is fine; the CLAIMS are false
    cases.append(_case(
        "bad_ungrounded_claims",
        "Plan notes cite claims the knowledge base contradicts (25-30% jumps, no deloads needed)",
        GREEN_HYPERTROPHY, {"action": "increase", "load_adjustment_percent": 7.0,
                            "exercises_to_swap": []},
        KB_INCREASE, p, history(9000),
        expect_low_dimensions=["groundedness"],
    ))

    p = plan(
        "Endurance development block built around maximal strength.",
        monday=day("strength", [
            ex("Squat", 5, 3, 110, 8.5, ["quads", "glutes"]),
            ex("Deadlift", 5, 3, 130, 8.5, ["hamstrings", "back"]),
        ], "Heavy triples."),
        wednesday=day("strength", [
            ex("Bench Press", 5, 3, 80, 8.5, ["chest", "triceps"]),
            ex("Overhead Press", 5, 3, 50, 8.0, ["shoulders"]),
        ], "Heavy pressing."),
        friday=day("strength", [
            ex("Front Squat", 5, 3, 90, 8.5, ["quads"]),
            ex("Barbell Row", 5, 3, 70, 8.0, ["back", "biceps"]),
        ], "More heavy triples."),
    )
    scale_to_volume(p, 3100)
    cases.append(_case(
        "bad_wrong_goal_reps",
        "Endurance goal but the plan is pure heavy triples with zero conditioning",
        GREEN_ENDURANCE, {"action": "increase", "load_adjustment_percent": 5.0,
                          "exercises_to_swap": []},
        KB_ENDURANCE, p, history(3000),
        expect_low_dimensions=["goal_alignment"],
    ))

    p = plan(
        "Deload volume per protocol.",
        monday=day("deload", [ex("Squat", 2, 5, 100, 9.0, ["quads", "glutes"])],
                   "Low volume but near-maximal effort."),
        wednesday=day("deload", [ex("Bench Press", 2, 5, 75, 9.0, ["chest", "triceps"])],
                      "Heavy doubles feel."),
        friday=day("deload", [ex("Deadlift", 2, 3, 140, 9.0, ["hamstrings", "back"])],
                   "Grinding pulls."),
    )
    scale_to_volume(p, 4200)  # volume properly cut, intensity is the problem
    cases.append(_case(
        "bad_deload_max_intensity",
        "Deload with correct volume cut but RPE 9 grinding on every lift",
        RED_STRENGTH, {"action": "deload", "load_adjustment_percent": -35.0,
                       "exercises_to_swap": []},
        KB_DELOAD, p, history(7500),
        expect_low_dimensions=["safety"],
    ))

    p = plan(
        "Seven consecutive training days alternating muscle groups for maximum weekly stimulus.",
        monday=_upper(7.5), tuesday=_lower(7.5), wednesday=_upper(7.0), thursday=_lower(7.0),
        friday=_upper(7.0), saturday=_lower(6.5),
        sunday=day("strength", [ex("Dumbbell Curl", 3, 12, 12, 6.5, ["biceps"]),
                                ex("Triceps Pushdown", 3, 12, 25, 6.5, ["triceps"])],
                   "Arm pump day."),
    )
    scale_to_volume(p, 7500)
    cases.append(_case(
        "bad_no_rest_days",
        "Seven straight training days, zero rest, on a maintain week",
        AMBER_STRENGTH, {"action": "maintain", "load_adjustment_percent": 0.0,
                         "exercises_to_swap": []},
        KB_AMBER, p, history(7500),
        expect_low_dimensions=["safety"],
    ))

    return cases
