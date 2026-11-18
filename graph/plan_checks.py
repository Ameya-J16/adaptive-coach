"""Deterministic plan-quality checks.

The critic's safety and coherence rubric items are mechanical (consecutive
heavy days, deload volume math, RPE caps, swap compliance), so they are
enforced here in Python as hard caps the LLM judge cannot override. The LLM
still judges the qualitative dimensions (groundedness, goal alignment) and
can score BELOW a cap — never above it.
"""
from datetime import datetime, timedelta

DAY_ORDER = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]

HARD_RPE = 7.0          # an exercise above this makes a day "hard"
DELOAD_MAX_RPE = 9.0    # no deload exercise may target above this
DELOAD_MAX_VOLUME_RATIO = 0.85   # deload plan volume must be <= 85% of baseline
MAINTAIN_TOLERANCE = 0.15        # maintain must stay within +-15% of baseline


def _plan_exercise_volume(ex: dict) -> float:
    return (ex.get("sets") or 0) * (ex.get("reps") or 0) * (ex.get("weight_kg") or 0)


def plan_volume(weekly_plan: dict) -> float:
    total = 0.0
    for day in weekly_plan.get("days", {}).values():
        for ex in day.get("exercises", []):
            total += _plan_exercise_volume(ex)
    return total


def chronic_weekly_average(workout_history: list[dict]) -> float | None:
    """Average weekly volume over the chronic window — the same rolling 28-day
    window ACWR uses, so the critic judges a plan against the identical
    baseline the fatigue analysis was derived from. None if empty."""
    from graph.nodes.fatigue_analyst import ACUTE_DAYS, CHRONIC_DAYS

    def session_volume(s: dict) -> float:
        total = 0.0
        for ex in s.get("exercises", []):
            if ex.get("sets_data"):
                total += sum(st.get("reps", 0) * st.get("weight_kg", 0) for st in ex["sets_data"])
            else:
                total += _plan_exercise_volume(ex)
        return total

    today = datetime.now().date()
    chronic_end = today - timedelta(days=ACUTE_DAYS)
    chronic_start = chronic_end - timedelta(days=CHRONIC_DAYS - 1)
    total = sum(
        session_volume(s) for s in workout_history
        if chronic_start.isoformat() <= s.get("date", "") <= chronic_end.isoformat()
    )
    if total == 0:
        return None
    return total / (CHRONIC_DAYS / ACUTE_DAYS)


def _hard_muscle_groups(day: dict) -> set[str]:
    """Muscle groups loaded above HARD_RPE on this day."""
    groups: set[str] = set()
    for ex in day.get("exercises", []):
        if (ex.get("rpe_target") or 0) > HARD_RPE:
            groups.update(mg.lower() for mg in ex.get("muscle_groups", []))
    return groups


def check_plan(
    weekly_plan: dict,
    fatigue_report: dict,
    progression_decision: dict,
    workout_history: list[dict],
) -> dict:
    """Run every mechanical rubric check.

    Returns {"safety_cap": float, "coherence_cap": float, "violations": [str]}.
    Caps start at 1.0 and only go down; violations are human-readable and get
    fed to both the LLM judge and the plan_writer's revision feedback.
    """
    violations: list[str] = []
    safety_cap = 1.0
    coherence_cap = 1.0
    days = weekly_plan.get("days", {})

    # -- Safety: consecutive days hammering the same muscle groups hard --
    for a, b in zip(DAY_ORDER, DAY_ORDER[1:]):
        shared = _hard_muscle_groups(days.get(a, {})) & _hard_muscle_groups(days.get(b, {}))
        if shared:
            safety_cap = 0.0
            violations.append(
                f"SAFETY: {a} and {b} both load {sorted(shared)} above RPE {HARD_RPE} — "
                "consecutive heavy sessions for the same muscle group."
            )

    # -- Safety: red fatigue but 4+ hard training days --
    hard_days = [d for d in DAY_ORDER if _hard_muscle_groups(days.get(d, {}))]
    if fatigue_report.get("fatigue_level") == "red" and len(hard_days) >= 4:
        safety_cap = min(safety_cap, 0.7)
        violations.append(
            f"SAFETY: fatigue is RED but the plan has {len(hard_days)} hard days ({hard_days})."
        )

    # -- Safety: deload sessions must stay at or below RPE 9 --
    action = progression_decision.get("action", "maintain")
    if action == "deload":
        for d in DAY_ORDER:
            for ex in days.get(d, {}).get("exercises", []):
                if (ex.get("rpe_target") or 0) > DELOAD_MAX_RPE:
                    safety_cap = min(safety_cap, 0.8)
                    violations.append(
                        f"SAFETY: deload week but {ex.get('name')} on {d} targets "
                        f"RPE {ex.get('rpe_target')} (max {DELOAD_MAX_RPE})."
                    )

    # -- Coherence: planned volume must implement the progression action --
    baseline = chronic_weekly_average(workout_history)
    volume = plan_volume(weekly_plan)
    if baseline:
        ratio = volume / baseline
        if action == "deload" and ratio > DELOAD_MAX_VOLUME_RATIO:
            coherence_cap = 0.0
            violations.append(
                f"COHERENCE: action is deload but planned volume is {volume:.0f}kg = "
                f"{ratio:.0%} of the {baseline:.0f}kg weekly baseline (must be <= "
                f"{DELOAD_MAX_VOLUME_RATIO:.0%})."
            )
        elif action == "increase" and ratio <= 1.0:
            coherence_cap = min(coherence_cap, 0.4)
            violations.append(
                f"COHERENCE: action is increase but planned volume is {volume:.0f}kg = "
                f"{ratio:.0%} of baseline — no actual increase."
            )
        elif action == "maintain" and abs(ratio - 1.0) > MAINTAIN_TOLERANCE:
            coherence_cap = min(coherence_cap, 0.6)
            violations.append(
                f"COHERENCE: action is maintain but planned volume is {ratio:.0%} of "
                f"baseline (outside +-{MAINTAIN_TOLERANCE:.0%})."
            )

    # -- Coherence: swapped-out exercises must not reappear --
    swaps = [s.lower() for s in progression_decision.get("exercises_to_swap", [])]
    if swaps:
        for d in DAY_ORDER:
            for ex in days.get(d, {}).get("exercises", []):
                name = (ex.get("name") or "").lower()
                if any(s in name or name in s for s in swaps):
                    coherence_cap = min(coherence_cap, 0.5)
                    violations.append(
                        f"COHERENCE: {ex.get('name')} on {d} was marked for substitution "
                        "but still appears in the plan."
                    )

    return {"safety_cap": safety_cap, "coherence_cap": coherence_cap, "violations": violations}


# Any mechanical violation forces the overall score to at most this value —
# below the 0.75 replan threshold — because averaging alone cannot guarantee a
# rejection (safety 0.0 with three 1.0s still averages 0.75).
VIOLATION_SCORE_CEILING = 0.7


def apply_caps(llm_scores: dict, checks: dict) -> tuple[dict, float]:
    """Combine judge scores with deterministic caps. Returns (scores, overall)."""
    scores = {
        "safety": min(llm_scores.get("safety", 0.5), checks["safety_cap"]),
        "coherence": min(llm_scores.get("coherence", 0.5), checks["coherence_cap"]),
        "groundedness": llm_scores.get("groundedness", 0.5),
        "goal_alignment": llm_scores.get("goal_alignment", 0.5),
    }
    overall = sum(scores.values()) / len(scores)
    if checks["violations"]:
        overall = min(overall, VIOLATION_SCORE_CEILING)
    return scores, round(overall, 3)
