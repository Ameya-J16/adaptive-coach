"""Nutrition targets, computed deterministically.

This was previously a GPT-4o call, but every rule in its prompt was
arithmetic (Mifflin-St Jeor BMR, activity multipliers, g/kg macro targets,
carb periodisation). Doing the maths in Python makes the output exact,
reproducible and free — an LLM adds nothing here.
"""
from graph.state import AgentState
from tracing.langfuse_config import trace_node

# Profiles do not record sex, so we use the midpoint of the male (+5) and
# female (-161) Mifflin-St Jeor constants. Documented, not hidden.
_MIFFLIN_NEUTRAL_CONSTANT = -78

_DEFAULTS = {"weight_kg": 80, "height_cm": 175, "age": 28}

# Carb periodisation: training days sit above the baseline, rest days below,
# giving the 400 kcal training/rest gap the original spec asked for (300-500).
_TRAINING_DAY_KCAL_SHIFT = 200


def _bmr(weight_kg: float, height_cm: float, age: float) -> float:
    return 10 * weight_kg + 6.25 * height_cm - 5 * age + _MIFFLIN_NEUTRAL_CONSTANT


def _activity_multiplier(training_days: int) -> float:
    if training_days >= 5:
        return 1.725
    if training_days == 4:
        return 1.65
    if training_days == 3:
        return 1.55
    return 1.4


def compute_nutrition_targets(user_profile: dict, progression_decision: dict) -> dict:
    weight = user_profile.get("weight_kg") or _DEFAULTS["weight_kg"]
    height = user_profile.get("height_cm") or _DEFAULTS["height_cm"]
    age = user_profile.get("age") or _DEFAULTS["age"]
    goals = (user_profile.get("goals") or "").lower()
    training_days = user_profile.get("available_days", 3)

    tdee = _bmr(weight, height, age) * _activity_multiplier(training_days)

    # Progression adjustment (deload: -12.5%, increase: +7.5% — midpoints of
    # the 10-15% / 5-10% ranges the original spec used).
    action = progression_decision.get("action", "maintain")
    if action == "deload":
        tdee *= 0.875
    elif action == "increase":
        tdee *= 1.075

    # Goal adjustment: fat loss runs a moderate deficit with higher protein.
    cutting = "fat" in goals and "loss" in goals
    if cutting:
        tdee *= 0.85
        protein_g = round(2.4 * weight)
    else:
        protein_g = round(2.0 * weight)

    fat_g = round(0.9 * weight)

    def day_targets(calories: float) -> dict:
        carb_kcal = calories - protein_g * 4 - fat_g * 9
        carbs_g = max(50, round(carb_kcal / 4))
        return {
            "calories": round(calories),
            "protein_g": protein_g,
            "carbs_g": carbs_g,
            "fat_g": fat_g,
        }

    timing_notes = (
        "Place most carbs around training: a carb + protein meal 2-3h pre-workout, "
        "and 0.3g/kg protein with fast carbs within 2h post-workout. "
        "A slow protein (e.g. casein or dairy) before bed supports overnight recovery."
    )
    if cutting:
        timing_notes += (
            " On the deficit, keep protein evenly spread across 4+ meals to protect lean mass."
        )

    return {
        "training_day": day_targets(tdee + _TRAINING_DAY_KCAL_SHIFT),
        "rest_day": day_targets(tdee - _TRAINING_DAY_KCAL_SHIFT),
        "timing_notes": timing_notes,
    }


def nutrition_advisor(state: AgentState) -> dict:
    with trace_node("nutrition_advisor"):
        targets = compute_nutrition_targets(
            state.get("user_profile", {}),
            state.get("progression_decision", {}),
        )
        return {"nutrition_targets": targets}
