"""Tests for the deterministic nutrition calculator."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from graph.nodes.nutrition_advisor import compute_nutrition_targets, _bmr, _activity_multiplier


PROFILE = {
    "goals": "hypertrophy",
    "available_days": 4,
    "weight_kg": 84,
    "height_cm": 179,
    "age": 24,
}


class TestBMR:
    def test_mifflin_neutral(self):
        # 10*84 + 6.25*179 - 5*24 - 78 = 840 + 1118.75 - 120 - 78
        assert _bmr(84, 179, 24) == 1760.75

    def test_activity_multipliers(self):
        assert _activity_multiplier(2) == 1.4
        assert _activity_multiplier(3) == 1.55
        assert _activity_multiplier(4) == 1.65
        assert _activity_multiplier(6) == 1.725


class TestNutritionTargets:
    def test_output_shape(self):
        t = compute_nutrition_targets(PROFILE, {"action": "maintain"})
        for day in ("training_day", "rest_day"):
            assert set(t[day]) == {"calories", "protein_g", "carbs_g", "fat_g"}
        assert isinstance(t["timing_notes"], str)

    def test_training_day_exceeds_rest_day(self):
        t = compute_nutrition_targets(PROFILE, {"action": "maintain"})
        gap = t["training_day"]["calories"] - t["rest_day"]["calories"]
        assert 300 <= gap <= 500
        assert t["training_day"]["carbs_g"] > t["rest_day"]["carbs_g"]
        # protein and fat stay constant across day types
        assert t["training_day"]["protein_g"] == t["rest_day"]["protein_g"]
        assert t["training_day"]["fat_g"] == t["rest_day"]["fat_g"]

    def test_deload_cuts_calories(self):
        maintain = compute_nutrition_targets(PROFILE, {"action": "maintain"})
        deload = compute_nutrition_targets(PROFILE, {"action": "deload"})
        assert deload["training_day"]["calories"] < maintain["training_day"]["calories"]

    def test_increase_adds_calories(self):
        maintain = compute_nutrition_targets(PROFILE, {"action": "maintain"})
        increase = compute_nutrition_targets(PROFILE, {"action": "increase"})
        assert increase["training_day"]["calories"] > maintain["training_day"]["calories"]

    def test_fat_loss_runs_deficit_with_high_protein(self):
        cut_profile = dict(PROFILE, goals="fat loss", weight_kg=96)
        bulk = compute_nutrition_targets(dict(PROFILE, weight_kg=96), {"action": "maintain"})
        cut = compute_nutrition_targets(cut_profile, {"action": "maintain"})
        assert cut["training_day"]["calories"] < bulk["training_day"]["calories"]
        assert cut["training_day"]["protein_g"] == round(2.4 * 96)

    def test_missing_metrics_fall_back_to_defaults(self):
        t = compute_nutrition_targets({"goals": "strength"}, {})
        assert t["training_day"]["protein_g"] == round(2.0 * 80)
        assert t["training_day"]["calories"] > 0
