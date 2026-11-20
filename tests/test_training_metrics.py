"""Tests for the RIR-based training metrics.

The point of these metrics is that they see things tonnage cannot, so several
tests are written as direct comparisons against the volume-load view.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from datetime import date, timedelta

from graph.nodes.fatigue_analyst import _exercise_volume
from graph.training_metrics import (
    FAILURE_RIR_THRESHOLD,
    MAV_SETS,
    failure_exposure,
    readiness_signals,
    rir_from_rpe,
    rpe_drift,
    stimulating_reps,
    weekly_muscle_summary,
)

TODAY = date.today()


def _sets(n, reps, weight, rpe):
    return [{"set_number": i + 1, "reps": reps, "weight_kg": weight, "rpe": rpe}
            for i in range(n)]


def _session(days_ago, exercises):
    return {"date": (TODAY - timedelta(days=days_ago)).isoformat(), "exercises": exercises}


def _ex(name, sets_data, muscles):
    return {"name": name, "sets_data": sets_data, "muscle_groups": list(muscles)}


class TestRIRModel:
    def test_rir_from_rpe(self):
        assert rir_from_rpe(10.0) == 0.0
        assert rir_from_rpe(8.0) == 2.0
        assert rir_from_rpe(None) is None
        assert rir_from_rpe(11.0) == 0.0, "RIR can never be negative"

    def test_sets_far_from_failure_give_no_stimulus(self):
        assert stimulating_reps(12, rir_from_rpe(4.0)) == 0

    def test_closer_to_failure_gives_more_stimulus(self):
        easy = stimulating_reps(10, rir_from_rpe(6.0))
        moderate = stimulating_reps(10, rir_from_rpe(8.0))
        hard = stimulating_reps(10, rir_from_rpe(10.0))
        assert easy < moderate < hard

    def test_short_set_to_failure_is_all_stimulus(self):
        # 5 reps at RIR 1 — every rep is within the threshold.
        assert stimulating_reps(5, 1.0) == 5

    def test_stimulus_cannot_exceed_reps_performed(self):
        assert stimulating_reps(2, 0.0) == 2


class TestSeesWhatTonnageCannot:
    def test_bodyweight_work_counts(self):
        pullups = _ex("Pull-Up", _sets(3, 10, 0, 9.0), ["back"])
        assert _exercise_volume(pullups) == 0, "tonnage is blind to bodyweight work"

        summary = weekly_muscle_summary([_session(1, [pullups])])
        assert summary["back"]["stimulating_reps"] > 0
        assert summary["back"]["hard_sets"] == 3

    def test_equal_tonnage_different_stimulus(self):
        """5x5@100 and 3x20@42 are near-identical tonnage, different sessions."""
        heavy = _ex("Squat", _sets(5, 5, 100, 9.0), ["quads"])
        light = _ex("Squat", _sets(3, 20, 42, 6.0), ["quads"])
        assert abs(_exercise_volume(heavy) - _exercise_volume(light)) < 50

        heavy_stim = weekly_muscle_summary([_session(1, [heavy])])["quads"]
        light_stim = weekly_muscle_summary([_session(1, [light])])["quads"]
        assert heavy_stim["stimulating_reps"] > light_stim["stimulating_reps"]

    def test_rpe_drift_detects_identical_work_getting_harder(self):
        """The signal tonnage structurally cannot see: nothing about the work
        changed, so volume load is flat, but effort rose."""
        early = _session(21, [_ex("Squat", _sets(3, 5, 100, 7.0), ["quads"])])
        late = _session(2, [_ex("Squat", _sets(3, 5, 100, 9.0), ["quads"])])
        assert _exercise_volume(early["exercises"][0]) == _exercise_volume(late["exercises"][0])

        drift = rpe_drift([early, late])
        assert drift and drift[0]["exercise"] == "Squat"
        assert drift[0]["direction"] == "harder"
        assert drift[0]["rpe_change"] == 2.0

    def test_drift_ignores_different_prescriptions(self):
        """Comparing 100kg to 120kg says nothing about fatigue."""
        a = _session(14, [_ex("Squat", _sets(3, 5, 100, 7.0), ["quads"])])
        b = _session(2, [_ex("Squat", _sets(3, 5, 120, 9.0), ["quads"])])
        assert rpe_drift([a, b]) == []

    def test_drift_reports_improvement_too(self):
        early = _session(21, [_ex("Bench", _sets(3, 5, 80, 9.0), ["chest"])])
        late = _session(2, [_ex("Bench", _sets(3, 5, 80, 7.0), ["chest"])])
        assert rpe_drift([early, late])[0]["direction"] == "easier"


class TestVolumeLandmarks:
    def test_below_mev(self):
        s = weekly_muscle_summary([_session(1, [_ex("Curl", _sets(3, 10, 20, 8.0), ["biceps"])])])
        assert s["biceps"]["status"] == "below MEV"

    def test_above_mav(self):
        many = [_session(d, [_ex("Squat", _sets(7, 8, 80, 8.0), ["quads"])]) for d in (1, 3, 5)]
        assert weekly_muscle_summary(many)["quads"]["hard_sets"] > MAV_SETS
        assert weekly_muscle_summary(many)["quads"]["status"] == "above MAV"

    def test_window_excludes_older_sessions(self):
        old = _session(30, [_ex("Squat", _sets(5, 5, 100, 9.0), ["quads"])])
        assert weekly_muscle_summary([old]) == {}


class TestFailureExposure:
    def test_all_sets_to_failure_flags_high(self):
        s = [_session(1, [_ex("Squat", _sets(5, 5, 100, 10.0), ["quads"])])]
        exposure = failure_exposure(s)
        assert exposure["share"] == 1.0
        assert exposure["flag"] == "high"

    def test_submaximal_training_flags_low(self):
        s = [_session(1, [_ex("Squat", _sets(5, 5, 80, 7.0), ["quads"])])]
        exposure = failure_exposure(s)
        assert exposure["near_failure_sets"] == 0
        assert exposure["flag"] == "low"

    def test_threshold_boundary(self):
        s = [_session(1, [_ex("Squat", _sets(1, 5, 100, 10 - FAILURE_RIR_THRESHOLD), ["quads"])])]
        assert failure_exposure(s)["near_failure_sets"] == 1


class TestReadinessSignals:
    def test_bundles_every_signal(self):
        s = [_session(1, [_ex("Squat", _sets(4, 5, 100, 9.0), ["quads"])])]
        out = readiness_signals(s)
        assert set(out) == {"muscle_summary", "failure_exposure", "rpe_drift", "drift_flag"}

    def test_handles_empty_history(self):
        out = readiness_signals([])
        assert out["muscle_summary"] == {}
        assert out["failure_exposure"]["total_sets"] == 0
