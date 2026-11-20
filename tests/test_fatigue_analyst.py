"""Tests for ACWR computation and fatigue level classification.

ACWR uses rolling windows: acute = last 7 days, chronic = the 28 days before
that, expressed as a weekly average. The windows are adjacent and never
overlap, so a session cannot count as both acute and chronic load.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from datetime import datetime, timedelta

from graph.nodes.fatigue_analyst import (
    ACUTE_DAYS,
    CHRONIC_DAYS,
    _classify_fatigue,
    _compute_acwr,
)

TODAY = datetime.now().date()


def _session(days_ago: int, volume: float) -> dict:
    """One session `days_ago` before today carrying exactly `volume` kg."""
    return {
        "date": (TODAY - timedelta(days=days_ago)).isoformat(),
        "exercises": [{"sets": 1, "reps": 1, "weight_kg": volume}],
        "avg_rpe": 7.5,
        "session_type": "strength",
    }


def _chronic(weekly_volume: float) -> list[dict]:
    """4 weeks of training inside the chronic window, one session per week."""
    # Chronic window is days 7..34 back; place sessions at 10, 17, 24, 31.
    return [_session(d, weekly_volume) for d in (10, 17, 24, 31)]


def _acute(total_volume: float, sessions: int = 3) -> list[dict]:
    """Training inside the last 7 days, split evenly across `sessions`."""
    per = total_volume / sessions
    return [_session(d, per) for d in range(sessions)]


class TestRollingWindows:
    def test_steady_training_gives_acwr_near_one(self):
        acwr = _compute_acwr(_chronic(7500) + _acute(7500), [])
        assert 0.95 <= acwr <= 1.05, f"steady load should sit near 1.0, got {acwr}"

    def test_load_spike_raises_acwr(self):
        acwr = _compute_acwr(_chronic(7500) + _acute(13000), [])
        assert acwr > 1.5
        assert _classify_fatigue(acwr)[0] == "red"

    def test_light_week_lowers_acwr(self):
        acwr = _compute_acwr(_chronic(7500) + _acute(3000), [])
        assert acwr < 0.8

    def test_windows_do_not_overlap(self):
        """A session on the acute/chronic boundary must count once, as acute."""
        boundary = _session(ACUTE_DAYS - 1, 5000)  # newest day still in acute
        just_outside = _session(ACUTE_DAYS, 5000)  # first chronic day

        acute_only = _compute_acwr([just_outside] + [boundary], [])
        # chronic = 5000 over 28 days -> weekly 1250; acute = 5000 -> 4.0
        assert acute_only == 4.0

    def test_sessions_older_than_chronic_window_are_ignored(self):
        ancient = _session(CHRONIC_DAYS + ACUTE_DAYS + 10, 999999)
        with_ancient = _compute_acwr(_chronic(7500) + _acute(7500) + [ancient], [])
        without = _compute_acwr(_chronic(7500) + _acute(7500), [])
        assert with_ancient == without

    def test_duplicate_sessions_counted_once(self):
        """history and current_week_log overlap in real runs — dedupe by value."""
        acute = _acute(7500)
        both = _compute_acwr(_chronic(7500) + acute, acute)
        once = _compute_acwr(_chronic(7500) + acute, [])
        assert both == once


class TestDayOfWeekInvariance:
    """Regression test for the partial-week bias.

    The old calendar-week acute window compared Monday..today against complete
    weeks, so the same training pattern read 0.51 on Monday and 1.22 on Friday.
    With rolling windows an athlete training at a constant daily rate must read
    the same ACWR no matter which day they check.
    """

    def test_constant_daily_training_is_stable_across_weekdays(self):
        # Train every single day at the same volume, covering both windows.
        sessions = [_session(d, 1000) for d in range(CHRONIC_DAYS + ACUTE_DAYS)]
        acwr = _compute_acwr(sessions, [])
        assert 0.95 <= acwr <= 1.05, (
            f"constant daily training should read ~1.0 on any weekday, got {acwr}"
        )

    def test_partial_week_does_not_deflate_acwr(self):
        """Two sessions in the last 7 days read the same whether 'today' is
        early or late in the calendar week, because the window is rolling."""
        sessions = _chronic(4000) + [_session(1, 2000), _session(3, 2000)]
        acwr = _compute_acwr(sessions, [])
        # chronic weekly = 4000, acute = 4000 -> 1.0, independent of weekday
        assert acwr == 1.0


class TestNoBaseline:
    def test_empty_history_returns_none(self):
        assert _compute_acwr([], []) is None

    def test_only_recent_training_has_no_chronic_baseline(self):
        """Someone who started training this week has nothing to compare to."""
        assert _compute_acwr(_acute(5000), []) is None

    def test_no_baseline_defaults_to_amber(self):
        level, status = _classify_fatigue(None)
        assert level == "amber"
        assert status == "no_baseline"


class TestFatigueClassification:
    def test_thresholds(self):
        assert _classify_fatigue(1.6) == ("red", "computed")
        assert _classify_fatigue(1.4) == ("amber", "computed")
        assert _classify_fatigue(1.0) == ("green", "computed")
        assert _classify_fatigue(0.5) == ("green", "computed")
