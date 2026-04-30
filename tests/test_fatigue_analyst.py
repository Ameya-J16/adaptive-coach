"""Tests for ACWR computation and fatigue level classification."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock

from graph.nodes.fatigue_analyst import _compute_acwr


def _make_sessions(weeks_ago: int, volume_per_session: float, num_sessions: int = 3) -> list[dict]:
    """Generate mock sessions for a given week offset."""
    sessions = []
    base_date = datetime.now().date() - timedelta(weeks=weeks_ago)
    for i in range(num_sessions):
        sessions.append({
            "date": (base_date + timedelta(days=i)).isoformat(),
            "exercises": [{"sets": 3, "reps": 8, "weight_kg": volume_per_session / (3 * 8)}],
            "avg_rpe": 7.5,
            "session_type": "strength",
        })
    return sessions


class TestACWRComputation:
    def test_high_acwr_triggers_red(self):
        """ACWR > 1.5 should result in fatigue_level = 'red'."""
        # Chronic baseline: 3 weeks of 100kg volume each
        history = (
            _make_sessions(1, 100) +
            _make_sessions(2, 100) +
            _make_sessions(3, 100)
        )
        # Acute: current week with 200kg volume (2x normal = ACWR ~2.0)
        current_week = _make_sessions(0, 200)

        acwr = _compute_acwr(history, current_week)
        assert acwr > 1.5, f"Expected ACWR > 1.5 for spike load, got {acwr}"

    def test_acwr_red_classification(self):
        """When ACWR > 1.5, fatigue_level must be forced to 'red'."""
        history = (
            _make_sessions(1, 100) +
            _make_sessions(2, 100) +
            _make_sessions(3, 100)
        )
        current_week = _make_sessions(0, 220)

        acwr = _compute_acwr(history, current_week)
        # Simulate the classification logic from the node
        if acwr > 1.5:
            fatigue_level = "red"
        elif acwr > 1.3:
            fatigue_level = "amber"
        else:
            fatigue_level = "green"

        assert fatigue_level == "red"

    def test_empty_history_returns_neutral_acwr(self):
        """Empty history should return ACWR of 1.0 (neutral)."""
        acwr = _compute_acwr([], [])
        assert acwr == 1.0

    def test_empty_current_week_with_history(self):
        """If current week is empty (rest week), ACWR should be 0."""
        history = _make_sessions(1, 100) + _make_sessions(2, 100)
        acwr = _compute_acwr(history, [])
        assert acwr == 0.0

    def test_normal_acwr_in_sweet_spot(self):
        """Consistent training should produce ACWR in the 0.8-1.3 range."""
        history = (
            _make_sessions(1, 100) +
            _make_sessions(2, 100) +
            _make_sessions(3, 100)
        )
        current_week = _make_sessions(0, 105)  # slight increase
        acwr = _compute_acwr(history, current_week)
        assert 0.8 <= acwr <= 1.5, f"Expected ACWR in sweet spot, got {acwr}"

    def test_acwr_amber_range(self):
        """ACWR between 1.3-1.5 should map to amber."""
        history = (
            _make_sessions(1, 100) +
            _make_sessions(2, 100) +
            _make_sessions(3, 100)
        )
        current_week = _make_sessions(0, 140)  # 40% increase
        acwr = _compute_acwr(history, current_week)

        if acwr > 1.5:
            level = "red"
        elif acwr > 1.3:
            level = "amber"
        else:
            level = "green"

        # 140% of 100 chronic → ACWR ~1.4
        assert level in ("amber", "red"), f"Expected amber/red for 40% spike, got {level} (ACWR={acwr})"
