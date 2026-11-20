"""Session memory round-trips through a real SQLite file. The context loader
calls this on every run, so a broken constructor fails every request."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import memory.session_memory as sm


def test_messages_round_trip(tmp_path, monkeypatch):
    monkeypatch.setattr(sm, "DB_PATH", tmp_path / "sessions.db")
    assert sm.get_messages_as_dicts("athlete") == []
    sm.add_message("athlete", "human", "my knee hurts")
    sm.add_message("athlete", "ai", "swapped squats for goblet squats")
    assert sm.get_messages_as_dicts("athlete") == [
        {"role": "human", "content": "my knee hurts"},
        {"role": "ai", "content": "swapped squats for goblet squats"},
    ]
    assert sm.get_messages_as_dicts("someone_else") == []
