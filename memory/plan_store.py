"""Persistence for the current weekly plan and its revision history.

The planning pipeline produces a plan; the adaptation agent *edits* one. That
only works if a plan outlives the request that created it, so it lives here
rather than in Streamlit session state.

Every mutation is appended to a change log. That log is the audit trail of what
the agent did and why — useful for debugging, and it is the thing to show
someone who asks "what does the agent actually decide?"
"""
import json
from datetime import datetime
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data" / "plans"

DAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]


def _plan_file(user_id: str) -> Path:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    return DATA_DIR / f"{user_id}.json"


def save_plan(
    user_id: str,
    plan: dict,
    source: str = "planner",
    fatigue_report: dict | None = None,
    progression_decision: dict | None = None,
) -> None:
    """Persist a freshly generated plan, resetting the change log.

    The fatigue report and progression decision are stored with it because the
    adaptation and explain paths do not re-run the planning pipeline — without
    this they would judge the plan against an empty decision and misread its
    intent (e.g. treating an increase week as a maintain week).
    """
    record = {
        "plan": plan,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "source": source,
        "fatigue_report": fatigue_report or {},
        "progression_decision": progression_decision or {},
        "changes": [],
    }
    with open(_plan_file(user_id), "w") as f:
        json.dump(record, f, indent=2, default=str)


def load_context(user_id: str) -> tuple[dict, dict]:
    """Return (fatigue_report, progression_decision) the stored plan was built under."""
    record = load_record(user_id)
    if not record:
        return {}, {}
    return record.get("fatigue_report", {}), record.get("progression_decision", {})


def load_record(user_id: str) -> dict | None:
    path = _plan_file(user_id)
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def load_plan(user_id: str) -> dict | None:
    record = load_record(user_id)
    return record["plan"] if record else None


def has_plan(user_id: str) -> bool:
    return load_record(user_id) is not None


def update_plan(user_id: str, plan: dict, change: str) -> None:
    """Write a modified plan back and record what changed."""
    record = load_record(user_id)
    if record is None:
        raise ValueError(f"no stored plan for {user_id}")
    record["plan"] = plan
    record.setdefault("changes", []).append({
        "at": datetime.now().isoformat(timespec="seconds"),
        "change": change,
    })
    with open(_plan_file(user_id), "w") as f:
        json.dump(record, f, indent=2, default=str)


def get_changes(user_id: str) -> list[dict]:
    record = load_record(user_id)
    return record.get("changes", []) if record else []
