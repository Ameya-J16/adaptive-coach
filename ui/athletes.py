"""Who the site is coaching: demo personas, the athlete switcher, a quick
deterministic fatigue snapshot (no LLM call), and the public-demo guards.

DEMO_MODE (set it in the hosted app's secrets) turns on two protections for a
public deploy:
- each visitor works on a private copy of the demo athletes, so one person's
  "my knee hurts" edit never shows up in someone else's plan;
- each browser session gets MAX_AI_RUNS_PER_SESSION model runs (default 5),
  so an open "Generate" button can't run up the API bill.
Locally it stays off and everything behaves as a single-user app.

OWNER_TOKEN unlocks owner mode on a public deploy: open the site once with
?owner=<token> and that browser session gets unlimited runs and the links to
its Langfuse traces, which visitors never see.
"""
import os
import secrets
import shutil
from datetime import date
from pathlib import Path

import streamlit as st

PERSONAS = {
    "red_overreach": ("Overreached lifter", "Load spiked and RPE keeps climbing at the same weights."),
    "amber_caution": ("Lifter in the caution zone", "Training is running hot; one more push could tip it."),
    "green_progress": ("Fresh lifter", "Steady load, flat effort — ready to progress."),
    "fat_loss_cut": ("Fat-loss cut", "Dumbbell training while eating in a deficit."),
    "endurance_runner": ("Endurance runner", "Running intervals plus light accessory lifting."),
    "default_user": ("Default athlete", "A steady intermediate lifter."),
    "fresh_user": ("Brand-new user", "No history yet — the coach has no baseline."),
}
CUSTOM = "__custom__"
ZONE_TEXT = {"green": "Green — ready", "amber": "Amber — caution", "red": "Red — overreached"}


def label(user_id: str) -> str:
    return PERSONAS.get(user_id, (user_id.replace("_", " ").title(), ""))[0]


def current_athlete() -> str:
    return st.session_state.get("athlete", "red_overreach")


def set_athlete(user_id: str) -> None:
    st.session_state["athlete"] = user_id


def athlete_picker(key: str) -> str:
    """Selectbox of demo personas plus a free-text custom id. Returns the id."""
    current = current_athlete()
    options = list(PERSONAS) + [CUSTOM]
    index = options.index(current) if current in PERSONAS else len(options) - 1
    choice = st.selectbox(
        "Athlete",
        options,
        index=index,
        format_func=lambda u: "Your own ID…" if u == CUSTOM else f"{label(u)}  ·  {u}",
        key=f"{key}_athlete_select",
    )
    if choice == CUSTOM:
        typed = st.text_input(
            "Your athlete ID",
            value="" if current in PERSONAS else current,
            placeholder="e.g. ameya",
            key=f"{key}_athlete_custom",
        ).strip()
        user_id = typed or current
    else:
        user_id = choice
    set_athlete(user_id)
    return user_id


@st.cache_data(ttl=60, show_spinner=False)
def snapshot(user_id: str) -> dict:
    """ACWR zone and recent activity, computed in code — instant and free."""
    from graph.nodes.fatigue_analyst import _classify_fatigue, _compute_acwr
    from memory.workout_store import get_all_sessions, get_current_week, get_history

    acwr = _compute_acwr(get_history(user_id, weeks=5), get_current_week(user_id))
    zone, status = _classify_fatigue(acwr)
    sessions = get_all_sessions(user_id)
    last = max((s.get("date", "") for s in sessions), default="")
    days_since = None
    if last:
        try:
            days_since = (date.today() - date.fromisoformat(last)).days
        except ValueError:
            pass
    return {
        "acwr": acwr,
        "zone": zone,
        "status": status,
        "sessions_total": len(sessions),
        "sessions_week": len(get_current_week(user_id)),
        "last_session": last,
        "days_since": days_since,
    }


# ── Public-demo guards ──────────────────────────────────────────────────────

def demo_mode() -> bool:
    return os.getenv("DEMO_MODE", "").strip().lower() in {"1", "true", "yes", "on"}


def max_runs() -> int:
    try:
        return max(0, int(os.getenv("MAX_AI_RUNS_PER_SESSION", "5")))
    except ValueError:
        return 5


def is_owner() -> bool:
    """True locally, or on a public deploy once ?owner=<OWNER_TOKEN> was used."""
    if not demo_mode():
        return True
    if st.session_state.get("is_owner"):
        return True
    token = os.getenv("OWNER_TOKEN", "").strip()
    supplied = st.query_params.get("owner", "")
    if token and secrets.compare_digest(str(supplied), token):
        st.session_state["is_owner"] = True
        return True
    return False


def runs_left() -> int | None:
    """Model runs this session may still make; None means unlimited."""
    if not demo_mode() or is_owner():
        return None
    return max(0, max_runs() - st.session_state.get("ai_runs", 0))


def count_run() -> None:
    st.session_state["ai_runs"] = st.session_state.get("ai_runs", 0) + 1


def _copy_if_missing(src: Path, dst: Path) -> None:
    if src.exists() and not dst.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)


def data_id(key: str) -> str:
    """The storage id behind an athlete. In demo mode a persona maps to this
    visitor's private copy, created from the seeded persona on first use."""
    if not demo_mode() or key not in PERSONAS:
        return key
    visitor = st.session_state.setdefault("visitor_id", secrets.token_hex(4))
    private = f"{key}--{visitor}"
    from memory.workout_store import DATA_DIR as WORKOUTS_DIR

    _copy_if_missing(WORKOUTS_DIR / f"{key}.json", WORKOUTS_DIR / f"{private}.json")
    profiles = Path("data") / "profiles"  # same location main._load_profile reads
    _copy_if_missing(profiles / f"{key}.json", profiles / f"{private}.json")
    return private
