"""Who the site is coaching: demo personas, the athlete switcher and a quick,
deterministic fatigue snapshot (no LLM call) for headers and cards."""
from datetime import date

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
