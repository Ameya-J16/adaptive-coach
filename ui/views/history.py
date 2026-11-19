"""Training history."""
from datetime import date, timedelta

import streamlit as st

from ui import athletes
from ui.components import empty_state, footer, kpis, page_head, section


def render() -> None:
    left, right = st.columns([2.2, 1], vertical_alignment="bottom")
    with left:
        page_head("History", "Everything you've logged, and how load and effort are trending.")
    with right:
        user_id = athletes.athlete_picker("history")

    from memory.workout_store import get_all_sessions

    sessions = get_all_sessions(user_id)
    if not sessions:
        empty_state("Nothing logged yet", "Log a workout and your history, trends and stats show up here.")
        footer()
        return

    import pandas as pd

    df = pd.DataFrame([{
        "Date": s.get("date"),
        "Type": s.get("session_type"),
        "Avg RPE": s.get("avg_rpe"),
        "Total Volume (kg)": s.get("total_volume", 0),
        "Exercises": len(s.get("exercises", [])),
        "Notes": s.get("notes", ""),
    } for s in sessions])
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date", ascending=False)

    cutoff = pd.Timestamp(date.today() - timedelta(days=28))
    recent = df[df["Date"] >= cutoff]
    snap = athletes.snapshot(user_id)
    kpis([
        {"label": "Sessions logged", "value": len(df), "note": f"{len(recent)} in the last 4 weeks"},
        {"label": "Volume, last 4 weeks", "value": f'{recent["Total Volume (kg)"].sum():,.0f} kg',
         "note": "Sets × reps × load"},
        {"label": "Average RPE", "value": f'{recent["Avg RPE"].mean():.1f}' if len(recent) else "—",
         "note": "Last 4 weeks"},
        {"label": "Fatigue zone", "value": snap["zone"].upper(), "tone": snap["zone"],
         "note": f'ACWR {snap["acwr"]:.2f}' if snap["acwr"] is not None else "No baseline yet"},
    ])

    if len(df) > 1:
        left, right = st.columns(2, gap="medium")
        with left:
            section("Weekly volume")
            vol = df[["Date", "Total Volume (kg)"]].set_index("Date").resample("W").sum()
            vol.index = vol.index.strftime("w/e %d %b")  # categorical axis: proper-width bars
            st.bar_chart(vol, height=240, sort=False)  # keep date order, not alphabetical
        with right:
            section("Weekly average RPE")
            rpe = df[["Date", "Avg RPE"]].set_index("Date").resample("W").mean()
            st.line_chart(rpe, height=240)

    section("All sessions")
    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Date": st.column_config.DateColumn(format="ddd D MMM YYYY"),
            "Total Volume (kg)": st.column_config.NumberColumn(format="%,.0f"),
            "Avg RPE": st.column_config.ProgressColumn(min_value=0, max_value=10, format="%.1f"),
        },
    )

    section("Session detail")
    ordered = sorted(sessions, key=lambda x: x.get("date", ""), reverse=True)
    labels = [f"{s.get('date')} — {str(s.get('session_type', '')).upper()}" for s in ordered]
    picked = st.selectbox("Session", labels, key="history_session_select", label_visibility="collapsed")
    sel = ordered[labels.index(picked)]
    cols = st.columns(2, gap="medium")
    for i, ex in enumerate(sel.get("exercises", [])):
        with cols[i % 2]:
            with st.container(border=True):
                st.markdown(f"**{ex.get('name', 'Exercise')}**")
                sets_data = ex.get("sets_data", [])
                if sets_data:
                    st.dataframe(
                        pd.DataFrame([{
                            "Set": s.get("set_number", j + 1),
                            "Weight (kg)": s.get("weight_kg", 0),
                            "Reps": s.get("reps", 0),
                            "RPE": s.get("rpe", 0),
                        } for j, s in enumerate(sets_data)]),
                        use_container_width=True,
                        hide_index=True,
                    )
                else:
                    st.caption(
                        f"{ex.get('sets', '?')}×{ex.get('reps', '?')} @ {ex.get('weight_kg', '?')} kg"
                    )
    footer()
