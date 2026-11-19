"""Log a workout, set by set."""
from datetime import datetime

import streamlit as st

from ui import athletes
from ui.components import footer, page_head, section


def render() -> None:
    left, right = st.columns([2.2, 1], vertical_alignment="bottom")
    with left:
        page_head("Log a workout", "Record every set — weight, reps and RPE feed the fatigue analysis.")
    with right:
        user_id = athletes.athlete_picker("log")

    # ── Session-level state init ──
    if "log_exercises" not in st.session_state:
        st.session_state.log_exercises = []
    if "log_submitted" not in st.session_state:
        st.session_state.log_submitted = False

    # ── Session meta (outside form so Add Exercise button works) ──
    col1, col2 = st.columns(2)
    with col1:
        session_type = st.selectbox("Session Type", ["strength", "cardio", "HIIT", "rest", "deload"], key="log_session_type")
        session_date = st.date_input("Date", value=datetime.now().date(), key="log_date")
    with col2:
        avg_rpe = st.slider("Average Session RPE", 1.0, 10.0, 7.0, 0.5, key="log_avg_rpe")
        notes = st.text_area("Session Notes", height=80, key="log_notes")

    section("Exercises", "Add each movement, then its sets.")

    # ── Add exercise button ──
    if st.button("＋ Add Exercise", key="add_exercise_btn"):
        st.session_state.log_exercises.append({
            "name": "",
            "sets_data": [{"set_number": 1, "reps": 8, "weight_kg": 60.0, "rpe": 7.0}],
            "muscle_groups": [],
            "notes": "",
        })
        st.rerun()

    # ── Render each exercise ──
    exercises_to_delete = []
    for ex_idx, ex in enumerate(st.session_state.log_exercises):
        with st.container(border=True):
            ecol1, ecol2 = st.columns([4, 1])
            with ecol1:
                ex["name"] = st.text_input(
                    "Exercise name",
                    value=ex["name"],
                    placeholder="e.g. Squat, Bench Press, Deadlift",
                    key=f"ex_name_{ex_idx}",
                    label_visibility="collapsed",
                )
            with ecol2:
                if st.button("Remove", key=f"del_ex_{ex_idx}", type="secondary"):
                    exercises_to_delete.append(ex_idx)

            # Set header row
            h1, h2, h3, h4, h5 = st.columns([1, 2, 2, 2, 1])
            h1.caption("Set")
            h2.caption("Weight (kg)")
            h3.caption("Reps")
            h4.caption("RPE")
            h5.caption("")

            # Render each set
            sets_to_delete = []
            for s_idx, s in enumerate(ex["sets_data"]):
                sc1, sc2, sc3, sc4, sc5 = st.columns([1, 2, 2, 2, 1])
                with sc1:
                    st.markdown(f"**{s_idx + 1}**")
                with sc2:
                    s["weight_kg"] = st.number_input(
                        "kg", value=float(s["weight_kg"]), min_value=0.0, step=2.5,
                        key=f"set_weight_{ex_idx}_{s_idx}", label_visibility="collapsed"
                    )
                with sc3:
                    s["reps"] = st.number_input(
                        "reps", value=int(s["reps"]), min_value=1, max_value=100,
                        key=f"set_reps_{ex_idx}_{s_idx}", label_visibility="collapsed"
                    )
                with sc4:
                    s["rpe"] = st.number_input(
                        "rpe", value=float(s["rpe"]), min_value=1.0, max_value=10.0, step=0.5,
                        key=f"set_rpe_{ex_idx}_{s_idx}", label_visibility="collapsed"
                    )
                with sc5:
                    if st.button("✕", key=f"del_set_{ex_idx}_{s_idx}"):
                        sets_to_delete.append(s_idx)

            # Remove deleted sets
            for s_idx in reversed(sets_to_delete):
                ex["sets_data"].pop(s_idx)
            if sets_to_delete:
                st.rerun()

            # Renumber sets
            for i, s in enumerate(ex["sets_data"]):
                s["set_number"] = i + 1

            # Add set button
            if st.button("＋ Add Set", key=f"add_set_{ex_idx}"):
                last = ex["sets_data"][-1] if ex["sets_data"] else {"weight_kg": 60.0, "rpe": 7.0}
                ex["sets_data"].append({
                    "set_number": len(ex["sets_data"]) + 1,
                    "reps": 8,
                    "weight_kg": last["weight_kg"],
                    "rpe": last["rpe"],
                })
                st.rerun()

            # Per-exercise volume
            vol = sum(s["reps"] * s["weight_kg"] for s in ex["sets_data"])
            st.caption(f"Volume: {vol:.0f} kg · {len(ex['sets_data'])} sets")

    # Remove deleted exercises
    for ex_idx in reversed(exercises_to_delete):
        st.session_state.log_exercises.pop(ex_idx)
    if exercises_to_delete:
        st.rerun()

    # ── Log Workout button ──
    st.divider()

    # Surface the result of the previous run's save (st.rerun() wipes messages
    # written before it, so the confirmation has to survive in session_state).
    if st.session_state.pop("log_success", None):
        st.success(st.session_state.pop("log_success_msg", "Workout logged."))

    if st.button("Log Workout", type="primary", use_container_width=True, key="log_submit_btn"):
        from memory.workout_store import log_workout

        valid_exercises = [
            e for e in st.session_state.log_exercises
            if e["name"].strip() and e.get("sets_data")
        ]

        if not valid_exercises:
            st.error("Nothing to log — add at least one named exercise with a set.")
        else:
            session = {
                "date": session_date.isoformat(),
                "exercises": valid_exercises,
                "avg_rpe": avg_rpe,
                "session_type": session_type,
                "notes": notes,
            }
            log_workout(user_id, session)
            athletes.snapshot.clear()  # fatigue zone and session counts changed
            total_vol = sum(
                s["reps"] * s["weight_kg"]
                for e in valid_exercises
                for s in e.get("sets_data", [])
            )
            st.session_state.log_success = True
            st.session_state.log_success_msg = (
                f"Workout logged for {session_date.isoformat()} — "
                f"{len(valid_exercises)} exercises | Total volume: {total_vol:.0f} kg. "
                f"See History for the trend."
            )
            st.session_state.log_exercises = []
            st.rerun()

    footer()
