"""
Streamlit UI for AdaptiveCoach.
Run with: streamlit run app.py
"""
import json
import os
from datetime import datetime
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(
    page_title="AdaptiveCoach",
    page_icon="🏋️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Sidebar: user identity ──────────────────────────────────────────────────
with st.sidebar:
    st.title("🏋️ AdaptiveCoach")
    st.caption("Multi-Agent Fitness Intelligence")
    st.divider()
    user_id = st.text_input("User ID", value="default_user", key="user_id_input")
    st.caption("Your plans and workouts are stored locally under this ID.")

# ── Page tabs ───────────────────────────────────────────────────────────────
tab_plan, tab_log, tab_history, tab_profile = st.tabs([
    "📅 Weekly Plan", "📝 Log Workout", "📊 History", "👤 Profile"
])


# ─────────────────────────────────────────────────────────────────────────────
# TAB 1: GENERATE WEEKLY PLAN
# ─────────────────────────────────────────────────────────────────────────────
with tab_plan:
    st.header("Adaptive Weekly Training Plan")
    st.write("The multi-agent pipeline analyses your history, assesses fatigue, and writes a grounded plan.")

    col1, col2 = st.columns([2, 1])
    with col1:
        generate_btn = st.button("Generate Adaptive Plan", type="primary", use_container_width=True)

    if generate_btn:
        from graph.graph import graph
        from main import _load_profile

        initial_state = {
            "user_id": user_id,
            "user_profile": _load_profile(user_id),
            "workout_history": [],
            "current_week_log": [],
            "fatigue_report": {},
            "retrieved_context": "",
            "progression_decision": {},
            "nutrition_targets": {},
            "weekly_plan": {},
            "critic_score": 0.0,
            "critic_feedback": "",
            "loop_count": 0,
            "messages": [],
        }

        with st.spinner("Running multi-agent pipeline (context → fatigue → planner → nutrition → writer → critic)..."):
            result = graph.invoke(initial_state)

        st.session_state["last_result"] = result

    if "last_result" in st.session_state:
        import pandas as pd
        result = st.session_state["last_result"]
        fr = result.get("fatigue_report", {})
        pd_res = result.get("progression_decision", {})
        plan = result.get("weekly_plan", {})
        nt = result.get("nutrition_targets", {})

        # ── Fatigue badge ──
        st.subheader("Fatigue Assessment")
        fatigue_level = fr.get("fatigue_level", "green")
        badge_color = {"green": "🟢", "amber": "🟡", "red": "🔴"}.get(fatigue_level, "⚪")
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.metric("Fatigue Level", f"{badge_color} {fatigue_level.upper()}")
        with col_b:
            st.metric("ACWR", f"{fr.get('acwr', 0):.2f}")
        with col_c:
            st.metric("Progression Action", pd_res.get("action", "N/A").upper())

        with st.expander("Fatigue Details"):
            st.write(f"**Dominant Pattern:** {fr.get('dominant_stress_pattern', 'N/A')}")
            st.write(f"**Recommendation:** {fr.get('recommendation_hint', 'N/A')}")
            st.write(f"**Rationale:** {pd_res.get('rationale', 'N/A')}")

        # ── Critic score ──
        st.subheader("Plan Quality Score")
        score = result.get("critic_score", 0.0)
        st.progress(score, text=f"Critic Score: {score:.2f} / 1.00")
        st.caption(f"Critic loops run: {result.get('loop_count', 0)}")
        if result.get("critic_feedback"):
            with st.expander("Critic Feedback"):
                st.write(result["critic_feedback"])

        # ── Weekly plan + daily nutrition ──
        st.subheader("7-Day Training Plan & Daily Nutrition")
        days = plan.get("days", {})
        day_order = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
        rest_types = {"rest", "deload"}
        training_day_nt = nt.get("training_day", {})
        rest_day_nt = nt.get("rest_day", {})

        for day in day_order:
            content = days.get(day, {})
            session_type = content.get("session_type", "rest")
            is_rest = session_type.lower() in rest_types
            icon = {"strength": "🏋️", "cardio": "🏃", "HIIT": "⚡", "rest": "😴", "deload": "🔄"}.get(session_type, "📋")
            day_nt = rest_day_nt if is_rest else training_day_nt

            with st.expander(f"{icon} {day.capitalize()} — {session_type.upper()}"):
                exercises = content.get("exercises", [])

                # Exercise table
                if exercises:
                    ex_data = [{
                        "Exercise": e.get("name", ""),
                        "Sets": e.get("sets", 0),
                        "Reps": e.get("reps", 0),
                        "Weight (kg)": e.get("weight_kg", 0),
                        "RPE Target": e.get("rpe_target", 0),
                        "Muscle Groups": ", ".join(e.get("muscle_groups", [])),
                    } for e in exercises]
                    st.dataframe(pd.DataFrame(ex_data), use_container_width=True, hide_index=True)
                else:
                    st.write("Rest day — prioritise sleep and nutrition.")

                if content.get("notes"):
                    st.info(content["notes"])

                # Daily nutrition for this day
                if day_nt:
                    st.markdown("**Nutrition Target for Today**")
                    nc1, nc2, nc3, nc4 = st.columns(4)
                    with nc1:
                        st.metric("Calories", f"{day_nt.get('calories', 0)} kcal")
                    with nc2:
                        st.metric("Protein", f"{day_nt.get('protein_g', 0)}g")
                    with nc3:
                        st.metric("Carbs", f"{day_nt.get('carbs_g', 0)}g")
                    with nc4:
                        st.metric("Fat", f"{day_nt.get('fat_g', 0)}g")

        if plan.get("overall_notes"):
            st.info(f"**Overall Notes:** {plan['overall_notes']}")

        # ── Nutrition timing notes ──
        if nt.get("timing_notes"):
            st.subheader("Nutrition Timing")
            st.info(nt["timing_notes"])

        # ── RAG grounding transparency ──
        retrieved = result.get("retrieved_context", "")
        if retrieved:
            with st.expander("📚 Sports Science Sources Used (RAG Grounding)"):
                st.markdown(retrieved)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 2: LOG WORKOUT (Hevy-style per-set logging)
# ─────────────────────────────────────────────────────────────────────────────
with tab_log:
    st.header("Log a Workout Session")

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

    st.divider()
    st.subheader("Exercises")

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
    if st.button("Log Workout", type="primary", use_container_width=True, key="log_submit_btn"):
        from memory.workout_store import log_workout
        valid_exercises = [e for e in st.session_state.log_exercises if e["name"].strip()]
        session = {
            "date": session_date.isoformat(),
            "exercises": valid_exercises,
            "avg_rpe": avg_rpe,
            "session_type": session_type,
            "notes": notes,
        }
        log_workout(user_id, session)
        total_vol = sum(
            s["reps"] * s["weight_kg"]
            for e in valid_exercises
            for s in e.get("sets_data", [])
        )
        st.success(f"Workout logged! {len(valid_exercises)} exercises | Total volume: {total_vol:.0f} kg")
        st.session_state.log_exercises = []
        st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# TAB 3: HISTORY
# ─────────────────────────────────────────────────────────────────────────────
with tab_history:
    st.header("Training History")

    from memory.workout_store import get_all_sessions
    sessions = get_all_sessions(user_id)

    if not sessions:
        st.info("No workouts logged yet. Head to 'Log Workout' to get started.")
    else:
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

        st.dataframe(df, use_container_width=True, hide_index=True)

        # ── Per-session drill-down ──
        st.subheader("Session Detail")
        session_labels = [f"{s.get('date')} — {s.get('session_type', '').upper()}" for s in sorted(sessions, key=lambda x: x.get("date",""), reverse=True)]
        if session_labels:
            selected_label = st.selectbox("Select a session to inspect", session_labels, key="history_session_select")
            selected_idx = session_labels.index(selected_label)
            sorted_sessions = sorted(sessions, key=lambda x: x.get("date",""), reverse=True)
            sel = sorted_sessions[selected_idx]

            for ex in sel.get("exercises", []):
                with st.container(border=True):
                    st.markdown(f"**{ex.get('name', 'Unknown')}**")
                    sets_data = ex.get("sets_data", [])
                    if sets_data:
                        set_rows = [{
                            "Set": s.get("set_number", i+1),
                            "Weight (kg)": s.get("weight_kg", 0),
                            "Reps": s.get("reps", 0),
                            "RPE": s.get("rpe", 0),
                            "Volume": f"{s.get('reps',0) * s.get('weight_kg',0):.0f} kg",
                        } for i, s in enumerate(sets_data)]
                        st.dataframe(pd.DataFrame(set_rows), use_container_width=True, hide_index=True)
                    else:
                        st.caption("No per-set data recorded.")

        if len(sessions) > 1:
            st.subheader("Weekly Volume Trend")
            volume_df = df[["Date", "Total Volume (kg)"]].copy()
            volume_df = volume_df.set_index("Date").resample("W").sum()
            st.line_chart(volume_df)

            st.subheader("RPE Trend")
            rpe_df = df[["Date", "Avg RPE"]].copy()
            rpe_df = rpe_df.set_index("Date").resample("W").mean()
            st.line_chart(rpe_df)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 4: PROFILE
# ─────────────────────────────────────────────────────────────────────────────
with tab_profile:
    st.header("User Profile")

    profile_path = Path("data") / "profiles" / f"{user_id}.json"
    profile_path.parent.mkdir(parents=True, exist_ok=True)

    existing = {}
    if profile_path.exists():
        with open(profile_path) as f:
            existing = json.load(f)

    with st.form("profile_form"):
        col1, col2 = st.columns(2)
        with col1:
            name = st.text_input("Name", value=existing.get("name", user_id))
            goals = st.selectbox(
                "Primary Goal",
                ["strength", "hypertrophy", "fat loss", "endurance"],
                index=["strength", "hypertrophy", "fat loss", "endurance"].index(
                    existing.get("goals", "strength")
                ) if existing.get("goals") in ["strength", "hypertrophy", "fat loss", "endurance"] else 0,
            )
            experience_level = st.selectbox(
                "Experience Level",
                ["beginner", "intermediate", "advanced"],
                index=["beginner", "intermediate", "advanced"].index(
                    existing.get("experience_level", "intermediate")
                ) if existing.get("experience_level") in ["beginner", "intermediate", "advanced"] else 1,
            )
        with col2:
            available_days = st.slider("Training Days Per Week", 1, 7, existing.get("available_days", 4))
            weight_kg = st.number_input("Body Weight (kg)", value=float(existing.get("weight_kg", 75)), step=0.5)
            height_cm = st.number_input("Height (cm)", value=int(existing.get("height_cm", 175)), step=1)
            age = st.number_input("Age", value=int(existing.get("age", 25)), min_value=16, max_value=80, step=1)

        equipment_options = ["barbell", "dumbbells", "rack", "cables", "machines", "bodyweight only", "kettlebells", "bands"]
        equipment = st.multiselect(
            "Available Equipment",
            equipment_options,
            default=existing.get("equipment", ["barbell", "dumbbells", "rack"]),
        )

        save_btn = st.form_submit_button("Save Profile", type="primary")

    if save_btn:
        profile_data = {
            "name": name,
            "goals": goals,
            "experience_level": experience_level,
            "available_days": available_days,
            "weight_kg": weight_kg,
            "height_cm": height_cm,
            "age": age,
            "equipment": equipment,
        }
        with open(profile_path, "w") as f:
            json.dump(profile_data, f, indent=2)
        st.success("Profile saved!")
