"""Athlete profile."""
import json
from pathlib import Path

import streamlit as st

from ui import athletes
from ui.components import footer, page_head


def render() -> None:
    left, right = st.columns([2.2, 1], vertical_alignment="bottom")
    with left:
        page_head("Profile", "Goals, experience and equipment shape every plan the coach writes.")
    with right:
        user_id = athletes.athlete_picker("profile")

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
    footer()
