"""
AdaptiveCoach website (Streamlit).
Run with: streamlit run app.py

The pages live in ui/views/; this file sets up the theme, seeds demo data on
hosted deploys and wires the top navigation.
"""
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(
    page_title="AdaptiveCoach — training plans that adapt",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ── Demo data on hosted deploys ─────────────────────────────────────────────
# Hosted deploys (Streamlit Community Cloud, Docker) start with an empty data/
# folder and wipe it on restart. Seed the demo personas once per process when
# no workout data exists; never touches a local install that already has data.
@st.cache_resource(show_spinner="Loading demo athletes…")
def _seed_demo_data_if_empty() -> None:
    workouts_dir = Path(__file__).parent / "data" / "workouts"
    if workouts_dir.exists() and any(workouts_dir.glob("*.json")):
        return
    from scripts.seed_demo_data import seed
    seed()


_seed_demo_data_if_empty()

from ui import styles  # noqa: E402  (after set_page_config)
from ui.views import coach, history, home, log, profile  # noqa: E402

styles.inject()

# Check ?owner=<token> on whichever page the visitor lands on. The query
# string is dropped as soon as they navigate, so checking only on the Coach
# page missed owners who opened the link on Home.
from ui import athletes  # noqa: E402

athletes.is_owner()
_assets = Path(__file__).parent / "ui" / "assets"
st.logo(str(_assets / "logo.svg"), size="large")

coach_page = st.Page(coach.render, title="Coach", url_path="coach", icon=":material/bolt:")
pages = [
    st.Page(lambda: home.render(coach_page), title="Home", url_path="home",
            icon=":material/home:", default=True),
    coach_page,
    st.Page(log.render, title="Log workout", url_path="log", icon=":material/edit_note:"),
    st.Page(history.render, title="History", url_path="history", icon=":material/monitoring:"),
    st.Page(profile.render, title="Profile", url_path="profile", icon=":material/person:"),
]
st.navigation(pages, position="top").run()
