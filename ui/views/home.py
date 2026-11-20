"""Landing page."""
from html import escape

import streamlit as st

from ui import athletes
from ui.components import chip, footer, html, section


def render(coach_page) -> None:
    html(
        '<section class="ac-hero"><div class="ac-hero-grid"><div>'
        '<div class="ac-eyebrow">Multi-agent training coach</div>'
        '<h1 class="ac-h1">Training plans that adapt to how you <em>actually</em> feel.</h1>'
        '<p class="ac-lede">AdaptiveCoach reads your logged sessions, works out how fatigued you '
        "really are, and writes next week's plan — then a critic agent checks it against "
        "peer-reviewed research before you ever see it. Tell it your knee hurts and it rewrites "
        "the week.</p>"
        "</div>"
        '<div class="ac-hero-panel">'
        + "".join(
            f'<div class="ac-flow-step"><div class="ac-flow-dot">{i}</div>'
            f"<div><b>{escape(t)}</b><span>{escape(d)}</span></div></div>"
            for i, (t, d) in enumerate(
                [
                    ("Fatigue analyst", "ACWR in code, effort trends judged by the model"),
                    ("Supervisor", "Decides: new plan, adapt this week, or explain"),
                    ("Planner + writer", "Pull evidence, set the load, write seven days"),
                    ("Critic", "Scores safety & evidence — sends weak plans back"),
                ],
                start=1,
            )
        )
        + "</div></div></section>"
    )

    c1, c2, _ = st.columns([1.1, 1.1, 3])
    with c1:
        if st.button("Open the coach", type="primary", use_container_width=True):
            st.switch_page(coach_page)
    with c2:
        st.link_button(
            "View on GitHub", "https://github.com/Ameya-J16/adaptive-coach", use_container_width=True
        )

    html(
        '<div class="ac-stats" style="margin-top:1.6rem">'
        + "".join(
            f'<div class="ac-stat"><div class="ac-stat-num">{n}</div>'
            f'<div class="ac-stat-label">{escape(lbl)}</div></div>'
            for n, lbl in [
                ("6", "specialist agents"),
                ("12", "tools they can call"),
                ("44", "peer-reviewed papers"),
                ("100%", "critic accuracy on golden set"),
                ("100+", "automated tests in CI"),
            ]
        )
        + "</div>"
    )

    section("How it works", "Every step is either code you can test or an agent you can trace.")
    html(
        '<div class="ac-grid-3">'
        + "".join(
            f'<div class="ac-card"><div class="ac-card-icon">{icon}</div>'
            f"<h3>{escape(t)}</h3><p>{escape(d)}</p></div>"
            for icon, t, d in [
                ("01", "Reads your real training",
                 "Every logged set and RPE feeds the analysis — not a questionnaire."),
                ("02", "Knows when to back off",
                 "Acute vs chronic load, RPE drift at matched weights, failure exposure and "
                 "per-muscle volume decide push, hold or deload."),
                ("03", "Grounded in research",
                 "Agents search 44 open-licence sports-science papers and cite them; the critic "
                 "checks claims against the same corpus."),
                ("04", "Safety rules the AI can't override",
                 "Back-to-back heavy days, fake deloads and overreach under red fatigue are caught "
                 "in code and force a rewrite."),
                ("05", "Adapts on request",
                 "“My knee hurts” or “I can't train Thursday” — the adaptation "
                 "agent edits the saved plan, and a safety gate rejects anything it breaks."),
                ("06", "Explains itself",
                 "Ask why you're deloading and a read-only agent investigates your data before "
                 "answering."),
            ]
        )
        + "</div>"
    )

    html('<hr class="ac-divider">')
    section("Try a demo athlete", "Each one is seeded with five weeks of realistic training.")
    ids = list(athletes.PERSONAS)
    cols = None
    for i, uid in enumerate(ids):
        if i % 3 == 0:  # one row of three at a time, so cards line up
            cols = st.columns(3)
        snap = athletes.snapshot(uid)
        zone = snap["zone"]
        acwr = f"ACWR {snap['acwr']:.2f}" if snap["acwr"] is not None else "No baseline yet"
        with cols[i % 3]:
            with st.container(border=True):
                html(
                    f'<div style="display:flex;justify-content:space-between;align-items:center;gap:8px">'
                    f'<b style="font-family:var(--ac-heading);font-size:1.02rem">{escape(athletes.label(uid))}</b>'
                    f'{chip(zone.upper(), zone)}</div>'
                    f'<p style="color:var(--ac-muted);font-size:0.88rem;margin:0.45rem 0 0.2rem">'
                    f"{escape(athletes.PERSONAS[uid][1])}</p>"
                    f'<p style="color:var(--ac-muted);font-size:0.8rem;margin:0">{escape(acwr)} · '
                    f'{snap["sessions_total"]} sessions logged</p>'
                )
                if st.button("Coach this athlete", key=f"home_pick_{uid}", use_container_width=True):
                    athletes.set_athlete(uid)
                    st.switch_page(coach_page)

    footer()
