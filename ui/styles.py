"""Site-wide CSS. Colours mirror .streamlit/config.toml, which themes the
built-in widgets; these variables theme the custom HTML components."""
import streamlit as st

CSS = """
<style>
:root {
  --ac-bg: #0A0D12;
  --ac-surface: #11161D;
  --ac-surface-2: #161C25;
  --ac-border: #232B36;
  --ac-text: #E7ECF2;
  --ac-muted: #8A96A6;
  --ac-accent: #C8F135;
  --ac-accent-ink: #0A0D12;
  --ac-green: #3DDC97;
  --ac-amber: #F5B841;
  --ac-red: #FF6B6B;
  --ac-blue: #7CC4FF;
  --ac-radius: 14px;
  --ac-heading: "Space Grotesk", Inter, system-ui, sans-serif;
}

/* ── Page frame ─────────────────────────────────────────────── */
.block-container, [data-testid="stMainBlockContainer"] {
  padding-top: 5.5rem !important; padding-bottom: 4rem; max-width: 1180px;
}
header[data-testid="stHeader"] {
  background: rgba(10, 13, 18, 0.82);
  backdrop-filter: blur(10px);
  border-bottom: 1px solid var(--ac-border);
}
[data-testid="stDecoration"] { display: none; }
footer { visibility: hidden; }

/* Primary buttons: dark ink on the lime accent for contrast. */
button[kind="primary"], button[data-testid="stBaseButton-primary"],
[data-testid="stFormSubmitButton"] button[kind="primaryFormSubmit"] {
  color: var(--ac-accent-ink) !important;
  font-weight: 600;
}
button[kind="primary"] p, button[data-testid="stBaseButton-primary"] p { color: var(--ac-accent-ink) !important; }
button[kind="primary"]:disabled, button[data-testid="stBaseButton-primary"]:disabled,
button[kind="primary"]:disabled p, button[data-testid="stBaseButton-primary"]:disabled p {
  color: var(--ac-muted) !important;
}

/* ── Typography helpers ─────────────────────────────────────── */
.ac-eyebrow {
  font: 600 0.78rem/1 Inter, sans-serif; letter-spacing: 0.14em; text-transform: uppercase;
  color: var(--ac-accent); margin-bottom: 0.9rem;
}
.ac-h1 {
  font-family: var(--ac-heading); font-weight: 700; letter-spacing: -0.02em;
  font-size: clamp(2.2rem, 5vw, 3.6rem); line-height: 1.04; margin: 0 0 1rem; color: var(--ac-text);
}
.ac-h1 em { font-style: normal; color: var(--ac-accent); }
.ac-lede { font-size: 1.12rem; line-height: 1.6; color: var(--ac-muted); max-width: 640px; margin: 0; }
.ac-section-title {
  font-family: var(--ac-heading); font-weight: 600; font-size: 1.45rem; letter-spacing: -0.01em;
  margin: 0.4rem 0 0.25rem; color: var(--ac-text);
}
.ac-section-sub { color: var(--ac-muted); margin: 0 0 1rem; font-size: 0.95rem; }
.ac-page-head { margin-bottom: 1.2rem; }
.ac-page-head h1 {
  font-family: var(--ac-heading); font-weight: 700; font-size: 2rem; letter-spacing: -0.02em;
  margin: 0 0 0.3rem; padding: 0; color: var(--ac-text);
}
.ac-page-head p { color: var(--ac-muted); margin: 0; }

/* ── Hero ───────────────────────────────────────────────────── */
.ac-hero {
  position: relative; overflow: hidden;
  padding: 3.2rem 2.6rem 2.6rem; border-radius: 22px; border: 1px solid var(--ac-border);
  background:
    radial-gradient(900px 380px at 85% -10%, rgba(200, 241, 53, 0.16), transparent 60%),
    radial-gradient(700px 360px at -10% 120%, rgba(124, 196, 255, 0.12), transparent 60%),
    var(--ac-surface);
  margin-bottom: 1.2rem;
}
.ac-hero-grid { display: grid; grid-template-columns: 1.35fr 1fr; gap: 2rem; align-items: center; }
@media (max-width: 860px) { .ac-hero-grid { grid-template-columns: 1fr; } .ac-hero { padding: 2.2rem 1.4rem; } }
.ac-hero-panel {
  border: 1px solid var(--ac-border); border-radius: var(--ac-radius); background: rgba(10, 13, 18, 0.6);
  padding: 1.1rem 1.2rem; font-size: 0.9rem;
}
.ac-flow-step { display: flex; gap: 0.75rem; align-items: flex-start; padding: 0.55rem 0; }
.ac-flow-step + .ac-flow-step { border-top: 1px dashed var(--ac-border); }
.ac-flow-dot {
  flex: none; width: 26px; height: 26px; border-radius: 8px; display: grid; place-items: center;
  font: 700 0.75rem/1 Inter, sans-serif; color: var(--ac-accent-ink); background: var(--ac-accent);
}
.ac-flow-step b { color: var(--ac-text); font-weight: 600; }
.ac-flow-step span { color: var(--ac-muted); display: block; font-size: 0.84rem; margin-top: 2px; }

/* ── Stats strip ────────────────────────────────────────────── */
.ac-stats {
  display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); gap: 0;
  border: 1px solid var(--ac-border); border-radius: var(--ac-radius); overflow: hidden; margin: 0.4rem 0 2.2rem;
  background: var(--ac-surface);
}
.ac-stat { padding: 1.05rem 1.2rem; border-right: 1px solid var(--ac-border); }
.ac-stat:last-child { border-right: 0; }
.ac-stat-num { font-family: var(--ac-heading); font-weight: 700; font-size: 1.6rem; color: var(--ac-text); }
.ac-stat-label { color: var(--ac-muted); font-size: 0.82rem; margin-top: 2px; }

/* ── Cards & grids ──────────────────────────────────────────── */
.ac-grid { display: grid; gap: 14px; grid-template-columns: repeat(auto-fill, minmax(240px, 1fr)); }
.ac-grid-3 { display: grid; gap: 14px; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); }
.ac-card {
  border: 1px solid var(--ac-border); border-radius: var(--ac-radius); background: var(--ac-surface);
  padding: 1.15rem 1.2rem;
}
.ac-card h3 {
  font-family: var(--ac-heading); font-size: 1.05rem; font-weight: 600; margin: 0.15rem 0 0.4rem;
  padding: 0; color: var(--ac-text);
}
.ac-card p { color: var(--ac-muted); font-size: 0.9rem; line-height: 1.55; margin: 0; }
.ac-card-icon {
  width: 34px; height: 34px; border-radius: 10px; display: grid; place-items: center; margin-bottom: 0.6rem;
  background: rgba(200, 241, 53, 0.1); color: var(--ac-accent); font-weight: 700; font-size: 0.85rem;
}

/* ── Chips ──────────────────────────────────────────────────── */
.ac-chip {
  display: inline-flex; align-items: center; gap: 6px; padding: 3px 10px; border-radius: 999px;
  font: 600 0.74rem/1.6 Inter, sans-serif; letter-spacing: 0.02em; border: 1px solid var(--ac-border);
  color: var(--ac-muted); background: var(--ac-surface-2); white-space: nowrap;
}
.ac-chip::before { content: ""; width: 6px; height: 6px; border-radius: 50%; background: currentColor; }
.ac-chip.plain::before { display: none; }
.ac-chip.green { color: var(--ac-green); border-color: rgba(61, 220, 151, 0.35); background: rgba(61, 220, 151, 0.08); }
.ac-chip.amber { color: var(--ac-amber); border-color: rgba(245, 184, 65, 0.35); background: rgba(245, 184, 65, 0.08); }
.ac-chip.red { color: var(--ac-red); border-color: rgba(255, 107, 107, 0.35); background: rgba(255, 107, 107, 0.08); }
.ac-chip.accent { color: var(--ac-accent); border-color: rgba(200, 241, 53, 0.35); background: rgba(200, 241, 53, 0.08); }
.ac-chip.blue { color: var(--ac-blue); border-color: rgba(124, 196, 255, 0.35); background: rgba(124, 196, 255, 0.08); }

/* ── KPI tiles ──────────────────────────────────────────────── */
.ac-kpis { display: grid; gap: 12px; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); margin: 0.2rem 0 1.4rem; }
.ac-kpi { border: 1px solid var(--ac-border); border-radius: var(--ac-radius); background: var(--ac-surface); padding: 0.95rem 1.05rem; }
.ac-kpi-label { color: var(--ac-muted); font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.08em; font-weight: 600; }
.ac-kpi-value { font-family: var(--ac-heading); font-weight: 700; font-size: 1.55rem; color: var(--ac-text); margin-top: 0.25rem; }
.ac-kpi-note { color: var(--ac-muted); font-size: 0.8rem; margin-top: 0.15rem; }
.ac-kpi.green .ac-kpi-value { color: var(--ac-green); }
.ac-kpi.amber .ac-kpi-value { color: var(--ac-amber); }
.ac-kpi.red .ac-kpi-value { color: var(--ac-red); }
.ac-kpi.accent .ac-kpi-value { color: var(--ac-accent); }

/* ── Athlete bar ────────────────────────────────────────────── */
.ac-athlete { display: flex; gap: 1rem; align-items: center; flex-wrap: wrap; }
.ac-avatar {
  width: 48px; height: 48px; border-radius: 14px; display: grid; place-items: center; flex: none;
  font-family: var(--ac-heading); font-weight: 700; color: var(--ac-accent-ink); background: var(--ac-accent);
}
.ac-athlete-name { font-family: var(--ac-heading); font-weight: 600; font-size: 1.2rem; color: var(--ac-text); }
.ac-athlete-meta { color: var(--ac-muted); font-size: 0.86rem; margin-top: 2px; }

/* ── Week grid ──────────────────────────────────────────────── */
.ac-week { display: grid; gap: 12px; grid-template-columns: repeat(4, minmax(0, 1fr)); }
@media (max-width: 1100px) { .ac-week { grid-template-columns: repeat(2, minmax(0, 1fr)); } }
@media (max-width: 640px) { .ac-week { grid-template-columns: 1fr; } }
.ac-day {
  border: 1px solid var(--ac-border); border-radius: var(--ac-radius); background: var(--ac-surface);
  padding: 1rem 1.05rem; display: flex; flex-direction: column; gap: 0.7rem; min-height: 180px;
}
.ac-day.rest { background: repeating-linear-gradient(135deg, var(--ac-surface), var(--ac-surface) 10px, #0E1319 10px, #0E1319 20px); }
.ac-day-head { display: flex; justify-content: space-between; align-items: center; gap: 8px; }
.ac-day-name { font-family: var(--ac-heading); font-weight: 600; font-size: 1.02rem; color: var(--ac-text); }
.ac-ex { list-style: none; margin: 0 !important; padding: 0 !important; display: flex; flex-direction: column; gap: 0.55rem; }
.ac-ex li { margin: 0 !important; padding: 0 !important; display: flex; justify-content: space-between; gap: 10px; font-size: 0.88rem; }
.ac-ex-name { color: var(--ac-text); font-weight: 500; }
.ac-ex-detail { color: var(--ac-muted); font-variant-numeric: tabular-nums; white-space: nowrap; }
.ac-ex-sub { color: var(--ac-muted); font-size: 0.76rem; display: block; margin-top: 1px; }
.ac-day-note { color: var(--ac-muted); font-size: 0.82rem; line-height: 1.5; border-top: 1px dashed var(--ac-border); padding-top: 0.6rem; }
.ac-day-fuel { display: flex; gap: 10px; flex-wrap: wrap; margin-top: auto; color: var(--ac-muted); font-size: 0.76rem; }
.ac-day-fuel b { color: var(--ac-text); font-weight: 600; }
.ac-rest-msg { color: var(--ac-muted); font-size: 0.88rem; }

/* ── Coach response ─────────────────────────────────────────── */
.ac-reply {
  border: 1px solid rgba(200, 241, 53, 0.35); border-radius: var(--ac-radius);
  background: linear-gradient(180deg, rgba(200, 241, 53, 0.06), transparent 70%), var(--ac-surface);
  padding: 1.1rem 1.25rem; margin: 0.4rem 0 1.4rem;
}
.ac-reply-head { display: flex; gap: 8px; align-items: center; flex-wrap: wrap; margin-bottom: 0.6rem; }
.ac-reply-q { color: var(--ac-muted); font-size: 0.88rem; }
.ac-reply-body { color: var(--ac-text); line-height: 1.6; font-size: 0.96rem; white-space: pre-wrap; }
.ac-reply ul { margin: 0.6rem 0 0; padding-left: 1.1rem; color: var(--ac-text); font-size: 0.9rem; }
.ac-reply li { margin: 0.2rem 0; }

/* ── Misc ───────────────────────────────────────────────────── */
.ac-divider { height: 1px; background: var(--ac-border); margin: 2rem 0 1.6rem; border: 0; }
.ac-empty {
  border: 1px dashed var(--ac-border); border-radius: var(--ac-radius); padding: 2rem 1.4rem; text-align: center;
  color: var(--ac-muted);
}
.ac-empty b { color: var(--ac-text); display: block; font-family: var(--ac-heading); font-size: 1.1rem; margin-bottom: 0.3rem; }
.ac-footer {
  margin-top: 3rem; padding-top: 1.4rem; border-top: 1px solid var(--ac-border); color: var(--ac-muted);
  font-size: 0.84rem; display: flex; justify-content: space-between; flex-wrap: wrap; gap: 0.6rem;
}
.ac-footer a { color: var(--ac-text); text-decoration: none; }
.ac-footer a:hover { color: var(--ac-accent); }
.ac-source { color: var(--ac-muted); font-size: 0.86rem; line-height: 1.55; }
</style>
"""


def inject() -> None:
    st.markdown(CSS, unsafe_allow_html=True)
