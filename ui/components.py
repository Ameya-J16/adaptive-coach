"""HTML building blocks for the site.

Everything that comes from data or a model (plan text, notes, exercise names,
the coach's replies) is escaped before it is placed in HTML.
"""
from html import escape

import streamlit as st

DAY_ORDER = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
REST_TYPES = {"rest", "deload_rest", "off", ""}
ZONE_CLASS = {"green": "green", "amber": "amber", "red": "red"}
ACTION_CLASS = {"increase": "green", "maintain": "blue", "deload": "amber"}


def html(markup: str) -> None:
    """Render a block of trusted markup (callers escape any dynamic text)."""
    st.html(markup)


def chip(text: str, tone: str = "", plain: bool = False) -> str:
    classes = " ".join(c for c in ("ac-chip", tone, "plain" if plain else "") if c)
    return f'<span class="{classes}">{escape(str(text))}</span>'


def page_head(title: str, subtitle: str = "") -> None:
    sub = f"<p>{escape(subtitle)}</p>" if subtitle else ""
    html(f'<div class="ac-page-head"><h1>{escape(title)}</h1>{sub}</div>')


def section(title: str, subtitle: str = "") -> None:
    sub = f'<p class="ac-section-sub">{escape(subtitle)}</p>' if subtitle else ""
    html(f'<div class="ac-section-title">{escape(title)}</div>{sub}')


def kpis(items: list[dict]) -> None:
    """items: {label, value, note?, tone?}"""
    tiles = []
    for it in items:
        note = f'<div class="ac-kpi-note">{escape(str(it["note"]))}</div>' if it.get("note") else ""
        tiles.append(
            f'<div class="ac-kpi {it.get("tone", "")}">'
            f'<div class="ac-kpi-label">{escape(it["label"])}</div>'
            f'<div class="ac-kpi-value">{escape(str(it["value"]))}</div>{note}</div>'
        )
    html(f'<div class="ac-kpis">{"".join(tiles)}</div>')


def empty_state(title: str, body: str) -> None:
    html(f'<div class="ac-empty"><b>{escape(title)}</b>{escape(body)}</div>')


def _fmt_weight(w) -> str:
    try:
        w = float(w)
    except (TypeError, ValueError):
        return ""
    if w <= 0:
        return "BW"
    return f"{w:g} kg"


def _exercise_row(ex: dict) -> str:
    name = escape(str(ex.get("name", "Exercise")))
    sets, reps = ex.get("sets"), ex.get("reps")
    scheme = f"{sets}×{reps}" if sets and reps else ""
    weight = _fmt_weight(ex.get("weight_kg"))
    detail = " · ".join(p for p in (scheme, weight) if p)
    rpe = ex.get("rpe_target")
    muscles = ", ".join(ex.get("muscle_groups", []) or [])
    sub_parts = [f"RPE {rpe:g}" if isinstance(rpe, (int, float)) and rpe else "", muscles]
    sub = " · ".join(p for p in sub_parts if p)
    sub_html = f'<span class="ac-ex-sub">{escape(sub)}</span>' if sub else ""
    return (
        f'<li><div><span class="ac-ex-name">{name}</span>{sub_html}</div>'
        f'<span class="ac-ex-detail">{escape(detail)}</span></li>'
    )


def _fuel(nt: dict) -> str:
    if not nt:
        return ""
    return (
        '<div class="ac-day-fuel">'
        f'<span><b>{escape(str(nt.get("calories", "–")))}</b> kcal</span>'
        f'<span><b>{escape(str(nt.get("protein_g", "–")))}</b>g protein</span>'
        f'<span><b>{escape(str(nt.get("carbs_g", "–")))}</b>g carbs</span>'
        f'<span><b>{escape(str(nt.get("fat_g", "–")))}</b>g fat</span>'
        "</div>"
    )


def week_grid(plan: dict, nutrition: dict | None = None) -> None:
    """The 7-day plan as a responsive grid of day cards."""
    days = (plan or {}).get("days", {}) or {}
    nutrition = nutrition or {}
    cards = []
    for day in DAY_ORDER:
        content = days.get(day, {}) or {}
        session_type = str(content.get("session_type", "rest") or "rest")
        exercises = content.get("exercises", []) or []
        is_rest = not exercises or session_type.lower() in REST_TYPES
        tone = "plain" if is_rest else ("amber" if "deload" in session_type.lower() else "accent")
        badge = chip(session_type.replace("_", " ").title(), tone, plain=is_rest)
        if exercises:
            body = '<ul class="ac-ex">' + "".join(_exercise_row(e) for e in exercises) + "</ul>"
        else:
            body = '<div class="ac-rest-msg">Rest &amp; recover — sleep, walk, eat well.</div>'
        note = content.get("notes")
        note_html = f'<div class="ac-day-note">{escape(str(note))}</div>' if note else ""
        fuel = _fuel(nutrition.get("rest_day" if is_rest else "training_day", {}))
        cards.append(
            f'<div class="ac-day {"rest" if is_rest else ""}">'
            f'<div class="ac-day-head"><span class="ac-day-name">{day.title()}</span>{badge}</div>'
            f"{body}{note_html}{fuel}</div>"
        )
    html(f'<div class="ac-week">{"".join(cards)}</div>')


def footer() -> None:
    html(
        '<div class="ac-footer">'
        "<span>AdaptiveCoach · multi-agent training plans grounded in sports-science research</span>"
        '<span><a href="https://github.com/Ameya-J16/adaptive-coach" target="_blank">GitHub</a>'
        " · LangGraph · OpenAI · FAISS · Streamlit</span>"
        "</div>"
    )
