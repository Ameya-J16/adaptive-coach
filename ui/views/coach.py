"""The coach: generate a week, talk to the coach, see the plan."""
from html import escape

import streamlit as st

from ui import athletes
from ui.components import (
    ACTION_CLASS,
    chip,
    empty_state,
    footer,
    html,
    kpis,
    page_head,
    section,
    week_grid,
)

NODE_LABELS = {
    "context_loader": ("Context loader", "Loaded training history and session memory"),
    "fatigue_analyst": ("Fatigue analyst", "Computed ACWR and read the effort trends"),
    "supervisor": ("Supervisor", "Decided what this request needs"),
    "progression_planner": ("Progression planner", "Searched the research and set the load"),
    "nutrition_advisor": ("Nutrition", "Calculated training- and rest-day targets"),
    "plan_writer": ("Plan writer", "Wrote the seven days"),
    "critic": ("Critic", "Scored the plan for safety, coherence and evidence"),
    "adaptation_agent": ("Adaptation agent", "Edited your saved plan"),
    "explainer": ("Explainer", "Investigated your data to answer"),
}
ROUTE_TEXT = {"plan": "New plan", "adapt": "Adapted your week", "explain": "Answered your question"}
EXAMPLES = [
    "My left knee hurts when I squat",
    "I can't train on Thursday this week",
    "Why is this week set up the way it is?",
    "I slept badly — go easier tomorrow",
]


def _initial_state(user_id: str, request: str) -> dict:
    from main import _load_profile

    return {
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
        "user_request": request,
    }


def _run(user_id: str, request: str) -> dict | None:
    """Stream the graph node by node, showing each agent as it finishes."""
    from graph.graph import graph
    from tracing.langfuse_config import flush, trace_run

    state = _initial_state(user_id, request)
    result = dict(state)
    status = st.status("Your coaching team is working…", expanded=True)
    try:
        with trace_run(
            "coach_request" if request else "generate_plan",
            user_id=user_id,
            tags=["streamlit"],
            input={"request": request or "plan my week"},
        ) as trace:
            for update in graph.stream(state, stream_mode="updates"):
                for node, patch in update.items():
                    result.update(patch or {})
                    title, desc = NODE_LABELS.get(node, (node, ""))
                    line = f"**{title}** — {desc}"
                    if node == "supervisor" and result.get("route"):
                        line += f"  \n↳ chose **{ROUTE_TEXT.get(result['route'], result['route'])}**"
                    if node == "critic":
                        line += (
                            f"  \n↳ score **{result.get('critic_score', 0):.2f}**"
                            f" · review round {result.get('loop_count', 0)}"
                        )
                    status.write(line)
            if trace.get("span") is not None:
                try:
                    trace["span"].update(output={
                        "route": result.get("route"),
                        "critic_score": result.get("critic_score"),
                    })
                except Exception:
                    pass
        status.update(label="Done", state="complete", expanded=False)
        result["_trace_url"] = trace.get("trace_url")
        result["_request"] = request
        return result
    except Exception as e:
        status.update(label="Something went wrong", state="error", expanded=True)
        st.error(f"{type(e).__name__}: {e}")
        return None
    finally:
        flush()
        athletes.snapshot.clear()


def _athlete_header(user_id: str) -> None:
    snap = athletes.snapshot(user_id)
    initials = "".join(w[0] for w in athletes.label(user_id).split()[:2]).upper() or "A"
    desc = athletes.PERSONAS.get(user_id, ("", "Your own training log."))[1]
    left, right = st.columns([2.2, 1], vertical_alignment="center")
    with left:
        html(
            '<div class="ac-athlete">'
            f'<div class="ac-avatar">{escape(initials)}</div><div>'
            f'<div class="ac-athlete-name">{escape(athletes.label(user_id))}</div>'
            f'<div class="ac-athlete-meta">{escape(desc)} · <code>{escape(user_id)}</code></div>'
            "</div></div>"
        )
    with right:
        athletes.athlete_picker("coach")

    acwr = snap["acwr"]
    kpis([
        {"label": "Fatigue zone", "value": snap["zone"].upper(), "tone": snap["zone"],
         "note": athletes.ZONE_TEXT[snap["zone"]] if acwr is not None else "No 4-week baseline yet"},
        {"label": "ACWR", "value": f"{acwr:.2f}" if acwr is not None else "—",
         "note": "7-day vs 28-day load"},
        {"label": "Sessions this week", "value": snap["sessions_week"],
         "note": f'{snap["sessions_total"]} logged in total'},
        {"label": "Last session",
         "value": "Today" if snap["days_since"] == 0 else
                  (f'{snap["days_since"]}d ago' if snap["days_since"] is not None else "—"),
         "note": snap["last_session"] or "Nothing logged yet"},
    ])


def _actions(user_id: str) -> None:
    from memory.plan_store import has_plan

    plan_exists = has_plan(user_id)
    left, right = st.columns([1, 1.35], gap="medium")
    with left:
        with st.container(border=True):
            html(
                '<div class="ac-card-icon">⚡</div>'
                '<h3 style="font-family:var(--ac-heading);margin:0 0 0.3rem;font-size:1.1rem">'
                "Plan my week</h3>"
                '<p style="color:var(--ac-muted);font-size:0.9rem;margin:0 0 0.8rem">'
                "Runs the full team: fatigue, evidence, load, nutrition, seven days — and a critic "
                "that sends weak drafts back. Takes a minute or two.</p>"
            )
            if st.button(
                "Generate a new week" if plan_exists else "Generate my first week",
                type="primary", use_container_width=True, key="gen_plan",
            ):
                st.session_state["pending"] = (user_id, "")
    with right:
        with st.container(border=True):
            html(
                '<div class="ac-card-icon">💬</div>'
                '<h3 style="font-family:var(--ac-heading);margin:0 0 0.3rem;font-size:1.1rem">'
                "Talk to your coach</h3>"
                '<p style="color:var(--ac-muted);font-size:0.9rem;margin:0 0 0.6rem">'
                "Report a niggle, a schedule change or ask why — the supervisor decides whether to "
                "adapt your week or explain it.</p>"
            )

            def _use_example():
                pick = st.session_state.get("coach_example")
                if pick:
                    st.session_state["coach_request"] = pick

            st.pills("Examples", EXAMPLES, key="coach_example", on_change=_use_example,
                     label_visibility="collapsed")
            with st.form("coach_form", clear_on_submit=False, border=False):
                request = st.text_area(
                    "Message", key="coach_request", height=80, label_visibility="collapsed",
                    placeholder="e.g. My left knee hurts when I squat",
                )
                sent = st.form_submit_button("Send to coach", use_container_width=True,
                                             disabled=not plan_exists)
            if not plan_exists:
                st.caption("Generate a first week, then the coach can adapt or explain it.")
            if sent and request.strip():
                st.session_state["pending"] = (user_id, request.strip())


def _reply(run: dict) -> None:
    route = run.get("route", "plan")
    request = run.get("_request", "")
    if not request:
        return
    head = chip(ROUTE_TEXT.get(route, route), "accent")
    q = f'<span class="ac-reply-q">You: “{escape(request)}”</span>'
    body = ""
    if route == "explain":
        body = f'<div class="ac-reply-body">{escape(run.get("explanation", "") or "")}</div>'
    elif route == "adapt":
        ar = run.get("adaptation_result", {}) or {}
        body = f'<div class="ac-reply-body">{escape(ar.get("summary", "") or "")}</div>'
        changes = ar.get("changes_made", []) or []
        rejected = ar.get("rejected_violations", []) or []
        if changes and not rejected:
            body += "<ul>" + "".join(f"<li>{escape(str(c))}</li>" for c in changes) + "</ul>"
        if rejected:
            body += (
                '<div style="margin-top:0.6rem">' + chip("Safety gate blocked this edit", "red")
                + "</div><ul>" + "".join(f"<li>{escape(str(v))}</li>" for v in rejected) + "</ul>"
            )
    else:
        body = f'<div class="ac-reply-body">{escape(run.get("route_reason", "") or "")}</div>'
    reason = run.get("route_reason")
    why = (
        f'<div class="ac-reply-q" style="margin-top:0.7rem">Supervisor: {escape(reason)}</div>'
        if reason and route != "plan" else ""
    )
    html(f'<div class="ac-reply"><div class="ac-reply-head">{head}{q}</div>{body}{why}</div>')


def _plan_section(user_id: str, run: dict | None) -> None:
    from memory.plan_store import get_changes, load_record

    record = load_record(user_id)
    fresh = bool(run and run.get("route") == "plan" and run.get("weekly_plan"))
    plan = run["weekly_plan"] if fresh else (record or {}).get("plan")
    if not plan:
        section("Your week")
        empty_state(
            "No plan yet",
            "Generate a week and it will appear here as seven day cards.",
        )
        return

    fr = (run or {}).get("fatigue_report") or (record or {}).get("fatigue_report", {}) or {}
    pdx = (run or {}).get("progression_decision") or (record or {}).get("progression_decision", {}) or {}
    nutrition = (run or {}).get("nutrition_targets", {}) if run else {}
    changes = get_changes(user_id)

    created = (record or {}).get("created_at", "")
    meta = f"Generated {created.replace('T', ' at ')}" if created else ""
    if changes:
        meta += f" · edited {len(changes)}× by the adaptation agent"
    section("Your week", meta)

    action = str(pdx.get("action", "—"))
    adj = pdx.get("load_adjustment_percent")
    items = [
        {"label": "This week", "value": action.upper(), "tone": ACTION_CLASS.get(action, ""),
         "note": f"Load {adj:+.0f}% vs baseline" if isinstance(adj, (int, float)) else ""},
        {"label": "Fatigue read", "value": str(fr.get("fatigue_level", "—")).upper(),
         "tone": fr.get("fatigue_level", ""), "note": "From the fatigue analyst"},
    ]
    if fresh:
        score = run.get("critic_score", 0.0)
        items.append({
            "label": "Critic score", "value": f"{score:.2f}",
            "tone": "green" if score >= 0.75 else "amber",
            "note": f"{run.get('loop_count', 0)} review round(s) · pass mark 0.75",
        })
        tday = (nutrition or {}).get("training_day", {})
        if tday:
            items.append({"label": "Training-day fuel", "value": f"{tday.get('calories', '–')} kcal",
                          "note": f"{tday.get('protein_g', '–')}g protein"})
    kpis(items)

    week_grid(plan, nutrition)
    if plan.get("overall_notes"):
        html(f'<div class="ac-card" style="margin-top:12px"><p>{escape(str(plan["overall_notes"]))}</p></div>')

    st.write("")
    tabs = st.tabs(["Why this plan", "Critic review", "Research used", "Change log", "Trace"])
    with tabs[0]:
        rows = [
            ("Pattern spotted", fr.get("dominant_stress_pattern")),
            ("Fatigue analyst", fr.get("recommendation_hint")),
            ("Progression decision", pdx.get("rationale")),
            ("Nutrition timing", (nutrition or {}).get("timing_notes")),
        ]
        shown = [(k, v) for k, v in rows if v]
        if shown:
            for k, v in shown:
                st.markdown(f"**{k}**")
                st.write(v)
        else:
            st.caption("Reasoning is shown for plans generated in this session.")
    with tabs[1]:
        if fresh and run.get("critic_feedback"):
            st.write(run["critic_feedback"])
        else:
            st.caption("The critic's review is shown right after a plan is generated.")
    with tabs[2]:
        ctx = (run or {}).get("retrieved_context") if fresh else ""
        if ctx:
            st.markdown(ctx)
        else:
            st.caption("The papers the planner cited appear here after a plan is generated.")
    with tabs[3]:
        if changes:
            for c in reversed(changes):
                st.markdown(f"**{c.get('at', '').replace('T', ' ')}** — {c.get('change', '')}")
        else:
            st.caption("No edits yet. Ask the coach to change something and it is logged here.")
    with tabs[4]:
        url = (run or {}).get("_trace_url")
        if url:
            st.link_button("Open this run in Langfuse", url)
            st.caption("Every agent step, tool call, prompt and token count for this run.")
        else:
            st.caption(
                "Langfuse tracing is off or this plan wasn't generated in this session. Add "
                "LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY to the app's secrets to record "
                "every run."
            )


def render() -> None:
    page_head("Coach", "Your training, read by a team of agents — plan the week or just talk to it.")
    user_id = athletes.current_athlete()
    _athlete_header(user_id)
    user_id = athletes.current_athlete()  # the picker may have changed it
    _actions(user_id)

    runs = st.session_state.setdefault("runs", {})
    pending = st.session_state.pop("pending", None)
    if pending:
        uid, request = pending
        result = _run(uid, request)
        if result is not None:
            runs[uid] = result

    run = runs.get(user_id)
    html('<hr class="ac-divider" style="margin:1.4rem 0 1.2rem">')
    if run:
        _reply(run)
    _plan_section(user_id, run)
    footer()
