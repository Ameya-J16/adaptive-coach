"""Training metrics built on reps-in-reserve rather than tonnage.

Why this exists: volume load (sets x reps x kg) is a poor proxy for training
stress in resistance training. 5x5 @ 100kg and 3x20 @ 42kg are the same
tonnage and completely different sessions; a heavy single logs almost nothing
while being the most fatiguing thing in a week; and anything without an
external load — pull-ups, dips, running — contributes exactly zero.

Reps-in-reserve fixes all three, because it measures how close a set came to
failure instead of how much mass moved. RIR is already in the data: sets are
logged with a per-set RPE, and on the RIR-based RPE scale RIR = 10 - RPE.

Four signals, each answering a question tonnage cannot:

  stimulating reps  — how much growth stimulus did a muscle actually receive?
  failure exposure  — how much of it came from the disproportionately
                      fatiguing sets taken to or near failure?
  RPE drift         — is the same weight for the same reps getting harder?
  set counts        — how does weekly volume compare to MEV/MAV landmarks?

None of these replace ACWR outright; they are the signals a lifter would
actually recognise, and they degrade gracefully where ACWR is blind.
"""
from collections import defaultdict
from datetime import datetime, timedelta

# Reps closer to failure than this drive most of the hypertrophy stimulus.
# A set of 10 taken to 2 RIR finishes at RIR 5,4,3,2 — four stimulating reps.
STIMULATING_RIR_THRESHOLD = 5

# Sets at or below this RIR carry a fatigue cost out of proportion to their
# stimulus, so they are counted separately rather than folded into the total.
FAILURE_RIR_THRESHOLD = 1

# Weekly hard sets per muscle group. From the practitioner summary's MEV/MAV
# landmarks; treated as rough bands, not precise prescriptions.
MEV_SETS = 8
MAV_SETS = 20


def rir_from_rpe(rpe: float | None) -> float | None:
    """Reps in reserve implied by an RIR-based RPE score."""
    if rpe is None:
        return None
    return max(0.0, 10.0 - float(rpe))


def stimulating_reps(reps: int, rir: float | None) -> int:
    """Reps in a set performed within STIMULATING_RIR_THRESHOLD of failure.

    A set of `reps` finishing at `rir` passes through descending RIR values;
    the ones at or below the threshold are the stimulating portion. Sets stopped
    far from failure contribute none, which is the behaviour tonnage misses.
    """
    if rir is None:
        return 0
    if rir > STIMULATING_RIR_THRESHOLD:
        return 0
    return int(min(reps, STIMULATING_RIR_THRESHOLD - rir + 1))


def _iter_sets(session: dict):
    """Yield (exercise, set_dict) for both logging formats."""
    for ex in session.get("exercises", []):
        if ex.get("sets_data"):
            for s in ex["sets_data"]:
                yield ex, s
        elif ex.get("sets"):
            # Averaged format: synthesise sets from the exercise-level fields.
            for _ in range(int(ex.get("sets", 0))):
                yield ex, {
                    "reps": ex.get("reps", 0),
                    "weight_kg": ex.get("weight_kg", 0),
                    "rpe": ex.get("rpe_target") or session.get("avg_rpe"),
                }


def _in_last_days(session: dict, days: int, today=None) -> bool:
    today = today or datetime.now().date()
    cutoff = (today - timedelta(days=days - 1)).isoformat()
    return bool(session.get("date")) and session["date"] >= cutoff


def weekly_muscle_summary(sessions: list[dict], days: int = 7, today=None) -> dict:
    """Per-muscle-group hard sets, stimulating reps and failure sets over a window.

    A set counts toward every muscle group the exercise lists, which is how
    set-counting landmarks are normally applied.
    """
    recent = [s for s in sessions if _in_last_days(s, days, today)]
    out = defaultdict(lambda: {"hard_sets": 0, "stimulating_reps": 0, "failure_sets": 0})

    for session in recent:
        for ex, s in _iter_sets(session):
            rir = rir_from_rpe(s.get("rpe"))
            if rir is None:
                continue
            stim = stimulating_reps(int(s.get("reps", 0)), rir)
            groups = ex.get("muscle_groups") or ["unspecified"]
            for g in groups:
                key = g.lower()
                if rir <= STIMULATING_RIR_THRESHOLD:
                    out[key]["hard_sets"] += 1
                out[key]["stimulating_reps"] += stim
                if rir <= FAILURE_RIR_THRESHOLD:
                    out[key]["failure_sets"] += 1

    for stats in out.values():
        sets_done = stats["hard_sets"]
        if sets_done < MEV_SETS:
            stats["status"] = "below MEV"
        elif sets_done > MAV_SETS:
            stats["status"] = "above MAV"
        else:
            stats["status"] = "productive range"
    return dict(out)


def failure_exposure(sessions: list[dict], days: int = 7, today=None) -> dict:
    """How much of the recent work was taken to or near failure."""
    recent = [s for s in sessions if _in_last_days(s, days, today)]
    total = near_failure = 0
    for session in recent:
        for _, s in _iter_sets(session):
            rir = rir_from_rpe(s.get("rpe"))
            if rir is None:
                continue
            total += 1
            if rir <= FAILURE_RIR_THRESHOLD:
                near_failure += 1
    share = (near_failure / total) if total else 0.0
    return {
        "total_sets": total,
        "near_failure_sets": near_failure,
        "share": round(share, 3),
        # Sustained high-failure training is the classic route to accumulated
        # fatigue without extra stimulus.
        "flag": "high" if share > 0.35 else ("moderate" if share > 0.2 else "low"),
    }


def rpe_drift(sessions: list[dict], min_occurrences: int = 2, today=None) -> list[dict]:
    """Detect the same work getting harder.

    For each exercise, compare the most recent sets against earlier sets at the
    SAME weight and reps. A rising RPE on identical work is accumulated fatigue
    that tonnage cannot see — the week looks unchanged because it is unchanged.
    """
    by_exercise: dict[str, list[tuple[str, float, int, float]]] = defaultdict(list)
    for session in sessions:
        date = session.get("date", "")
        for ex, s in _iter_sets(session):
            rpe = s.get("rpe")
            if rpe is None:
                continue
            by_exercise[ex.get("name", "?")].append(
                (date, float(s.get("weight_kg", 0)), int(s.get("reps", 0)), float(rpe))
            )

    findings = []
    for name, entries in by_exercise.items():
        entries.sort(key=lambda e: e[0])
        # Group by the (weight, reps) prescription so we compare like with like.
        matched: dict[tuple, list[tuple[str, float]]] = defaultdict(list)
        for date, weight, reps, rpe in entries:
            matched[(weight, reps)].append((date, rpe))

        for (weight, reps), points in matched.items():
            dates = sorted({d for d, _ in points})
            if len(dates) < min_occurrences:
                continue
            first_day, last_day = dates[0], dates[-1]
            early = [r for d, r in points if d == first_day]
            late = [r for d, r in points if d == last_day]
            delta = (sum(late) / len(late)) - (sum(early) / len(early))
            if abs(delta) < 0.5:
                continue
            findings.append({
                "exercise": name,
                "weight_kg": weight,
                "reps": reps,
                "from_date": first_day,
                "to_date": last_day,
                "rpe_change": round(delta, 2),
                "direction": "harder" if delta > 0 else "easier",
            })

    # Biggest movements first — those are the ones worth acting on.
    findings.sort(key=lambda f: abs(f["rpe_change"]), reverse=True)
    return findings


def readiness_signals(sessions: list[dict], today=None) -> dict:
    """Everything above in one call, for the fatigue analyst and its tools."""
    drift = rpe_drift(sessions, today=today)
    harder = [d for d in drift if d["direction"] == "harder"]
    return {
        "muscle_summary": weekly_muscle_summary(sessions, today=today),
        "failure_exposure": failure_exposure(sessions, today=today),
        "rpe_drift": drift[:5],
        "drift_flag": "rising effort at matched loads" if harder else "stable",
    }
