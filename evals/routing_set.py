"""Golden set for the supervisor: hand-labelled athlete requests and the path
each one should take. Labels follow the routing rules in prompts/supervisor.py:
disruption or pain with a plan in place -> adapt; a question that asks for no
change -> explain; a request for a new week -> plan; no plan yet -> plan.

Run live with: python -m evals.run_routing_eval
"""

ROUTES = ("plan", "adapt", "explain")

CASES = [
    # ── adapt: something disrupted the current week ──
    {"name": "knee_pain", "request": "My left knee hurts when I squat", "expected": "adapt"},
    {"name": "missed_day", "request": "I can't make it to the gym on Thursday", "expected": "adapt"},
    {"name": "broken_rack", "request": "The squat rack at my gym is broken all week",
     "expected": "adapt"},
    {"name": "poor_sleep", "request": "I only slept four hours, can tomorrow be lighter?",
     "expected": "adapt"},
    {"name": "travel", "request": "I'm travelling Wednesday to Friday and will only have dumbbells",
     "expected": "adapt"},
    # ── explain: a question, no change requested ──
    {"name": "why_deload", "request": "Why is this week a deload?", "expected": "explain"},
    {"name": "what_rpe", "request": "What does RPE 7 actually mean?", "expected": "explain"},
    {"name": "why_split", "request": "Why did you put bench on both Monday and Thursday?",
     "expected": "explain"},
    {"name": "protein_q", "request": "How much protein should I hit on rest days?",
     "expected": "explain"},
    {"name": "ambiguous", "request": "hmm", "expected": "explain",
     "note": "genuinely ambiguous -> the path that changes nothing"},
    # ── plan: a new week is what's wanted ──
    {"name": "fresh_week", "request": "Give me a fresh plan for next week", "expected": "plan"},
    {"name": "goal_change", "request": "My goal is now a marathon, redo my whole programme",
     "expected": "plan"},
    {"name": "start_over", "request": "Scrap this week and start over completely",
     "expected": "plan"},
    # ── deterministic short-circuit: nothing to adapt without a plan ──
    {"name": "no_plan_pain", "request": "My knee hurts", "expected": "plan", "plan_exists": False,
     "note": "no plan yet -> always plan, decided in code without the model"},
]
