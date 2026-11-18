# AdaptiveCoach — What It Is and How To Explain It

Study doc for interviews. Plain English, no jargon assumed.

---

## 1. The one concept you need

**Not an agent:** you stuff everything the AI might need into the prompt (the whole
workout history as text) and it writes an answer. Smart autocomplete.

**An agent:** you hand the AI a *menu of functions* it's allowed to call. It gets a
short brief, then asks for what it needs:

```
AI:   "call get_effort_trend"
code: runs it, hands back the result
AI:   "now call get_muscle_group_volume"
code: runs it, hands back the result
AI:   "ok, I know enough — here's my answer"
```

That loop is the whole thing. **The AI decides what to look at.** ~30 lines, in
`graph/agent_runner.py`:

```python
for _ in range(max_rounds):
    response = llm_with_tools.invoke(messages)
    if not response.tool_calls:          # done thinking
        return parse(response.content)
    for call in response.tool_calls:     # wants information
        result = tool_map[call["name"]].invoke(call["args"])
        messages.append(ToolMessage(result))
```

---

## 2. The 30-second pitch

> "It's a supervisor-routed multi-agent system for adaptive training plans. A supervisor
> reads the athlete's request and routes to one of three paths — generate, adapt, or
> explain. Six specialist agents do the work: fatigue analysis, progression planning,
> plan writing, a critic that can send plans back for revision, an adaptation agent that
> rewrites your week mid-flight when you get injured, and an explainer. They share a tool
> layer over the workout history plus a RAG index of 44 cited sports-science papers.
> Safety-critical arithmetic stays in deterministic Python the agents can't override —
> and I validated the critic against a 14-case golden set: 100% verdict accuracy, zero
> false passes."

---

## 3. The architecture

```
START
  |
  v
context_loader ---> fatigue_analyst ---> SUPERVISOR
                                             |   <- the LLM chooses here
        +------------------------------------+------------------------+
        v "plan"                             v "adapt"                v "explain"
  progression_planner                  adaptation_agent            explainer
        |                                    |                        |
        v                                    v                        v
  nutrition_advisor                         END                      END
        |
        v
  plan_writer <-------+
        |             | "replan" (score < 0.75 and loops < 3)
        v             |
     critic ----------+
        | "end"
        v
       END
```

**Why fatigue runs before the supervisor:** to route "my knee hurts" correctly you need
to know whether they're already in the red zone. Judging the words alone throws away the
state that makes the decision right.

---

## 4. The six agents

### Agent 1 — `fatigue_analyst`
**Question:** how beaten up is this person?

Python computes ACWR first (e.g. **1.655**). The AI *cannot change that number* —
arithmetic stays in code, because an AI doing maths on training load is how you tell an
exhausted lifter to add weight.

Real trace — nobody told it which tools or what order:
```
-> get_effort_trend()
-> get_muscle_group_volume()
-> get_weekly_volume_summary()
```
Found: RPE up **+1.3 on Squat, Bench and RDL at identical weight and reps**, 100% of sets
near failure, glutes above the recoverable ceiling.

Output: `Pattern: Rising RPE at identical weight and reps with 100% near-failure sets`

**Say:** "It decides which fatigue signals to trust and what story they tell. The number
is deterministic; the interpretation is the agent."

---

### The `supervisor` — the router
**Question:** what does this person actually need right now?

Picks one of three doors: **plan** (new week) / **adapt** (fix existing week) /
**explain** (answer, change nothing).

**Say:** "This is where the LLM controls program flow rather than just generating text.
Before it existed, asking 'why did you deload me?' regenerated an entire week to answer a
question."

---

### Agent 2 — `progression_planner`
**Question:** push, hold, or back off?

```
-> search_knowledge_base("deload protocol after overreaching")
-> get_weekly_volume_summary()
```

**It wrote that search query itself.** A green-zone athlete gets a different self-written
query ("volume progression for intermediate lifters"). Previously a Python function built
the search string from a template.

Output: `deload, -35%` with a rationale citing what it actually found.

**Say:** "This is agentic RAG — the agent formulates and refines its own knowledge-base
queries rather than retrieving from a fixed template."

---

### Agent 3 — `plan_writer`
**Question:** what are the seven days?

Before prescribing a weight it looks up what you actually lifted:
```
-> get_exercise_history("squat")
```
So it progresses from your real numbers instead of inventing plausible ones.

**Say:** "It grounds prescriptions in logged performance, and it's the node the critic
loop re-enters with feedback when a plan is rejected."

---

### Agent 4 — `critic`
**Question:** is this plan good enough to ship?

Scores four dimensions 0-1: safety, coherence, groundedness, goal alignment.
**Below 0.75 -> back to `plan_writer` for a rewrite.** Up to 3 times.
It can search the knowledge base itself to check a claimed principle is real, not just
plausible-sounding.

**The story:** this critic used to score **1.00 on every single plan** — the revision loop
had never once fired. A judge that always says yes is the same as no judge. The fix moved
the mechanical rules into Python as caps the AI can't score past.

**Say:** "It's an evaluator-optimizer loop — one agent writes, another critiques, the
first revises. I validated it against a 14-case golden set: 100% verdict accuracy, zero
false passes."

---

### Agent 5 — `adaptation_agent`  <- LEAD WITH THIS ONE
**Question:** life happened, now what?

You type: *"my left knee hurts when I squat."*

```
-> view_day("monday")
-> search_knowledge_base("substitutes for squats with knee pain")
-> swap_exercise(Squat -> Goblet Squat)                 <- CHANGES YOUR SAVED PLAN
-> swap_exercise(Leg Press -> Bulgarian Split Squat)    <- CHANGES YOUR SAVED PLAN
```

Every other agent *answers*. This one **acts** — those last two calls rewrote a file on
disk.

**Why it's the most agentic:** you can list the steps of "write a training plan." You
cannot list the steps of "knee hurts / gym shut / travelling Thursday / slept badly."
Open-ended problem, so no fixed path is possible.

**The safety gate (great interview beat):** it can rewrite your plan, so there's a check.
Version one compared the new plan against all the rules and rejected failures — and it
rejected a *correct* knee substitution, because the plan it was editing was already
running hot from an earlier week. So now it compares violations **before vs after** and
rejects only what the agent itself introduced.
**An agent is accountable for what it changed, not what it inherited.**

---

### Agent 6 — `explainer`
**Question:** why did you do that?

Read-only — deliberately no write tools. Ask "why is my volume going up?" and it
investigates before answering, so you get "your ACWR is 1.047 and RPE has been steady at
7.2" instead of "you needed recovery."

**Say:** "Same tool layer, no write access. Structurally incapable of changing your plan
while answering a question about it."

---

## 5. The two nodes that are NOT agents (say so proudly)

**`context_loader`** — opens JSON files, reads the last 5 weeks. No AI. No decision to
make.

**`nutrition_advisor`** — Mifflin-St Jeor equation, activity multiplier, macro split.
Pure Python. It *used* to be a GPT-4o call doing arithmetic.

**Say:** "It was an LLM call doing maths. I replaced it with code — exact, instant, free.
Knowing where *not* to put a model is part of the design."

This answer makes you look better, not worse.

---

## 6. The tools (12 total)

| Tool | Type | Purpose |
|---|---|---|
| `get_weekly_volume_summary` | read | 5-week load and RPE trend |
| `get_recent_sessions` | read | last n sessions, one line each |
| `get_exercise_history` | read | every logged set of one lift |
| `get_muscle_group_volume` | read | hard sets + stimulating reps vs MEV/MAV |
| `get_effort_trend` | read | matched-load RPE drift, failure exposure |
| `search_knowledge_base` | read | semantic search over 44 papers |
| `view_day` | read | inspect a day before editing |
| `set_day_to_rest` | **write** | convert a day to rest |
| `move_session` | **write** | shift a session (refuses to double-up a day) |
| `swap_exercise` | **write** | substitute one movement for another |
| `adjust_exercise_load` | **write** | change weight/RPE without swapping |
| `add_note_to_day` | **write** | append coaching guidance |

Tools are built as closures over one run's data, so a tool call can only ever see the
requesting user's history.

---

## 7. The safety model

**The rule: the model has judgement, the code has veto.**

| Check | Threshold | Effect |
|---|---|---|
| Consecutive heavy days, same muscles | RPE > 7.0 | safety capped at 0.0 |
| Red fatigue with 4+ hard days | >= 4 | safety capped at 0.7 |
| High RPE during a deload | > 9.0 | safety capped at 0.8 |
| Deload that doesn't reduce volume | > 85% baseline | coherence capped at 0.0 |
| Increase that doesn't increase | <= 100% | coherence capped at 0.4 |
| Maintain that drifts | +/-15% | coherence capped at 0.6 |
| Swapped exercise reappears | — | coherence capped at 0.5 |
| **Any violation at all** | **-> 0.7 max** | forces a revision |

That last row matters: with four dimensions averaged, a plan with safety 0.0 and three
perfect scores lands on exactly 0.75 and sneaks through. So any violation hard-caps the
overall below the threshold.

---

## 8. The knowledge base (RAG)

44 peer-reviewed papers from Europe PMC -> 896 chunks -> FAISS vector search.

**RAG in one line:** chop documents into chunks, turn each into numbers that capture
meaning, and when the agent asks a question find the closest chunks and feed them to the
model. A search engine bolted onto the model's input.

- **Licences checked in code** — a paper is only saved if the API confirms a
  redistributable Creative Commons licence. 15 were rejected. That's why it's safe to
  publish.
- **Citations survive chunking**, so the critic can verify a claim against a named paper.

| Domain | Papers |
|---|---|
| Resistance training | 16 |
| Endurance | 8 |
| Nutrition | 8 |
| Recovery | 8 |
| Training load | 4 |

**Volunteer the gap:** the canonical ACWR papers are CC BY-NC and had to be excluded, so
the metric the system leans on hardest has the thinnest evidence behind it.

---

## 9. Why ACWR isn't the hero

ACWR = this week's load / recent average, where load = sets x reps x kg. Three problems:

- 5x5 @ 100kg and 3x20 @ 42kg are **identical tonnage**, completely different sessions
- a heavy single barely registers despite being the most fatiguing thing in the week
- pull-ups, dips and running score **exactly zero** — the endurance persona's fatigue was
  being computed entirely from accessory lifting

So effort-based signals were added, all from per-set RPE already logged (`RIR = 10 - RPE`):

| Signal | Catches | Tonnage says |
|---|---|---|
| RPE drift at matched load | same weight/reps getting harder | nothing changed |
| Stimulating reps | reps within 5 RIR of failure | ignores failure proximity |
| Failure exposure | share of sets at <=1 RIR | invisible |
| Hard sets vs MEV/MAV | per-muscle distribution | averaged away |

**Say:** "I implemented the standard metric, read the critique literature, found tonnage
is a poor load proxy for lifting, and added a complementary signal." Much stronger than
either defending it or quietly deleting it.

---

## 10. Interview Q&A

**"What does agent three actually decide?"**
> Plan writer decides session structure, exercise selection, loads and RPE targets — and
> it queries my actual lifting history first so the numbers progress from reality. The
> strongest one is the adaptation agent: I type "my knee hurts," it inspects the day,
> searches the corpus for substitution evidence, and rewrites two exercises on disk.

**"Where does the LLM control the flow?"**
> `graph/nodes/supervisor.py`. It routes between generate / adapt / explain. Before it
> existed, every request walked the same fixed path.

**"Why isn't nutrition an agent?"**
> Because it's Mifflin-St Jeor plus multipliers. It was an LLM call doing arithmetic —
> slower, costlier, less reliable than Python.

**"How do you know the critic works?"**
> A 14-case golden set — 100% verdict accuracy, 0% false-pass. Before I built it the
> critic scored 1.00 on everything and the revision loop had never fired. I only found
> that because I measured.

**"Isn't ACWR contested?"**
> Yes — mathematical coupling, arbitrary thresholds, replication failures. It's also
> built on tonnage, which scores bodyweight work as zero. That's why it's one input
> alongside effort-based signals now.

**"What if the agent makes an unsafe change?"**
> Blocked. Violations are compared before and after the edit; anything the agent
> introduced is rejected and the original kept. Only *new* problems count — I learned
> that after v1 rejected a correct injury swap for a problem the plan already had.

**"What would you do next?"**
> Add mocked-LLM tests for the agent loop (it's the least-covered code, 15%), then extend the eval to cover routing and adaptation safety — the planning half is measured,
> the agentic half isn't yet. Then storage: it's flat JSON with no locking, so concurrent
> users would corrupt each other.

**"How is it tested?"**
> Three layers. 71 pytest unit tests that run offline in about a second — no API keys,
> no mocks, just hand-built plans and histories. A 14-case golden set that measures the
> critic's verdict accuracy and false-pass rate. And GitHub Actions running all of it on
> every push: ruff lint, the suite on Python 3.11–3.13 with coverage, the offline eval as
> a build gate, and a job that builds the FAISS index and runs the retrieval tests.

**"What was the hardest bug?"**
> Three ACWR defects that all under-reported fatigue — the dangerous direction, since an
> overreached athlete gets told to add load. The subtlest: my acute window was the
> calendar week and chronic was four complete weeks, so the same training week read 0.51
> on Monday and 1.22 on Friday. Fixed with rolling 7/28-day windows.

---

## 11. Numbers to quote

| | |
|---|---|
| Agents | 6 specialists + 1 supervisor |
| Tools | 12 (6 read, 6 write) |
| Papers | 44, all CC BY, 896 chunks |
| Tests | 71 (66 offline + 5 retrieval), ~1 s |
| Coverage | 56% overall; rule engine 97%, metrics 91% |
| CI jobs | 5 per push (lint, tests x3 Python versions, RAG) |
| Golden set | 14 cases |
| Verdict accuracy | 100% |
| False-pass rate | 0% |
| Replan threshold | 0.75 |
| Cost per plan | ~$0.10 on GPT-4o |

---

## 12. Vocabulary

| Term | Plain English |
|---|---|
| Agent | AI that decides what to do next instead of following your script |
| Workflow | AI slotted into steps you fixed in advance |
| Tool calling | Letting the model run functions you've approved |
| RAG | Search real documents, feed results to the model |
| Embedding | Text turned into numbers so similar meanings sit close together |
| Vector store (FAISS) | The database that finds closest numbers fast |
| LangGraph | Library for wiring steps together with branches and loops |
| Evaluator-optimizer | One AI writes, another critiques, the first rewrites |
| LLM-as-judge | Using a model to score output (needs its own testing) |
| Golden set | Hand-labelled examples proving the system works |
| ACWR | This week's load / your recent average |
| RIR | Reps in reserve — how many you had left. RPE 8 = 2 RIR |
| MEV / MAV | Minimum effective / maximum adaptive weekly sets per muscle |

---

## 13. Commands

```bash
python -m scripts.build_corpus        # fetch papers (licence-gated)
python -m rag.ingest                  # build the vector index
python -m scripts.refresh_eval_contexts
python -m scripts.seed_demo_data      # demo personas, prints each ACWR
python -m scripts.list_models         # what your API key can reach + tool-call test
python -m evals.run_eval --offline    # free
python -m evals.run_eval              # live
pip install -r requirements-dev.txt   # adds pytest-cov + ruff
ruff check .                          # lint
python -m pytest -q --cov=graph --cov=evals   # 71 tests + coverage
streamlit run app.py
```

**Note:** Streamlit caches imported modules. Editing anything other than `app.py` needs a
full Ctrl+C restart — a browser refresh silently runs stale code.

---

## 14. Testing and CI

**Three layers, cheapest first:**

| Layer | What | Cost |
|---|---|---|
| Unit tests | 71 pytest tests: ACWR maths, RIR metrics, rule engine, critic loop, adaptation tools + safety gate, nutrition, retrieval | free, ~1 s, offline |
| Golden set | 14 hand-labelled plans (6 good, 8 flawed) — offline mode checks the rule layer, live mode runs the real critic | offline free / live ~API calls |
| CI | GitHub Actions on every push and PR (`.github/workflows/ci.yml`) | free on public repos |

**The CI pipeline (5 jobs, run in parallel):**

1. **Lint** — `ruff` for syntax errors, undefined names, unused imports (it found 8)
2. **Tests x 3** — pytest + coverage on Python 3.11, 3.12, 3.13; then the offline
   golden-set eval, which fails the build if any of the 14 verdicts is wrong; uploads
   `coverage.xml` and JUnit results
3. **RAG** — downloads the embedding model, builds the FAISS index from the 44 papers,
   runs the 5 retrieval tests. Model and index are cached, so the index only rebuilds
   when the papers or the ingest code change.

**Tools:** pytest, pytest-cov, ruff, GitHub Actions (`actions/cache`,
`actions/upload-artifact`). No mocking library — tests build plain dicts.

**Be honest about:**
- Coverage is 56%. Deterministic code is well covered; the LLM-calling agents aren't,
  because testing them properly needs mocked model responses.
- It's CI, not CD — nothing deploys automatically.
- CI never calls the LLM. The live eval is a manual step so the pipeline costs nothing.

**Say:** "Every push is linted, tested on three Python versions, and gated on the golden
set — if a change to the safety rules stops them flagging one of the known-bad plans, the
build fails. The LLM half of the critic is checked by the live eval, run by hand."
