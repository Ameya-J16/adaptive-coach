# AdaptiveCoach — Multi-Agent Training Coach

> A LangGraph multi-agent system that reads an athlete's logged training, judges fatigue, and plans, adapts or explains their week. A supervisor model decides which path each request takes, specialist agents call tools to gather their own evidence from the training log and a corpus of 44 peer-reviewed papers, and a critic checked by deterministic safety rules sends weak plans back for revision. Every run is traced in Langfuse.

[![Live demo](https://img.shields.io/badge/Live%20demo-Open%20app-C8F135?style=flat&logo=streamlit&logoColor=black)](https://adaptive-coach-wmemrgmrvmrazabzrgsheu.streamlit.app/)
[![CI](https://github.com/Ameya-J16/adaptive-coach/actions/workflows/ci.yml/badge.svg)](https://github.com/Ameya-J16/adaptive-coach/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=flat&logo=python&logoColor=white)
![LangGraph](https://img.shields.io/badge/LangGraph-1C3C3C?style=flat)
![LangChain](https://img.shields.io/badge/LangChain-1C3C3C?style=flat&logo=langchain&logoColor=white)
![OpenAI](https://img.shields.io/badge/OpenAI-GPT--5.4%20mini-412991?style=flat&logo=openai&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-UI-FF4B4B?style=flat&logo=streamlit&logoColor=white)
![Langfuse](https://img.shields.io/badge/Langfuse-Observability-000000?style=flat)

**▶ Live demo: [adaptive-coach-wmemrgmrvmrazabzrgsheu.streamlit.app](https://adaptive-coach-wmemrgmrvmrazabzrgsheu.streamlit.app/)** — pick a demo athlete,
generate a week, then tell the coach "my knee hurts when I squat". Deployed automatically
from `main` by the CI/CD pipeline.

---

## Architecture

```
START
  │
  ▼
context_loader ──► fatigue_analyst ──► SUPERVISOR  (the model chooses the path)
 (code: history,    (ACWR in code;        │
  session memory)    agent reads           ├── "plan" ──► progression_planner ──► nutrition_advisor ──► plan_writer ◄──┐
                     effort trends)        │              (searches the papers)    (code: macros)          │           │ replan:
                                           │                                                               ▼           │ score < 0.75
                                           │                                                             critic ───────┘ and < 3 rounds
                                           │                                                   (rule engine caps + LLM judge)
                                           │                                                               │
                                           │                                                               ▼
                                           │                                                       plan saved ──► END
                                           ├── "adapt" ──► adaptation_agent ──► safety gate ──► END
                                           │               (edits the saved plan with write tools)
                                           └── "explain" ──► explainer ──► END
                                                            (read-only tools; changes nothing)

Every request ──► one Langfuse trace (each agent, tool call and model response)
```

`fatigue_analyst` runs before the supervisor on purpose: "my knee hurts" should be
routed knowing whether the athlete is already overreached, not from the words alone.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Orchestration | LangGraph `StateGraph`, typed `AgentState`, conditional edges for routing and the critic loop |
| Agents | Explicit tool-calling loop (`graph/agent_runner.py`) over 12 tools: 6 read, 6 write |
| LLM | OpenAI GPT-5.4 mini for the agents, GPT-5.4 for the critic; any OpenAI-compatible provider via `LLM_PROVIDER` / `LLM_MODEL` (Groq, Together, OpenRouter, Ollama) |
| Knowledge base | 44 CC BY papers from Europe PMC, licence-gated at download, plus a short practitioner summary — 896 chunks |
| Retrieval | `RecursiveCharacterTextSplitter` → FAISS → MMR (`k=4`, `fetch_k=12`); local `bge-base-en-v1.5` embeddings via fastembed (OpenAI optional) |
| Safety | Deterministic rule engine (`graph/plan_checks.py`) whose caps the LLM judge can't override |
| Memory | `SQLChatMessageHistory` (SQLite) for sessions; JSON files for workouts and plans |
| Observability | Langfuse: one trace per request, spans per agent, tokens, cost and latency per model call |
| UI | Streamlit, five pages with top navigation, custom theme |
| Tests | pytest (113 tests), pytest-cov, two golden-set eval harnesses |
| CI/CD | GitHub Actions: ruff, tests on Python 3.11–3.13, golden-set checks, FAISS build + retrieval tests, Docker smoke test → auto-deploy to Streamlit Community Cloud |
| Container | Docker image (Python 3.12, index baked in), smoke-tested in CI |

---

## Setup

### 1. Clone and install

```bash
git clone https://github.com/Ameya-J16/adaptive-coach.git
cd adaptive-coach
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure environment variables

```bash
cp .env.example .env
```

Edit `.env` and add your keys:

```env
OPENAI_API_KEY=sk-...
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_HOST=https://cloud.langfuse.com
```

> Langfuse is optional — the system degrades gracefully if keys are absent.

### 3. Build the RAG index

```bash
python -m rag.ingest
# 44 papers + practitioner summary → 896 chunks → local embeddings → rag/faiss_index/
```

### 4. Run the website

```bash
streamlit run app.py
```

Opens at http://localhost:8501 with five pages: **Home**, **Coach** (generate a
week, or tell the coach about a niggle or schedule change and let the supervisor
route it to the adaptation agent or the explainer), **Log workout**, **History**
and **Profile**. Seven demo athletes are seeded automatically on first start.
The site's theme lives in `.streamlit/config.toml` and `ui/styles.py`; pages are
in `ui/views/`.

### 5. CLI usage

```bash
python main.py --user-id ameya --action plan     # generate weekly plan
python main.py --user-id ameya --action log      # log a workout
python main.py --user-id ameya --action history  # view history
```

### 6. Run tests

```bash
pip install -r requirements-dev.txt
ruff check .                                   # lint
python -m pytest -q --cov=graph --cov=evals    # 113 tests + coverage (offline, no API keys)
python -m evals.run_eval --offline             # critic golden set, deterministic layer
python -m evals.run_routing_eval --offline     # routing golden set, offline checks
python -m evals.run_eval                       # full live critic eval (calls the LLM)
python -m evals.run_routing_eval               # supervisor routing on 14 labelled requests (calls the LLM)
```

GitHub Actions runs the same checks on every push and pull request
(`.github/workflows/ci.yml`): a ruff lint job, the test suite with coverage
on Python 3.11, 3.12 and 3.13, the offline critic and routing golden-set checks, a job that
builds the FAISS index from the paper corpus and runs the retrieval tests, and a
Docker smoke test that builds the image and waits for the app's health check.

### 7. Deploy (CD)

Every push to `main` that passes all of the checks above is published by the
`deploy` job to the **`deploy` branch**: a single snapshot commit of the code
plus the FAISS index built and tested in the same run. Streamlit Community
Cloud watches that branch and redeploys on every update, so nothing reaches
the live app without passing CI.

One-time setup (free, no tokens needed):

1. Sign in at [share.streamlit.io](https://share.streamlit.io) with GitHub.
2. **Create app** → repository `Ameya-J16/adaptive-coach`, branch **`deploy`**,
   main file `app.py`.
3. **Advanced settings** → Python 3.12, and paste the secrets the app needs
   (TOML), e.g. `OPENAI_API_KEY = "..."`, or `LLM_PROVIDER = "groq"` plus
   `GROQ_API_KEY = "..."`. Leave `EMBEDDING_PROVIDER` unset: the shipped index
   is built with local embeddings.
   To use a different OpenAI model, add e.g. `LLM_MODEL = "gpt-5.4-mini"`
   (reasoning models are supported; optional `LLM_REASONING_EFFORT = "low"`), or
   set one agent's model with e.g. `LLM_MODEL_CRITIC = "gpt-5.4"`.
   For a public deploy also add `DEMO_MODE = "true"`: each visitor then works on
   a private copy of the demo athletes, and gets `MAX_AI_RUNS_PER_SESSION`
   (default 5) model runs per visit so an open demo can't run up the API bill.
   Add `OWNER_TOKEN = "<any long random string>"` too, then open the site once
   as `https://<your-app>.streamlit.app/?owner=<token>`: that browser session
   gets unlimited runs and the links to its Langfuse traces, which visitors
   never see.
4. Deploy. The `deploy` branch appears after the first green push to `main`.

On first start the app seeds the demo personas, since the hosted `data/`
folder starts empty and resets whenever the app restarts.

The same app also runs as a container:

```bash
docker build -t adaptive-coach .
docker run -p 7860:7860 --env-file .env adaptive-coach   # http://localhost:7860
```

---

## Design Decisions

### Agents, not a fixed pipeline

Each specialist gets a short brief and a menu of tools, and decides itself what to
look at — the loop in `graph/agent_runner.py` sends the conversation to the model,
runs whatever tools it asks for, feeds the results back, and stops when it answers.

| Agent | Decides | Tools |
|---|---|---|
| `fatigue_analyst` | How fatigued the athlete is, from effort trends | training-log read tools |
| `supervisor` | Plan, adapt or explain (no tools — one judgement) | — |
| `progression_planner` | Push, hold or deload; writes its own literature searches | read tools + `search_knowledge_base` |
| `plan_writer` | The seven days, starting from logged lifts | read tools |
| `critic` | Whether the plan ships; can check claims against the papers | `search_knowledge_base` |
| `adaptation_agent` | How to change the saved week ("knee hurts", "can't train Thursday") | write tools (`view_day`, `swap_exercise`, `move_session`, `set_day_to_rest`, `adjust_exercise_load`, `add_note_to_day`) + read tools + `search_knowledge_base` |
| `explainer` | How to answer "why?" — read-only by design | read tools + `search_knowledge_base` |

Two steps are plain code on purpose: `context_loader` (loads data) and
`nutrition_advisor` (Mifflin–St Jeor plus macro split — arithmetic doesn't need a model).

### The critic: model judgement, code veto

The critic scores four dimensions — safety, coherence, groundedness, goal alignment —
but `graph/plan_checks.py` checks the mechanical rules first and caps the scores:
back-to-back heavy days on the same muscles, overreach under red fatigue, high RPE in a
deload, a "deload" that doesn't cut volume, an "increase" that doesn't increase, a
swapped-out lift reappearing. Any violation forces the overall score below the 0.75
pass mark, so a broken plan can't be averaged through. Below 0.75 the plan goes back
to `plan_writer` with the feedback, up to three rounds.

The adaptation agent has the same kind of guard: its edits are checked before and
after, and it is rejected only for violations it *introduced* — an agent is
accountable for what it changed, not for what it inherited.

### ACWR — computed in code, not by the model

The acute:chronic workload ratio gates whether a deload is prescribed, so
`fatigue_analyst.py` computes it from the logged volumes (rolling 7- and 28-day
windows) and overwrites whatever the model returns. Red and amber zones are a floor
the agent can't lower; in the green zone it may raise the level from effort signals
(RPE drift at matched loads, sets near failure, per-muscle volume against MEV/MAV).

### RAG — evidence the agents look up themselves

`scripts/build_corpus.py` fetches papers from Europe PMC and keeps only those whose
licence allows redistribution (CC BY / CC BY-SA / CC0); each chunk carries its
author, year and DOI. Agents call `search_knowledge_base` with queries they write
themselves; the planner's results are passed to the writer, shown on the Coach page's
*Research used* tab, and used by the critic to judge groundedness.

---

## Evaluation

Two hand-labelled golden sets (details and every run in
[`evals/results/`](evals/results/)):

| Set | What it measures | Latest (deployed models) |
|---|---|---|
| Critic — 14 plans (6 good, 8 flawed) | Does a bad plan ever pass? Is each flaw caught on the right dimension? | **90% verdict accuracy, 0 false passes**, 100% flaw detection |
| Routing — 14 requests | Does the supervisor send pain to *adapt*, questions to *explain*? | **14/14** |

The one critic miss is a false fail — a good plan rejected for overclaiming its own
notes — never a bad plan let through. Both sets also run offline in CI.

---

## Observability

Every LLM call and node execution is instrumented with Langfuse:

- `LangchainCallbackHandler` captures prompt, completion, latency, and token usage per node
- `trace_node(name)` context manager creates named spans mapping to each pipeline stage
- Critic scores across loop iterations are visible — enabling analysis of how often plans fail quality checks
- `trace_run(...)` wraps each request from the website in one root trace, so a whole
  run (every agent, tool call and LLM generation) shows up as a single tree, tagged
  with the athlete ID. The Coach page links straight to it ("Trace" tab).

**Seeing the traces:** create a free project at [cloud.langfuse.com](https://cloud.langfuse.com),
copy its public and secret keys (*Settings → API Keys*), and add them to `.env`
locally or to the Streamlit app's **Secrets** when deployed:

```toml
LANGFUSE_PUBLIC_KEY = "pk-lf-..."
LANGFUSE_SECRET_KEY = "sk-lf-..."
LANGFUSE_HOST = "https://cloud.langfuse.com"   # or https://us.cloud.langfuse.com for a US project
```

Then open the project's **Tracing** page; each plan or coach request is one trace.

Without tracing, a bad plan output is undebuggable. With Langfuse, every failure is traceable to a specific prompt, node, and model response.


---

## Project Structure

```
adaptive-coach/
├── app.py                      # Streamlit entry: theme, demo seeding, top navigation
├── main.py                     # CLI entry point
├── requirements.txt / requirements-dev.txt
├── Dockerfile · ruff.toml · .streamlit/config.toml
├── .github/workflows/ci.yml    # lint, tests, golden sets, RAG build, Docker, deploy
│
├── graph/
│   ├── graph.py                # StateGraph: supervisor routing + critic loop
│   ├── state.py                # AgentState TypedDict
│   ├── agent_runner.py         # the tool-calling loop every agent uses
│   ├── tools.py                # read tools + search_knowledge_base
│   ├── adaptation_tools.py     # write tools for the adaptation agent
│   ├── plan_checks.py          # deterministic rule engine (critic caps, safety gate)
│   ├── training_metrics.py     # RIR, RPE drift, failure exposure, MEV/MAV
│   └── nodes/                  # context_loader, fatigue_analyst, supervisor,
│                               # progression_planner, nutrition_advisor, plan_writer,
│                               # critic, adaptation_agent, explainer
├── prompts/                    # system + user prompt per agent
├── rag/
│   ├── ingest.py               # chunk → embed → FAISS
│   ├── retriever.py            # MMR search
│   ├── embeddings.py           # local fastembed or OpenAI
│   └── knowledge_base/
│       ├── corpus/             # 44 CC BY papers + manifest.json + LICENSES.md
│       └── sports_science.md   # short practitioner summary
├── config/models.py            # provider/model selection, per-agent overrides
├── memory/                     # session memory (SQLite), workout and plan stores (JSON)
├── tracing/langfuse_config.py  # trace_run / trace_node, silent no-op without keys
├── ui/                         # styles, components, demo guards, pages in ui/views/
├── evals/                      # golden sets, runners, results/
├── scripts/                    # build_corpus, seed_demo_data, list_models, ...
└── tests/                      # 113 tests, incl. end-to-end with a scripted model
```
