# AdaptiveCoach — Multi-Agent Fitness Intelligence System

> A LangGraph-orchestrated multi-agent system that analyses your training history, reasons about fatigue and progressive overload using RAG-grounded sports science, and autonomously writes an adaptive weekly training plan — with full Langfuse observability.

![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=flat&logo=python&logoColor=white)
![LangGraph](https://img.shields.io/badge/LangGraph-0.2-1C3C3C?style=flat)
![LangChain](https://img.shields.io/badge/LangChain-0.3-1C3C3C?style=flat&logo=langchain&logoColor=white)
![OpenAI](https://img.shields.io/badge/OpenAI-GPT--4o-412991?style=flat&logo=openai&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-UI-FF4B4B?style=flat&logo=streamlit&logoColor=white)
![Langfuse](https://img.shields.io/badge/Langfuse-Observability-000000?style=flat)

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                       AdaptiveCoach Pipeline                        │
└─────────────────────────────────────────────────────────────────────┘

 [User History]        [Session Memory]        [FAISS Index]
      │                      │                      │
      └──────────┬───────────┘                      │
                 ▼                                  │
        ┌────────────────┐                          │
        │ context_loader │                          │
        └───────┬────────┘                          │
                │                                   │
                ▼                                   │
        ┌────────────────┐                          │
        │fatigue_analyst │                          │
        │  ACWR · RPE    │                          │
        └───────┬────────┘                          │
                │                                   │
                ▼                                   │
        ┌───────────────────┐                       │
        │progression_planner│◄──────────────────────┘
        │  RAG retrieval    │   MMR search (k=4)
        └───────┬───────────┘
                │
                ▼
        ┌────────────────┐
        │nutrition_advisor│
        │ TDEE · macros  │
        │ carb periodise │
        └───────┬────────┘
                │
       ┌────────▼────────┐
       │                 │◄─────────────────────────┐
       │   plan_writer   │                          │ critic_feedback
       │  7-day plan     │                          │ (score < 0.75
       │  generation     │                          │  loop_count < 3)
       └────────┬────────┘                          │
                │                                   │
                ▼                                   │
        ┌───────────────┐                           │
        │    critic     │───────────────────────────┘
        │ Safety        │
        │ Coherence     │
        │ Groundedness  │──── score ≥ 0.75 ──► [Final Plan]
        │ Goal Alignment│     OR loop = 3
        └───────────────┘

  Every node  ──►  Langfuse (named spans + LangchainCallbackHandler)
  Every call  ──►  ChatPromptTemplate chain (no bare f-strings)
  All outputs ──►  Pydantic v2 / JsonOutputParser
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| Orchestration | LangGraph `StateGraph` with typed `AgentState` |
| LLM | OpenAI GPT-4o (temperature tuned per node) |
| Embeddings | `text-embedding-3-small` → FAISS |
| RAG | `RecursiveCharacterTextSplitter` → MMR retrieval |
| Prompts | `ChatPromptTemplate` per node (no f-strings) |
| Memory | `SQLChatMessageHistory` (SQLite) |
| Workout Storage | JSON flat-file keyed by `user_id` |
| Observability | Langfuse (spans, token counts, latency) |
| Schemas | Pydantic v2 |
| UI | Streamlit (4 tabs, Hevy-style per-set logging) |
| Tests | pytest |

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
# Chunks sports_science.md → embeds → saves to rag/faiss_index/
```

### 4. Run the Streamlit app

```bash
streamlit run app.py
```

### 5. CLI usage

```bash
python main.py --user-id ameya --action plan     # generate weekly plan
python main.py --user-id ameya --action log      # log a workout
python main.py --user-id ameya --action history  # view history
```

### 6. Run tests

```bash
pytest tests/ -v
```

---

## Agentic Design Decisions

### Why LangGraph?

Standard LangChain chains are DAGs — they run once and return. AdaptiveCoach requires a **stateful, looping pipeline** where the Critic node can evaluate plan quality and route execution back to the PlanWriter for revision.

- **Typed state** (`AgentState` TypedDict) — every node knows exactly what fields are available and what it must produce
- **Conditional edges** (`should_replan`) — if `critic_score < 0.75` and `loop_count < 3`, the graph re-routes to `plan_writer` with critic feedback injected into the prompt
- **Node isolation** — each node is a pure function receiving state and returning partial state updates, making individual nodes unit-testable in isolation

### The Critic Loop

Without a critic, LLM-generated plans may be internally inconsistent. The critic enforces 4 quality dimensions:

| Dimension | What it checks |
|---|---|
| **Safety** | No consecutive heavy compound sessions; deload rules respected |
| **Coherence** | Plan matches the progression decision (increase / maintain / deload) |
| **Groundedness** | Advice references retrieved sports science, not hallucinated |
| **Goal Alignment** | Rep ranges and intensities match the user's stated goal |

The system self-corrects up to 3 times before emitting the final plan.

### RAG — How Plans Stay Grounded

The `progression_planner` builds a targeted query from the fatigue report and user goals (e.g. `"deload protocol ACWR 1.7 overreaching recovery"`), retrieves 4 MMR-reranked chunks from FAISS, and:

1. Injects them into the `progression_planner` prompt
2. Stores them in `AgentState.retrieved_context` for `plan_writer`
3. Displays sources transparently in the Streamlit UI
4. Uses them as evidence in the `critic` Groundedness score

### ACWR — Computed in Python, Not by the LLM

The Acute:Chronic Workload Ratio is a safety-critical metric. Computing it inside the LLM risks hallucination on a number that directly gates whether the system prescribes a deload.

`fatigue_analyst.py` computes ACWR deterministically in Python from the stored workout volumes, then **overwrites** whatever the LLM returns with the real value.

---

## Observability

Every LLM call and node execution is instrumented with Langfuse:

- `LangchainCallbackHandler` captures prompt, completion, latency, and token usage per node
- `trace_node(name)` context manager creates named spans mapping to each pipeline stage
- Critic scores across loop iterations are visible — enabling analysis of how often plans fail quality checks

Without tracing, a bad plan output is undebuggable. With Langfuse, every failure is traceable to a specific prompt, node, and model response.

---

## Project Structure

```
adaptive_coach/
├── app.py                     # Streamlit UI (4 tabs, Hevy-style logging)
├── main.py                    # CLI entry point
├── requirements.txt
├── .env.example
│
├── graph/
│   ├── state.py               # AgentState TypedDict (13 fields)
│   ├── graph.py               # StateGraph + conditional critic loop
│   └── nodes/
│       ├── context_loader.py
│       ├── fatigue_analyst.py # Deterministic ACWR + LLM trend analysis
│       ├── progression_planner.py  # RAG retrieval hub
│       ├── nutrition_advisor.py    # TDEE + carb periodisation
│       ├── plan_writer.py
│       └── critic.py
│
├── rag/
│   ├── ingest.py              # Chunk → embed → FAISS
│   ├── retriever.py           # MMR search wrapper (lazy singleton)
│   └── knowledge_base/
│       └── sports_science.md  # 1,800+ word evidence base
│
├── prompts/                   # ChatPromptTemplate per node
├── models/
│   └── schemas.py             # Pydantic v2 (SetEntry, LoggedExercise, etc.)
├── memory/
│   ├── session_memory.py      # SQLChatMessageHistory (SQLite)
│   └── workout_store.py       # JSON workout log (per-set storage)
├── tracing/
│   └── langfuse_config.py     # Langfuse client + graceful degradation
└── tests/
    ├── test_fatigue_analyst.py
    ├── test_critic_loop.py
    └── test_rag_retriever.py
```
