# AdaptiveCoach — Multi-Agent Fitness Intelligence System

AdaptiveCoach is a LangGraph-orchestrated multi-agent system that analyses your training history, reasons about fatigue and progressive overload using RAG-grounded sports science, and autonomously writes an adaptive weekly training plan with full Langfuse observability.

---

## Architecture

```
                        ┌─────────────────────────────────────────────────────┐
                        │                  AdaptiveCoach Pipeline              │
                        └─────────────────────────────────────────────────────┘

  [User History]                                                   [FAISS Index]
  [Session Memory]                                                 [sports_science.md]
       │                                                                  │
       ▼                                                                  │
┌─────────────┐    ┌──────────────────┐    ┌──────────────────────┐      │
│             │    │                  │    │                        │◄─────┘
│  context_   │───►│  fatigue_analyst │───►│  progression_planner  │
│  loader     │    │                  │    │   (RAG retrieval here) │
│             │    │  ACWR + RPE      │    │                        │
└─────────────┘    │  trend analysis  │    └──────────┬─────────────┘
                   └──────────────────┘               │
                                                       ▼
                                          ┌──────────────────────┐
                                          │   nutrition_advisor   │
                                          │   TDEE + macros       │
                                          └──────────┬───────────┘
                                                     │
                                                     ▼
                                          ┌──────────────────────┐
                         ┌───────────────►│     plan_writer       │
                         │               │   7-day structured     │
                         │               │   plan generation      │
                         │               └──────────┬─────────────┘
                         │                          │
                         │  critic_feedback         ▼
                         │  (if score < 0.75) ┌──────────────────────┐
                         └────────────────────│       critic          │
                              loop_count < 3  │   Safety · Coherence  │
                                              │   Groundedness · Goals│
                                              └──────────┬─────────────┘
                                                         │
                                                    score ≥ 0.75
                                                  OR loop_count = 3
                                                         │
                                                         ▼
                                                   [Final Plan]

  Every node ──► Langfuse (named spans + LangchainCallbackHandler)
  Every LLM call ──► ChatPromptTemplate chain (no bare f-strings)
  Structured outputs ──► PydanticOutputParser / JsonOutputParser
```

---

## Setup

### 1. Clone and install

```bash
git clone <repo-url>
cd adaptive_coach
pip install -r requirements.txt
# or: uv pip install -r requirements.txt
```

### 2. Set environment variables

```bash
cp .env.example .env
# Edit .env and add your keys:
# OPENAI_API_KEY=sk-...
# LANGFUSE_PUBLIC_KEY=pk-lf-...
# LANGFUSE_SECRET_KEY=sk-lf-...
# LANGFUSE_HOST=https://cloud.langfuse.com
```

### 3. Build the RAG index

```bash
python -m rag.ingest
# Creates rag/faiss_index/ with the embedded sports science knowledge base
```

### 4. Run the CLI

```bash
# Generate an adaptive plan
python main.py --user-id ameya --action plan

# Log a workout
python main.py --user-id ameya --action log

# View history
python main.py --user-id ameya --action history
```

### 5. Run the Streamlit app

```bash
streamlit run app.py
```

### 6. Run tests

```bash
pytest tests/ -v
```

---

## Agentic Design Decisions

### Why LangGraph?

Standard LangChain chains are DAGs — they run once and return. AdaptiveCoach requires a **stateful, looping pipeline** where the Critic node can evaluate the plan quality and route execution back to the PlanWriter for revision. LangGraph's `StateGraph` with typed state (`AgentState`) and conditional edges makes this agentic loop explicit, inspectable, and debuggable.

Specifically:
- **Typed state** (`AgentState` TypedDict) ensures every node knows exactly what fields are available and what it must produce — no implicit data passing.
- **Conditional edges** (`should_replan`) implement the critic feedback loop cleanly: if score < 0.75 and budget remains, the graph re-routes to `plan_writer` with the critic's feedback injected into the prompt.
- **Node isolation** — each node is a pure function that receives state and returns partial state updates. This makes individual nodes unit-testable in isolation.

### What the Critic Loop Solves

Without a critic, LLM-generated plans may be internally inconsistent: a plan might be labelled "deload" while still prescribing RPE 9 sessions, or claim groundedness while ignoring the retrieved sports science context entirely. The critic enforces 4 distinct quality dimensions:

1. **Safety** — detects consecutive heavy compound sessions, deload violations
2. **Coherence** — verifies the plan matches the progression decision action
3. **Groundedness** — ensures the plan references retrieved knowledge, not hallucinated advice
4. **Goal alignment** — checks that rep ranges and intensities match the user's stated goal

By looping up to 3 times with specific actionable feedback, the system self-corrects without human intervention — a core property of agentic behaviour.

### How RAG Grounds the Plan

The `progression_planner` node builds a targeted query from the fatigue report and user goals (e.g., "deload protocol ACWR 1.7 overreaching recovery") and retrieves 4 chunks from the FAISS index with MMR reranking to reduce redundancy.

This retrieved context is:
1. Passed directly into the `progression_planner` prompt to ground the decision
2. Stored in `AgentState.retrieved_context` and passed to `plan_writer`
3. Displayed transparently in the Streamlit UI ("Sports Science Sources Used")
4. Evaluated by the `critic` as part of the Groundedness score

This prevents the LLM from hallucinating training advice and ensures every adaptation decision can be traced back to specific evidence.

---

## Observability

AdaptiveCoach instruments every LLM call and node execution with Langfuse:

- **LangchainCallbackHandler** is passed into every `chain.invoke()` call, capturing inputs, outputs, latency, and token usage for each node.
- **Named spans** via `trace_node(name)` context manager wrap each node's execution, creating a hierarchical trace that maps directly to the LangGraph pipeline stages.
- **What Langfuse traces:**
  - Full prompt + completion for each node (fatigue analysis, planner, nutrition, writer, critic)
  - Token counts and cost per node
  - Latency per node and total pipeline latency
  - Loop iterations (loop_count visible in state at each trace)
  - Critic scores across iterations — enabling analysis of how often plans fail quality checks

This matters because multi-agent systems are opaque by default. Without tracing, a bad plan output is undebuggable — you cannot tell which node produced incorrect reasoning. With Langfuse, every failure is traceable to a specific prompt, a specific node, and a specific model response.

---

## Project Structure

```
adaptive_coach/
├── main.py                    # CLI entry point
├── app.py                     # Streamlit UI (4 tabs)
├── requirements.txt
├── .env.example
├── README.md
├── graph/
│   ├── state.py               # AgentState TypedDict
│   ├── graph.py               # StateGraph with conditional critic loop
│   └── nodes/                 # One file per agent node
├── rag/
│   ├── ingest.py              # Chunk → embed → FAISS
│   ├── retriever.py           # MMR search wrapper
│   └── knowledge_base/
│       └── sports_science.md  # 1800+ word evidence base
├── prompts/                   # ChatPromptTemplate per node
├── memory/
│   ├── session_memory.py      # SQLChatMessageHistory (SQLite)
│   └── workout_store.py       # JSON-based workout log persistence
├── tracing/
│   └── langfuse_config.py     # Langfuse client + callbacks
├── models/
│   └── schemas.py             # Pydantic v2 models
└── tests/                     # pytest suite
```
