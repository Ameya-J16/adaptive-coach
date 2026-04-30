import json

from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import JsonOutputParser

from graph.state import AgentState
from prompts.progression_planner import PROGRESSION_PLANNER_PROMPT
from rag.retriever import retrieve
from tracing.langfuse_config import trace_node, get_callbacks


def _build_rag_query(fatigue_report: dict, user_profile: dict) -> str:
    fatigue_level = fatigue_report.get("fatigue_level", "green")
    goals = user_profile.get("goals", "strength")
    acwr = fatigue_report.get("acwr", 1.0)

    if fatigue_level == "red":
        return f"deload protocol ACWR {acwr} overreaching recovery"
    elif fatigue_level == "amber":
        return f"maintain training load exercise substitution fatigue management {goals}"
    else:
        return f"progressive overload principles {goals} volume increase periodisation"


def progression_planner(state: AgentState) -> dict:
    with trace_node("progression_planner"):
        fatigue_report = state.get("fatigue_report", {})
        user_profile = state.get("user_profile", {})
        workout_history = state.get("workout_history", [])

        rag_query = _build_rag_query(fatigue_report, user_profile)
        retrieved_context = retrieve(rag_query, k=4)

        llm = ChatOpenAI(model="gpt-4o", temperature=0.2)
        chain = PROGRESSION_PLANNER_PROMPT | llm | JsonOutputParser()

        result = chain.invoke(
            {
                "fatigue_report": json.dumps(fatigue_report, indent=2),
                "user_profile": json.dumps(user_profile, indent=2),
                "workout_history": json.dumps(workout_history, indent=2),
                "retrieved_context": retrieved_context,
            },
            config={"callbacks": get_callbacks()},
        )

        return {
            "progression_decision": result,
            "retrieved_context": retrieved_context,
        }
