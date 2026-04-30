import json

from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import JsonOutputParser

from graph.state import AgentState
from prompts.plan_writer import PLAN_WRITER_PROMPT
from tracing.langfuse_config import trace_node, get_callbacks


def plan_writer(state: AgentState) -> dict:
    with trace_node("plan_writer"):
        llm = ChatOpenAI(model="gpt-4o", temperature=0.3)
        chain = PLAN_WRITER_PROMPT | llm | JsonOutputParser()

        result = chain.invoke(
            {
                "user_profile": json.dumps(state.get("user_profile", {}), indent=2),
                "fatigue_report": json.dumps(state.get("fatigue_report", {}), indent=2),
                "progression_decision": json.dumps(state.get("progression_decision", {}), indent=2),
                "nutrition_targets": json.dumps(state.get("nutrition_targets", {}), indent=2),
                "retrieved_context": state.get("retrieved_context", ""),
                "critic_feedback": state.get("critic_feedback", ""),
            },
            config={"callbacks": get_callbacks()},
        )

        return {
            "weekly_plan": result,
            "loop_count": state.get("loop_count", 0),
        }
