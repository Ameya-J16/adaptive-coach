import json

from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import JsonOutputParser

from graph.state import AgentState
from prompts.critic import CRITIC_PROMPT
from tracing.langfuse_config import trace_node, get_callbacks


def critic(state: AgentState) -> dict:
    with trace_node("critic"):
        llm = ChatOpenAI(model="gpt-4o", temperature=0.0)
        chain = CRITIC_PROMPT | llm | JsonOutputParser()

        result = chain.invoke(
            {
                "user_profile": json.dumps(state.get("user_profile", {}), indent=2),
                "fatigue_report": json.dumps(state.get("fatigue_report", {}), indent=2),
                "progression_decision": json.dumps(state.get("progression_decision", {}), indent=2),
                "weekly_plan": json.dumps(state.get("weekly_plan", {}), indent=2),
            },
            config={"callbacks": get_callbacks()},
        )

        scores = [
            result.get("safety", 0.5),
            result.get("coherence", 0.5),
            result.get("groundedness", 0.5),
            result.get("goal_alignment", 0.5),
        ]
        overall = sum(scores) / len(scores)
        result["overall"] = round(overall, 3)

        new_loop_count = state.get("loop_count", 0) + 1

        return {
            "critic_score": overall,
            "critic_feedback": result.get("feedback", ""),
            "loop_count": new_loop_count,
        }
