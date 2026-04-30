import json

from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import JsonOutputParser

from graph.state import AgentState
from prompts.nutrition_advisor import NUTRITION_ADVISOR_PROMPT
from tracing.langfuse_config import trace_node, get_callbacks


def nutrition_advisor(state: AgentState) -> dict:
    with trace_node("nutrition_advisor"):
        progression_decision = state.get("progression_decision", {})
        user_profile = state.get("user_profile", {})
        current_week_log = state.get("current_week_log", [])

        llm = ChatOpenAI(model="gpt-4o", temperature=0.1)
        chain = NUTRITION_ADVISOR_PROMPT | llm | JsonOutputParser()

        result = chain.invoke(
            {
                "progression_decision": json.dumps(progression_decision, indent=2),
                "user_profile": json.dumps(user_profile, indent=2),
                "current_week_log": json.dumps(current_week_log, indent=2),
            },
            config={"callbacks": get_callbacks()},
        )

        return {"nutrition_targets": result}
