from langgraph.graph import StateGraph, START, END

from graph.state import AgentState
from graph.nodes.context_loader import context_loader
from graph.nodes.fatigue_analyst import fatigue_analyst
from graph.nodes.progression_planner import progression_planner
from graph.nodes.nutrition_advisor import nutrition_advisor
from graph.nodes.plan_writer import plan_writer
from graph.nodes.critic import critic


def should_replan(state: AgentState) -> str:
    """Conditional edge: loop back to plan_writer if score is low and budget remains."""
    if state.get("critic_score", 1.0) < 0.75 and state.get("loop_count", 0) < 3:
        return "replan"
    return "end"


def build_graph() -> StateGraph:
    builder = StateGraph(AgentState)

    builder.add_node("context_loader", context_loader)
    builder.add_node("fatigue_analyst", fatigue_analyst)
    builder.add_node("progression_planner", progression_planner)
    builder.add_node("nutrition_advisor", nutrition_advisor)
    builder.add_node("plan_writer", plan_writer)
    builder.add_node("critic", critic)

    builder.add_edge(START, "context_loader")
    builder.add_edge("context_loader", "fatigue_analyst")
    builder.add_edge("fatigue_analyst", "progression_planner")
    builder.add_edge("progression_planner", "nutrition_advisor")
    builder.add_edge("nutrition_advisor", "plan_writer")
    builder.add_edge("plan_writer", "critic")

    builder.add_conditional_edges(
        "critic",
        should_replan,
        {"replan": "plan_writer", "end": END},
    )

    return builder.compile()


graph = build_graph()
