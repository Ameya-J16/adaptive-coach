"""Graph topology.

    START -> context_loader -> fatigue_analyst -> supervisor
                                                     |
                        +----------------------------+----------------------------+
                        | plan                       | adapt                      | explain
                        v                            v                            v
              progression_planner            adaptation_agent                 explainer
                        |                            |                            |
              nutrition_advisor                     END                          END
                        |
                  plan_writer <----+
                        |          | replan (score < 0.75, budget remains)
                     critic -------+
                        |
                       END

context_loader and fatigue_analyst run before the supervisor because routing
needs to know what the athlete's state actually is — a request has to be judged
against real fatigue, not against the words alone. Everything after the
supervisor is chosen by the model rather than by the edge list, which is what
separates this from a fixed pipeline.
"""
from langgraph.graph import StateGraph, START, END

from graph.state import AgentState
from graph.nodes.context_loader import context_loader
from graph.nodes.fatigue_analyst import fatigue_analyst
from graph.nodes.supervisor import supervisor, route_from_supervisor
from graph.nodes.progression_planner import progression_planner
from graph.nodes.nutrition_advisor import nutrition_advisor
from graph.nodes.plan_writer import plan_writer
from graph.nodes.critic import critic
from graph.nodes.adaptation_agent import adaptation_agent
from graph.nodes.explainer import explainer


def should_replan(state: AgentState) -> str:
    """Conditional edge: loop back to plan_writer if score is low and budget remains."""
    if state.get("critic_score", 1.0) < 0.75 and state.get("loop_count", 0) < 3:
        return "replan"
    return "end"


def build_graph() -> StateGraph:
    builder = StateGraph(AgentState)

    builder.add_node("context_loader", context_loader)
    builder.add_node("fatigue_analyst", fatigue_analyst)
    builder.add_node("supervisor", supervisor)
    builder.add_node("progression_planner", progression_planner)
    builder.add_node("nutrition_advisor", nutrition_advisor)
    builder.add_node("plan_writer", plan_writer)
    builder.add_node("critic", critic)
    builder.add_node("adaptation_agent", adaptation_agent)
    builder.add_node("explainer", explainer)

    builder.add_edge(START, "context_loader")
    builder.add_edge("context_loader", "fatigue_analyst")
    builder.add_edge("fatigue_analyst", "supervisor")

    # The supervisor's choice, not the edge list, decides what runs next.
    builder.add_conditional_edges(
        "supervisor",
        route_from_supervisor,
        {
            "plan": "progression_planner",
            "adapt": "adaptation_agent",
            "explain": "explainer",
        },
    )

    builder.add_edge("progression_planner", "nutrition_advisor")
    builder.add_edge("nutrition_advisor", "plan_writer")
    builder.add_edge("plan_writer", "critic")
    builder.add_conditional_edges(
        "critic", should_replan, {"replan": "plan_writer", "end": END}
    )

    builder.add_edge("adaptation_agent", END)
    builder.add_edge("explainer", END)

    return builder.compile()


graph = build_graph()
