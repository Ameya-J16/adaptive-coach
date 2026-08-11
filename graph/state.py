from typing import TypedDict


class AgentState(TypedDict):
    user_id: str
    user_profile: dict           # goals, experience level, available days, equipment
    workout_history: list[dict]  # logged workouts covering the ACWR chronic window
    current_week_log: list[dict] # this week's completed sessions
    fatigue_report: dict         # output of FatigueAnalyst
    retrieved_context: str       # RAG results used for grounding
    progression_decision: dict   # output of ProgressionPlanner
    nutrition_targets: dict      # output of NutritionAdvisor
    weekly_plan: dict            # output of PlanWriter
    critic_score: float          # 0.0 - 1.0
    critic_feedback: str         # natural language feedback from critic
    loop_count: int              # tracks how many critic loops have run
    messages: list               # conversation history for session memory
    user_request: str            # free-text request; empty means "just plan my week"
    route: str                   # supervisor's chosen path: plan | adapt | explain
    route_reason: str            # why the supervisor chose it (shown in the UI)
    adaptation_result: dict      # output of the adaptation agent
    explanation: str             # output of the explain path
