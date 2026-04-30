from typing import TypedDict


class AgentState(TypedDict):
    user_id: str
    user_profile: dict           # goals, experience level, available days, equipment
    workout_history: list[dict]  # last 4 weeks of logged workouts
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
