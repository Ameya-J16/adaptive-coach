from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


# ── Logging models ────────────────────────────────────

class SetEntry(BaseModel):
    """A single logged set — weight and reps can differ per set."""
    set_number: int
    reps: int
    weight_kg: float = 0.0
    rpe: float = Field(default=7.0, ge=1.0, le=10.0)


class LoggedExercise(BaseModel):
    """An exercise as logged — each set stored independently."""
    name: str
    sets_data: list[SetEntry] = Field(default_factory=list)
    muscle_groups: list[str] = Field(default_factory=list)
    notes: str = ""


class WorkoutSession(BaseModel):
    date: str  # ISO format: YYYY-MM-DD
    exercises: list[LoggedExercise] = Field(default_factory=list)
    total_volume: float = 0.0  # sum of reps * weight_kg across all sets of all exercises
    avg_rpe: float = Field(default=7.0, ge=1.0, le=10.0)
    session_type: Literal["strength", "cardio", "HIIT", "rest", "deload"]
    notes: str = ""


# ── Plan models (what you should do) ─────────────────────────────────────────

class Exercise(BaseModel):
    """A planned exercise — prescribed as sets/reps/weight targets."""
    name: str
    sets: int
    reps: int
    weight_kg: float = 0.0
    rpe_target: float = Field(default=7.0, ge=1.0, le=10.0)
    muscle_groups: list[str] = Field(default_factory=list)


class DayPlan(BaseModel):
    session_type: str
    exercises: list[Exercise] = Field(default_factory=list)
    notes: str = ""


class WeeklyPlan(BaseModel):
    week_start_date: str
    days: dict[str, DayPlan]  # keys: "monday" through "sunday"
    overall_notes: str = ""


# ── Agent output models ───────────────────────────────────────────────────────

class FatigueReport(BaseModel):
    fatigue_level: Literal["green", "amber", "red"]
    acwr: float = Field(description="Acute:Chronic Workload Ratio")
    dominant_stress_pattern: str
    recommendation_hint: str


class ProgressionDecision(BaseModel):
    action: Literal["increase", "maintain", "deload", "swap"]
    load_adjustment_percent: float = Field(
        description="Percentage to adjust load (positive = increase, negative = decrease)"
    )
    exercises_to_swap: list[str] = Field(default_factory=list)
    rationale: str


class DayNutrition(BaseModel):
    calories: int
    protein_g: int
    carbs_g: int
    fat_g: int


class NutritionTargets(BaseModel):
    """Daily nutrition targets split by training vs rest days."""
    training_day: DayNutrition
    rest_day: DayNutrition
    timing_notes: str


class CriticScore(BaseModel):
    safety: float = Field(ge=0.0, le=1.0)
    coherence: float = Field(ge=0.0, le=1.0)
    groundedness: float = Field(ge=0.0, le=1.0)
    goal_alignment: float = Field(ge=0.0, le=1.0)
    overall: float = Field(ge=0.0, le=1.0)
    feedback: str
