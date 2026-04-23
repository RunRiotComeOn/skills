"""Eval-specific Pydantic data models. See spec §2."""

from __future__ import annotations

from datetime import datetime
from typing import Literal, Optional

from pydantic import BaseModel, Field


PreferenceCategory = Literal[
    "communication_style",
    "diagnostic_approach",
    "code_style",
    "tool_usage",
    "detail_level",
]

ConditionName = Literal["stateless", "declarative_memory", "skillmap"]

CompletionReason = Literal["user_accepted", "max_turns_exceeded", "user_gave_up"]

SimulatorAction = Literal["correct", "accept", "give_up"]


# ---------- Preference (LLM-A's ground truth) ----------

class Preference(BaseModel):
    id: str                           # "pref_01"
    description: str                  # opinionated, non-default preference
    priority: int                     # 1 = strongest, higher = weaker
    expected_correction_trigger: str  # how LLM-A will phrase the correction
    category: PreferenceCategory


class PreferenceProfile(BaseModel):
    profile_id: str                   # "profile_coding_v1"
    generator_model: str              # which LLM-A produced this
    task_type: str                    # "python_coding"
    preferences: list[Preference]     # length 8-12
    created_at: datetime


# ---------- Task ----------

class EvalTask(BaseModel):
    task_id: str
    source: Literal["livecodebench"]
    problem_statement: str
    reference_solution: Optional[str] = None
    test_cases: list[dict] = Field(default_factory=list)
    difficulty: Literal["easy", "medium", "hard"]
    contest_date: Optional[str] = None


# ---------- Interaction artifacts ----------

class SimulatedTurn(BaseModel):
    role: Literal["user", "assistant"]
    content: str
    # User turns: which pref IDs the user invoked in this correction.
    triggered_preferences: list[str] = Field(default_factory=list)
    # Assistant turns: pref IDs judged violated (post-hoc).
    violated_preferences: list[str] = Field(default_factory=list)


class TaskInteraction(BaseModel):
    task_id: str
    condition_name: ConditionName
    task_index_in_stream: int
    turns: list[SimulatedTurn]
    correction_count: int
    completion_reason: CompletionReason
    test_case_pass_rate: Optional[float] = None
    retrieved_skill_ids_at_start: list[str] = Field(default_factory=list)


# ---------- Stream & run ----------

class StreamRun(BaseModel):
    run_id: str
    profile_id: str
    condition_name: ConditionName
    task_stream: list[str]
    held_out_task_ids: list[str]
    interactions: list[TaskInteraction] = Field(default_factory=list)
    held_out_interactions: list[TaskInteraction] = Field(default_factory=list)
    started_at: datetime
    completed_at: Optional[datetime] = None


# ---------- Metric outputs ----------

class CorrectionRateCurve(BaseModel):
    condition_name: ConditionName
    task_indices: list[int]
    corrections_per_task: list[int]
    rolling_mean_window_3: list[float]


class PreferenceRecoveryResult(BaseModel):
    condition_name: ConditionName
    total_preferences: int
    recovered_preferences: list[str]
    recovery_rate: float
    judge_reasoning: dict[str, str]


class GeneralizationResult(BaseModel):
    condition_name: ConditionName
    held_out_avg_correction_count: float
    held_out_median_correction_count: float


class CorrectnessSanity(BaseModel):
    condition_name: ConditionName
    avg_test_pass_rate: float
    task_completion_rate: float


class EvalReport(BaseModel):
    profile_id: str
    completed_at: datetime
    correction_curves: list[CorrectionRateCurve]
    preference_recovery: list[PreferenceRecoveryResult]
    generalization: list[GeneralizationResult]
    correctness_sanity: list[CorrectnessSanity]
