"""Eval-specific Pydantic data models. See spec §2.

Two-axis correction tracking
----------------------------
The simulator decides to "correct" when EITHER a ground-truth preference is
violated OR the assistant's code failed tests. Each correction therefore
carries a `correction_axes` list — typically one of:

  ["preference"]                — user only objected to style/format/approach
  ["correctness"]               — user only objected because code was wrong
  ["preference", "correctness"] — both (e.g., wrong code AND wrong style)

Per-axis counts on TaskInteraction (preference_correction_count,
correctness_correction_count) feed the two parallel learning curves the
eval reports. Together they mirror the two-axis skill model in
skillmap/types.py.
"""

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

CorrectionAxis = Literal["preference", "correctness"]


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
    # User turns issued during a "correct" decision: which axes the user
    # was correcting on. May contain "preference", "correctness", or both.
    # Empty for the initial task message and for accept/give-up turns.
    correction_axes: list[CorrectionAxis] = Field(default_factory=list)
    # Assistant turns: pref IDs judged violated (post-hoc).
    violated_preferences: list[str] = Field(default_factory=list)


class TaskInteraction(BaseModel):
    task_id: str
    condition_name: ConditionName
    task_index_in_stream: int
    turns: list[SimulatedTurn]
    # Total user-correction turns this task. Equals the sum over user turns
    # whose correction_axes is non-empty. Kept for backward compat with the
    # legacy single-axis curve.
    correction_count: int
    # Per-axis breakdowns. A correction tagged with both axes increments
    # BOTH counters (so preference + correctness can exceed correction_count).
    preference_correction_count: int = 0
    correctness_correction_count: int = 0
    completion_reason: CompletionReason
    test_case_pass_rate: Optional[float] = None
    # Fraction of profile preferences NOT violated in the FINAL assistant
    # turn of the task. Parallel to test_case_pass_rate on the preference
    # axis: 1.0 means the delivered response respected every preference,
    # 0.0 means it violated all of them. None when there is no assistant
    # turn or the profile has no preferences.
    preference_acceptance_rate: Optional[float] = None
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
    """Stream-level decay curves split by axis.

    Reading the report:
      • `preference_per_task`  — primary metric for the preference-skill axis.
                                 SkillMap should bend this curve down faster
                                 than declarative_memory or stateless.
      • `correctness_per_task` — primary correction-side metric for the
                                 correctness-skill axis. Should also decay
                                 if correctness skills are firing.
      • `total_per_task`       — legacy combined view. May exceed the sum of
                                 the two above when a single correction is
                                 tagged with both axes.
    """
    condition_name: ConditionName
    task_indices: list[int]
    total_per_task: list[int]
    preference_per_task: list[int]
    correctness_per_task: list[int]
    rolling_mean_window_3_total: list[float]
    rolling_mean_window_3_preference: list[float]
    rolling_mean_window_3_correctness: list[float]


class PreferenceRecoveryResult(BaseModel):
    """LLM-judge recovery rate, preference axis only.

    For each ground-truth preference in the user profile, did SkillMap
    induce a skill that captures it? This is the secondary metric for
    the preference-skill axis (the primary being correction-rate decay).
    """
    condition_name: ConditionName
    total_preferences: int
    recovered_preferences: list[str]
    recovery_rate: float
    judge_reasoning: dict[str, str]


class GeneralizationResult(BaseModel):
    """Held-out generalization, split by axis.

    The interesting comparison is whether SkillMap's held-out per-axis
    correction rate looks more like its LATE-stream rate (skills generalize)
    or its EARLY-stream rate (skills only memorized seen tasks).
    """
    condition_name: ConditionName
    held_out_avg_correction_count: float
    held_out_median_correction_count: float
    held_out_avg_preference_corrections: float
    held_out_avg_correctness_corrections: float


class CorrectnessTrajectory(BaseModel):
    """Primary metric for the CORRECTNESS-skill axis.

    Compares average test-case pass rate in the EARLY half of the task
    stream vs the LATE half. A SkillMap that successfully induces useful
    correctness skills should show late > early; a stateless baseline
    should be flat. Held-out pass rate is reported separately and is
    the cleanest signal of generalization (no memorization possible).
    """
    condition_name: ConditionName
    avg_pass_rate_early: float       # first half of stream
    avg_pass_rate_late: float        # second half of stream
    avg_pass_rate_held_out: float
    task_completion_rate: float


class PreferenceTrajectory(BaseModel):
    """Primary outcome metric for the PREFERENCE-skill axis.

    Mirrors CorrectnessTrajectory: averages the per-task preference
    acceptance rate (fraction of profile preferences not violated by the
    final assistant turn) across the EARLY half of the stream, the LATE
    half, and the held-out set. A SkillMap that successfully induces
    preference-axis skills should show late > early; stateless should
    be roughly flat. Held-out is the cleanest generalization signal.
    """
    condition_name: ConditionName
    avg_acceptance_rate_early: float
    avg_acceptance_rate_late: float
    avg_acceptance_rate_held_out: float


class EvalReport(BaseModel):
    profile_id: str
    completed_at: datetime
    correction_curves: list[CorrectionRateCurve]
    preference_recovery: list[PreferenceRecoveryResult]
    generalization: list[GeneralizationResult]
    correctness_trajectory: list[CorrectnessTrajectory]
    preference_trajectory: list[PreferenceTrajectory] = Field(default_factory=list)
