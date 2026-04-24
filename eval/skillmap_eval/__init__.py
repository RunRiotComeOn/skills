"""Evaluation harness for SkillMap. Import-only consumer of skillmap/."""

from skillmap_eval.types import (
    CorrectionAxis,
    Preference,
    PreferenceProfile,
    EvalTask,
    SimulatedTurn,
    TaskInteraction,
    StreamRun,
    CorrectionRateCurve,
    PreferenceRecoveryResult,
    GeneralizationResult,
    CorrectnessTrajectory,
    EvalReport,
)

__all__ = [
    "CorrectionAxis",
    "Preference",
    "PreferenceProfile",
    "EvalTask",
    "SimulatedTurn",
    "TaskInteraction",
    "StreamRun",
    "CorrectionRateCurve",
    "PreferenceRecoveryResult",
    "GeneralizationResult",
    "CorrectnessTrajectory",
    "EvalReport",
]
