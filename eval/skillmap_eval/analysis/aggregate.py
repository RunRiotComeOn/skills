"""Aggregate per-condition StreamRuns into the top-level EvalReport."""

from __future__ import annotations

from datetime import datetime, timezone

from skillmap_eval.metrics import (
    compute_correction_curve,
    compute_correctness_sanity,
    compute_generalization,
)
from skillmap_eval.types import (
    EvalReport,
    PreferenceRecoveryResult,
    StreamRun,
)


def aggregate_runs(
    profile_id: str,
    runs: list[StreamRun],
    preference_recovery: list[PreferenceRecoveryResult],
) -> EvalReport:
    return EvalReport(
        profile_id=profile_id,
        completed_at=datetime.now(timezone.utc),
        correction_curves=[compute_correction_curve(r) for r in runs],
        preference_recovery=preference_recovery,
        generalization=[compute_generalization(r) for r in runs],
        correctness_sanity=[compute_correctness_sanity(r) for r in runs],
    )
