"""Held-out generalization, split by axis.

The cleanest test of whether the induced skills generalize: run the
finalized SkillMap (no further updates) on a held-out task set and
compare per-axis correction counts. Compare these against the LATE-stream
average (skillmap should be at parity → skills generalize) and the
EARLY-stream average (memorization would show the held-out average closer
to early than to late).
"""

from __future__ import annotations

import statistics

from skillmap_eval.types import (
    ConditionName,
    GeneralizationResult,
    StreamRun,
)


def compute_generalization(run: StreamRun) -> GeneralizationResult:
    interactions = run.held_out_interactions
    counts = [i.correction_count for i in interactions]
    pref_counts = [i.preference_correction_count for i in interactions]
    corr_counts = [i.correctness_correction_count for i in interactions]
    avg = (sum(counts) / len(counts)) if counts else 0.0
    med = statistics.median(counts) if counts else 0.0
    pref_avg = (sum(pref_counts) / len(pref_counts)) if pref_counts else 0.0
    corr_avg = (sum(corr_counts) / len(corr_counts)) if corr_counts else 0.0
    return GeneralizationResult(
        condition_name=_cast(run.condition_name),
        held_out_avg_correction_count=avg,
        held_out_median_correction_count=float(med),
        held_out_avg_preference_corrections=pref_avg,
        held_out_avg_correctness_corrections=corr_avg,
    )


def _cast(name: str) -> ConditionName:
    if name not in ("stateless", "declarative_memory", "skillmap"):
        raise ValueError(f"unexpected condition: {name!r}")
    return name  # type: ignore[return-value]
