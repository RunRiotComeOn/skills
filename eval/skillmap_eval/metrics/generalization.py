"""Metric 3: held-out correction rate per condition.

Key comparison: does SkillMap's held-out correction rate look like its
LATE-stream rate (generalization) or its EARLY-stream rate (memorization)?
"""

from __future__ import annotations

import statistics

from skillmap_eval.types import (
    ConditionName,
    GeneralizationResult,
    StreamRun,
)


def compute_generalization(run: StreamRun) -> GeneralizationResult:
    counts = [i.correction_count for i in run.held_out_interactions]
    avg = (sum(counts) / len(counts)) if counts else 0.0
    med = statistics.median(counts) if counts else 0.0
    return GeneralizationResult(
        condition_name=_cast(run.condition_name),
        held_out_avg_correction_count=avg,
        held_out_median_correction_count=float(med),
    )


def _cast(name: str) -> ConditionName:
    if name not in ("stateless", "declarative_memory", "skillmap"):
        raise ValueError(f"unexpected condition: {name!r}")
    return name  # type: ignore[return-value]
