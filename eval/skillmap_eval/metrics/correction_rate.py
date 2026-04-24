"""Per-axis correction-rate decay curves over the task stream.

This is the PRIMARY metric for the preference-skill axis (and a useful
secondary metric for the correctness-skill axis). For an effective
SkillMap, all three series should bend down across the stream, with the
preference series bending fastest because preference skills are the most
directly applicable to a "next response" check.
"""

from __future__ import annotations

from skillmap_eval.types import (
    CorrectionRateCurve,
    ConditionName,
    StreamRun,
    TaskInteraction,
)


def compute_correction_curve(run: StreamRun, window: int = 3) -> CorrectionRateCurve:
    indices = list(range(len(run.interactions)))
    total = [i.correction_count for i in run.interactions]
    pref = [i.preference_correction_count for i in run.interactions]
    corr = [i.correctness_correction_count for i in run.interactions]
    return CorrectionRateCurve(
        condition_name=_cast(run.condition_name),
        task_indices=indices,
        total_per_task=total,
        preference_per_task=pref,
        correctness_per_task=corr,
        rolling_mean_window_3_total=_rolling_mean(total, window),
        rolling_mean_window_3_preference=_rolling_mean(pref, window),
        rolling_mean_window_3_correctness=_rolling_mean(corr, window),
    )


def _rolling_mean(xs: list[int], window: int) -> list[float]:
    out: list[float] = []
    for i in range(len(xs)):
        lo = max(0, i - window + 1)
        chunk = xs[lo : i + 1]
        out.append(sum(chunk) / len(chunk) if chunk else 0.0)
    return out


def _cast(name: str) -> ConditionName:
    if name not in ("stateless", "declarative_memory", "skillmap"):
        raise ValueError(f"unexpected condition: {name!r}")
    return name  # type: ignore[return-value]
