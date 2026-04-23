"""Metric 1: correction-rate decay curve over the task stream."""

from __future__ import annotations

from skillmap_eval.types import (
    CorrectionRateCurve,
    ConditionName,
    StreamRun,
)


def compute_correction_curve(run: StreamRun, window: int = 3) -> CorrectionRateCurve:
    indices = list(range(len(run.interactions)))
    counts = [i.correction_count for i in run.interactions]
    rolling = _rolling_mean(counts, window)
    return CorrectionRateCurve(
        condition_name=_cast(run.condition_name),
        task_indices=indices,
        corrections_per_task=counts,
        rolling_mean_window_3=rolling,
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
