"""Preference-acceptance trajectory — outcome metric for the PREFERENCE-skill axis.

Mirrors compute_correctness_trajectory: per-task we already record the
fraction of profile preferences that the FINAL assistant turn did NOT
violate (TaskInteraction.preference_acceptance_rate). Here we just split
the stream into early/late windows and report each window's average,
plus the held-out average.

Reading the result:
  • SkillMap should show late > early if preference-axis skills are
    actually steering the model toward style/format/approach the user
    prefers across tasks.
  • Stateless and declarative_memory baselines should be roughly flat —
    no per-user procedural memory for preferences.
  • Held-out gap (late vs held-out) shows how much of the lift is genuine
    generalization vs in-stream memorization.
"""

from __future__ import annotations

from skillmap_eval.types import (
    ConditionName,
    PreferenceTrajectory,
    StreamRun,
    TaskInteraction,
)


def compute_preference_trajectory(run: StreamRun) -> PreferenceTrajectory:
    stream = list(run.interactions)
    held_out = list(run.held_out_interactions)
    midpoint = len(stream) // 2
    early = stream[:midpoint]
    late = stream[midpoint:]

    return PreferenceTrajectory(
        condition_name=_cast(run.condition_name),
        avg_acceptance_rate_early=_mean_acceptance(early),
        avg_acceptance_rate_late=_mean_acceptance(late),
        avg_acceptance_rate_held_out=_mean_acceptance(held_out),
    )


def _mean_acceptance(interactions: list[TaskInteraction]) -> float:
    rates = [
        i.preference_acceptance_rate
        for i in interactions
        if i.preference_acceptance_rate is not None
    ]
    return (sum(rates) / len(rates)) if rates else 0.0


def _cast(name: str) -> ConditionName:
    if name not in ("stateless", "declarative_memory", "skillmap"):
        raise ValueError(f"unexpected condition: {name!r}")
    return name  # type: ignore[return-value]
