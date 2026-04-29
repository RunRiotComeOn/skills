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
        avg_acceptance_rate_early=mean_preference_acceptance_rate(early)[0],
        avg_acceptance_rate_late=mean_preference_acceptance_rate(late)[0],
        avg_acceptance_rate_held_out=mean_preference_acceptance_rate(held_out)[0],
    )


def mean_preference_acceptance_rate(
    interactions: list[TaskInteraction],
) -> tuple[float, int]:
    """Average preference_acceptance_rate over a list of interactions.

    Tasks with `preference_acceptance_rate is None` (no assistant turn,
    or empty preference profile) are dropped — counted via the second
    element of the returned tuple.
    """
    rates = [
        i.preference_acceptance_rate
        for i in interactions
        if i.preference_acceptance_rate is not None
    ]
    return ((sum(rates) / len(rates)) if rates else 0.0, len(rates))


def mean_first_turn_preference_acceptance_rate(
    interactions: list[TaskInteraction],
) -> tuple[float, int]:
    """Average first_turn_preference_acceptance_rate over interactions.

    This is the "did the model get it right on the first try" metric. It
    is the one that should actually move when memory is doing its job —
    the legacy `preference_acceptance_rate` (final assistant turn) tends
    to saturate near 1.0 because the model almost always recovers after
    a correction or two.

    Tasks with a None rate are dropped; the second tuple element reports
    how many interactions contributed.
    """
    rates = [
        i.first_turn_preference_acceptance_rate
        for i in interactions
        if i.first_turn_preference_acceptance_rate is not None
    ]
    return ((sum(rates) / len(rates)) if rates else 0.0, len(rates))


def mean_first_turn_preference_violation_count(
    interactions: list[TaskInteraction],
) -> tuple[float, int]:
    """Average count of preferences violated in the first assistant turn.

    Mirror of `mean_first_turn_preference_acceptance_rate` in absolute
    units. Useful when feeding into a rolling-mean stream curve, where a
    fraction can wash out (the profile size is fixed) and an integer
    count preserves the shape of the decay.
    """
    counts = [
        i.first_turn_preference_violation_count
        for i in interactions
        if i.first_turn_preference_violation_count is not None
    ]
    return (
        (sum(counts) / len(counts)) if counts else 0.0,
        len(counts),
    )


def _cast(name: str) -> ConditionName:
    if name not in ("stateless", "declarative_memory", "skillmap"):
        raise ValueError(f"unexpected condition: {name!r}")
    return name  # type: ignore[return-value]
