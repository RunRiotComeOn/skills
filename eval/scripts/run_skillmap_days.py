"""Run an incremental day-batched eval across one or more conditions.

Usage:
    env PYTHONPATH=. python eval/scripts/run_skillmap_days.py \
        --profile haiku_minimax_livecodebench_v1 \
        --conditions stateless,declarative_memory,skillmap

Behavior:
  - If the profile does not exist, generate it first with Stage 1.
  - Sample disjoint LiveCodeBench tasks with a fixed per-day difficulty mix.
  - Run EACH selected condition over the SAME task stream so trajectories
    are directly comparable across conditions.
  - Reuse condition state across days (SkillMap via skill_map.json,
    DeclarativeMemory via declarative_memory.json, Stateless trivially).
  - Persist a per-day-per-condition stream JSON plus an aggregate summary
    JSON. The summary reports per-condition cross-day rolling-mean
    correction-rate curves so the relative slopes can be eyeballed.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from skillmap_eval.conditions import (
    DeclarativeMemoryCondition,
    SkillMapCondition,
    StatelessCondition,
)
from skillmap_eval.conditions.base import Condition
from skillmap_eval.metrics import (
    mean_first_turn_preference_acceptance_rate,
    mean_first_turn_preference_violation_count,
    mean_preference_acceptance_rate,
    mean_test_case_pass_rate,
    mean_test_case_pass_rate_strict,
    task_completion_rate,
)
from skillmap_eval.preferences import PreferenceElicitor
from skillmap_eval.runner import InteractionLoop
from skillmap_eval.simulator import UserSimulator
from skillmap_eval.tasks import AIMELoader, LiveCodeBenchLoader
from skillmap_eval.types import EvalTask, PreferenceProfile, StreamRun, TaskInteraction


SUPPORTED_CONDITIONS = ("stateless", "declarative_memory", "skillmap")


@dataclass
class DaySummary:
    day_index: int
    condition: str
    n_tasks: int
    task_ids: list[str]
    total_corrections: int
    avg_corrections_per_task: float
    avg_preference_corrections_per_task: float
    avg_correctness_corrections_per_task: float
    # Fraction of tasks the user ended by accepting (completion_reason).
    # NOT preference acceptance — see avg_preference_acceptance_rate below.
    task_completion_rate: float
    # Per-axis outcome metrics (parallel to EvalReport's two trajectories).
    avg_test_case_pass_rate_measurable: float
    avg_test_case_pass_rate_strict: float
    n_with_test_case_pass_rate: int
    # Legacy "did the FINAL turn comply with all prefs?" — saturates near 1.0
    # because the model almost always recovers after a correction or two.
    # Kept for continuity with prior runs but read it alongside the
    # discriminative first-turn metric below.
    avg_preference_acceptance_rate: float
    n_with_preference_acceptance_rate: int
    # First-turn (pre-correction) preference compliance — the metric that
    # should actually move when memory is doing its job.
    avg_first_turn_preference_acceptance_rate: float
    avg_first_turn_preference_violation_count: float
    n_with_first_turn_preference_acceptance_rate: int
    avg_retrieved_skills_at_start: float
    retrieval_nonempty_rate: float
    # SkillMap-specific catalog state. Zeroed for non-SkillMap conditions.
    total_skills_after_day: int
    confirmed_skills_after_day: int
    tentative_skills_after_day: int
    pending_split_skills_after_day: int
    deprecated_skills_after_day: int


@dataclass
class ConditionStreamCurves:
    """Per-condition rolling-mean curves indexed over the concatenated stream
    of ALL days. Used to eyeball whether a condition's correction rate or
    first-turn violation count is bending down faster than the others."""

    condition: str
    n_tasks: int
    task_indices: list[int]
    correction_count_per_task: list[int]
    preference_correction_count_per_task: list[int]
    correctness_correction_count_per_task: list[int]
    first_turn_violation_count_per_task: list[float]
    rolling_window: int
    rolling_mean_corrections: list[float]
    rolling_mean_preference_corrections: list[float]
    rolling_mean_correctness_corrections: list[float]
    rolling_mean_first_turn_violations: list[float]


def _load_cfg(yaml_path: str | Path) -> dict[str, Any]:
    import yaml

    return yaml.safe_load(Path(yaml_path).read_text(encoding="utf-8"))


def _profiles_dir(cfg: dict[str, Any]) -> Path:
    path = Path(cfg["paths"]["profiles_dir"])
    return path if path.is_absolute() else ROOT / path


def _results_root(cfg: dict[str, Any]) -> Path:
    path = Path(cfg["paths"]["results_dir"])
    return path if path.is_absolute() else ROOT / path


def _storage_root() -> Path:
    return ROOT / "data" / "skillmap_day_runs"


def _make_loader(cfg: dict[str, Any]) -> LiveCodeBenchLoader | AIMELoader:
    source = cfg["tasks"].get("source", "livecodebench")
    cache_dir = cfg["tasks"]["cache_dir"]
    if source == "aime":
        return AIMELoader(cache_dir=cache_dir)
    return LiveCodeBenchLoader(cache_dir=cache_dir)


def _task_type_for_source(source: str) -> str:
    return "math_competition" if source == "aime" else "python_coding"


async def _ensure_profile(cfg: dict[str, Any], profile_id: str) -> Path:
    path = _profiles_dir(cfg) / f"{profile_id}.json"
    if path.exists():
        return path

    llm = cfg["llm"]
    model = llm.get("dev_override_a") or llm["llm_a_model"]
    region = llm.get("region", "us-east-1")
    retry_max = int(cfg["preferences"]["retry_max"])
    n_preferences = int(cfg["preferences"]["n_preferences"])

    loader = _make_loader(cfg)
    all_tasks = await loader.load()
    task_examples = _sample_day_batches(
        tasks=all_tasks,
        days=1,
        tasks_per_day=3,
        difficulty_mix={k: float(v) for k, v in cfg["tasks"]["difficulty_mix"].items()},
        seed=int(cfg["tasks"]["seed"]),
    )[0]

    source = cfg["tasks"].get("source", "livecodebench")
    elicitor = PreferenceElicitor(
        llm_a_model=model,
        region=region,
        retry_max=retry_max,
    )
    profile = await elicitor.run(
        task_type=_task_type_for_source(source),
        n_preferences=n_preferences,
        task_examples=task_examples,
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    profile.profile_id = profile_id
    path.write_text(profile.model_dump_json(indent=2), encoding="utf-8")
    return path


def _load_profile(path: Path) -> PreferenceProfile:
    return PreferenceProfile.model_validate_json(path.read_text(encoding="utf-8"))


def _sample_day_batches(
    tasks: list[EvalTask],
    days: int,
    tasks_per_day: int,
    difficulty_mix: dict[str, float],
    seed: int,
    repeat_tasks: bool = False,
) -> list[list[EvalTask]]:
    import random

    if repeat_tasks:
        # Sample one fixed batch and reuse it for every day.
        per_day_counts = _integer_allocation(difficulty_mix, tasks_per_day)
        by_diff: dict[str, list[EvalTask]] = {diff: [] for diff in difficulty_mix}
        for task in tasks:
            if task.difficulty in by_diff:
                by_diff[task.difficulty].append(task)
        rng = random.Random(seed)
        for diff, items in by_diff.items():
            rng.shuffle(items)
            if len(items) < per_day_counts[diff]:
                raise ValueError(
                    f"not enough {diff} tasks: need {per_day_counts[diff]}, have {len(items)}"
                )
        batch: list[EvalTask] = []
        for diff in ("easy", "medium", "hard"):
            batch.extend(by_diff[diff][: per_day_counts[diff]])
        return [batch for _ in range(days)]

    total = days * tasks_per_day
    overall_counts = _integer_allocation(difficulty_mix, total)
    per_day_counts = _integer_allocation(difficulty_mix, tasks_per_day)

    by_diff2: dict[str, list[EvalTask]] = {diff: [] for diff in difficulty_mix}
    for task in tasks:
        if task.difficulty in by_diff2:
            by_diff2[task.difficulty].append(task)

    rng2 = random.Random(seed)
    for diff, items in by_diff2.items():
        rng2.shuffle(items)
        if len(items) < overall_counts[diff]:
            raise ValueError(
                f"not enough {diff} tasks for {days} days: need {overall_counts[diff]}, have {len(items)}"
            )

    selected = {
        diff: by_diff2[diff][: overall_counts[diff]]
        for diff in overall_counts
    }

    batches: list[list[EvalTask]] = []
    for day_idx in range(days):
        day_batch: list[EvalTask] = []
        for diff in ("easy", "medium", "hard"):
            start = day_idx * per_day_counts[diff]
            end = start + per_day_counts[diff]
            day_batch.extend(selected[diff][start:end])
        batches.append(day_batch)
    return batches


def _integer_allocation(mix: dict[str, float], total: int) -> dict[str, int]:
    raw = {k: v * total for k, v in mix.items()}
    floor = {k: int(v) for k, v in raw.items()}
    deficit = total - sum(floor.values())
    fracs = sorted(((raw[k] - floor[k], k) for k in raw), reverse=True)
    for _, key in fracs[:deficit]:
        floor[key] += 1
    return floor


def _build_condition(
    name: str,
    *,
    llm_b_model: str,
    region: str,
    temperature_b: float,
    storage_root: Path,
) -> Condition:
    if name == "stateless":
        return StatelessCondition(
            llm_b_model=llm_b_model,
            region=region,
            temperature=temperature_b,
        )
    if name == "declarative_memory":
        return DeclarativeMemoryCondition(
            llm_b_model=llm_b_model,
            region=region,
            temperature=temperature_b,
            storage_root=str(storage_root),
            resume_existing=True,
        )
    if name == "skillmap":
        return SkillMapCondition(
            llm_b_model=llm_b_model,
            region=region,
            temperature=temperature_b,
            storage_root=str(storage_root),
            resume_existing=True,
        )
    raise ValueError(f"unknown condition: {name!r}")


def _make_day_summary(
    day_index: int,
    condition_name: str,
    run: StreamRun,
    skill_count: int,
) -> DaySummary:
    interactions = run.interactions
    n_tasks = len(interactions)
    total_corrections = sum(item.correction_count for item in interactions)
    total_pref_corrections = sum(item.preference_correction_count for item in interactions)
    total_correctness_corrections = sum(item.correctness_correction_count for item in interactions)
    total_retrieved = sum(len(item.retrieved_skill_ids_at_start) for item in interactions)
    retrieved_nonempty = sum(1 for item in interactions if item.retrieved_skill_ids_at_start)

    avg_pass_measurable, n_pass = mean_test_case_pass_rate(interactions)
    avg_pass_strict = mean_test_case_pass_rate_strict(interactions)
    avg_pref_acceptance, n_pref = mean_preference_acceptance_rate(interactions)
    avg_first_turn_acceptance, n_first = mean_first_turn_preference_acceptance_rate(interactions)
    avg_first_turn_violations, _ = mean_first_turn_preference_violation_count(interactions)

    return DaySummary(
        day_index=day_index,
        condition=condition_name,
        n_tasks=n_tasks,
        task_ids=run.task_stream,
        total_corrections=total_corrections,
        avg_corrections_per_task=(total_corrections / n_tasks) if n_tasks else 0.0,
        avg_preference_corrections_per_task=(total_pref_corrections / n_tasks) if n_tasks else 0.0,
        avg_correctness_corrections_per_task=(total_correctness_corrections / n_tasks) if n_tasks else 0.0,
        task_completion_rate=task_completion_rate(interactions),
        avg_test_case_pass_rate_measurable=avg_pass_measurable,
        avg_test_case_pass_rate_strict=avg_pass_strict,
        n_with_test_case_pass_rate=n_pass,
        avg_preference_acceptance_rate=avg_pref_acceptance,
        n_with_preference_acceptance_rate=n_pref,
        avg_first_turn_preference_acceptance_rate=avg_first_turn_acceptance,
        avg_first_turn_preference_violation_count=avg_first_turn_violations,
        n_with_first_turn_preference_acceptance_rate=n_first,
        avg_retrieved_skills_at_start=(total_retrieved / n_tasks) if n_tasks else 0.0,
        retrieval_nonempty_rate=(retrieved_nonempty / n_tasks) if n_tasks else 0.0,
        total_skills_after_day=skill_count,
        confirmed_skills_after_day=skill_count,
        tentative_skills_after_day=0,
        pending_split_skills_after_day=0,
        deprecated_skills_after_day=0,
    )


def _rolling_mean(xs: list[float], window: int) -> list[float]:
    """Right-aligned rolling mean: index i averages xs[max(0, i-w+1) : i+1].

    The first few entries average over fewer elements than `window`, so the
    curve has the right length but starts noisy — same shape as
    correction_rate.compute_correction_curve in the metrics package.
    """
    out: list[float] = []
    for i in range(len(xs)):
        lo = max(0, i - window + 1)
        chunk = xs[lo : i + 1]
        out.append(sum(chunk) / len(chunk) if chunk else 0.0)
    return out


def _condition_stream_curves(
    condition_name: str,
    interactions: list[TaskInteraction],
    window: int,
) -> ConditionStreamCurves:
    """Build cross-day stream curves for one condition. The interactions
    list MUST be ordered by stream task index across days (i.e. day 1's
    tasks first, then day 2's, etc.)."""
    indices = list(range(len(interactions)))
    corrections = [it.correction_count for it in interactions]
    pref = [it.preference_correction_count for it in interactions]
    correctness = [it.correctness_correction_count for it in interactions]
    # Use 0.0 in place of None so the curve has uniform length; the legend
    # for this metric is "violations on the first assistant turn", and a
    # task with no assistant turn / empty profile is most fairly counted
    # as zero violations (it neither helped nor hurt the curve).
    first_turn = [
        float(it.first_turn_preference_violation_count or 0)
        for it in interactions
    ]
    return ConditionStreamCurves(
        condition=condition_name,
        n_tasks=len(interactions),
        task_indices=indices,
        correction_count_per_task=corrections,
        preference_correction_count_per_task=pref,
        correctness_correction_count_per_task=correctness,
        first_turn_violation_count_per_task=first_turn,
        rolling_window=window,
        rolling_mean_corrections=_rolling_mean(
            [float(c) for c in corrections], window
        ),
        rolling_mean_preference_corrections=_rolling_mean(
            [float(c) for c in pref], window
        ),
        rolling_mean_correctness_corrections=_rolling_mean(
            [float(c) for c in correctness], window
        ),
        rolling_mean_first_turn_violations=_rolling_mean(first_turn, window),
    )


def _load_existing_run(day_path: Path) -> StreamRun | None:
    if not day_path.exists():
        return None
    return StreamRun.model_validate_json(day_path.read_text(encoding="utf-8"))


def _day_path(results_root: Path, day_index: int, condition_name: str) -> Path:
    return results_root / f"day_{day_index:02d}_{condition_name}.json"


def _condition_skill_count(condition: Condition) -> int:
    if isinstance(condition, SkillMapCondition) and condition.skill_map is not None:
        return len(condition.skill_map.list_skills())
    return 0


async def _run_condition_for_day(
    *,
    condition: Condition,
    condition_name: str,
    day_index: int,
    tasks_for_day: list[EvalTask],
    interaction_loop: InteractionLoop,
    simulator: UserSimulator,
    profile: PreferenceProfile,
    session_id: str,
    results_root: Path,
) -> StreamRun:
    day_path = _day_path(results_root, day_index, condition_name)
    existing_run = _load_existing_run(day_path)
    if existing_run is not None:
        run = existing_run
        if run.completed_at is not None and len(run.interactions) >= len(tasks_for_day):
            return run
    else:
        run = StreamRun(
            run_id=f"{session_id}_day{day_index:02d}_{condition_name}",
            profile_id=profile.profile_id,
            condition_name=condition_name,  # type: ignore[arg-type]
            task_stream=[task.task_id for task in tasks_for_day],
            held_out_task_ids=[],
            started_at=datetime.now(timezone.utc),
        )

    completed_task_ids = {interaction.task_id for interaction in run.interactions}
    remaining_tasks = [
        task for task in tasks_for_day if task.task_id not in completed_task_ids
    ]

    for task in remaining_tasks:
        task_index = next(
            idx for idx, candidate in enumerate(tasks_for_day)
            if candidate.task_id == task.task_id
        )
        try:
            interaction = await interaction_loop.run_single_task(
                task=task,
                task_index=task_index,
                condition=condition,
                simulator=simulator,
            )
        except Exception as e:
            print(
                f"[run_skillmap_days] day {day_index} cond {condition_name} "
                f"task {task.task_id} failed ({type(e).__name__}): {e}. Skipping.",
                file=sys.stderr,
                flush=True,
            )
            continue
        run.interactions.append(interaction)
        day_path.write_text(run.model_dump_json(indent=2), encoding="utf-8")

    run.completed_at = datetime.now(timezone.utc)
    day_path.write_text(run.model_dump_json(indent=2), encoding="utf-8")
    return run


async def _run(args: argparse.Namespace) -> Path:
    cfg = _load_cfg(args.config)
    profile_path = await _ensure_profile(cfg, args.profile)
    profile = _load_profile(profile_path)

    loader = _make_loader(cfg)
    all_tasks = await loader.load()
    day_batches = _sample_day_batches(
        tasks=all_tasks,
        days=args.days,
        tasks_per_day=args.tasks_per_day,
        difficulty_mix={k: float(v) for k, v in cfg["tasks"]["difficulty_mix"].items()},
        seed=int(cfg["tasks"]["seed"]),
        repeat_tasks=args.repeat_tasks,
    )

    llm = cfg["llm"]
    region = llm.get("region", "us-east-1")
    llm_a_model = llm.get("dev_override_a") or llm["llm_a_model"]
    llm_b_model = llm.get("dev_override_b") or llm["llm_b_model"]

    requested_conditions = [
        name.strip() for name in args.conditions.split(",") if name.strip()
    ]
    for name in requested_conditions:
        if name not in SUPPORTED_CONDITIONS:
            raise ValueError(
                f"unknown condition {name!r}. Supported: {SUPPORTED_CONDITIONS}"
            )

    session_id = args.session_id or (
        f"{args.profile}_days_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    )
    results_root = _results_root(cfg) / "skillmap_days" / session_id
    results_root.mkdir(parents=True, exist_ok=True)

    interaction_loop = InteractionLoop(
        max_turns=int(cfg["interaction"]["max_turns_per_task"]),
        sanity_timeout_s=float(cfg["sanity_check"]["timeout_per_test_seconds"]),
    )
    simulator = UserSimulator(
        llm_a_model=llm_a_model,
        profile=profile,
        region=region,
        max_turns_per_task=int(cfg["interaction"]["max_turns_per_task"]),
        give_up_threshold_repeats=int(cfg["interaction"]["give_up_threshold_repeats"]),
        retry_max=3,
    )

    storage_root = _storage_root()
    conditions: dict[str, Condition] = {}
    for name in requested_conditions:
        cond = _build_condition(
            name,
            llm_b_model=llm_b_model,
            region=region,
            temperature_b=float(llm.get("temperature_b", 0.7)),
            storage_root=storage_root,
        )
        await cond.setup(profile_id=profile.profile_id, run_id=session_id)
        conditions[name] = cond

    # Per-condition list of (day_index, DaySummary). Plus the full ordered
    # interaction list across all days so we can build cross-day rolling
    # curves at the end.
    day_summaries: list[DaySummary] = []
    cross_day_interactions: dict[str, list[TaskInteraction]] = {
        name: [] for name in requested_conditions
    }

    try:
        for day_index, tasks_for_day in enumerate(day_batches, start=1):
            for name in requested_conditions:
                run = await _run_condition_for_day(
                    condition=conditions[name],
                    condition_name=name,
                    day_index=day_index,
                    tasks_for_day=tasks_for_day,
                    interaction_loop=interaction_loop,
                    simulator=simulator,
                    profile=profile,
                    session_id=session_id,
                    results_root=results_root,
                )
                cross_day_interactions[name].extend(run.interactions)
                day_summaries.append(
                    _make_day_summary(
                        day_index=day_index,
                        condition_name=name,
                        run=run,
                        skill_count=_condition_skill_count(conditions[name]),
                    )
                )
    finally:
        for cond in conditions.values():
            try:
                await cond.teardown()
            except Exception:
                pass

    rolling_window = max(1, args.rolling_window)
    correction_curves = [
        asdict(_condition_stream_curves(name, cross_day_interactions[name], rolling_window))
        for name in requested_conditions
    ]

    summary = {
        "session_id": session_id,
        "profile_id": profile.profile_id,
        "days": args.days,
        "tasks_per_day": args.tasks_per_day,
        "conditions": requested_conditions,
        "rolling_window": rolling_window,
        "storage_paths": {
            name: str(storage_root / profile.profile_id / session_id / _storage_filename(name))
            for name in requested_conditions
            if _storage_filename(name) is not None
        },
        "day_summaries": [asdict(item) for item in day_summaries],
        "correction_curves": correction_curves,
    }
    summary_path = results_root / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary_path


def _storage_filename(condition_name: str) -> str | None:
    """Return the on-disk file name for a condition's persistent state, or
    None for stateless conditions that don't persist anything."""
    if condition_name == "skillmap":
        return "skill_map.json"
    if condition_name == "declarative_memory":
        return "declarative_memory.json"
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", default="kimi_k25_livecodebench_v1")
    parser.add_argument("--config", default=str(ROOT / "eval_config.yaml"))
    parser.add_argument("--days", type=int, default=3)
    parser.add_argument("--tasks-per-day", type=int, default=20)
    parser.add_argument("--session-id", default=None)
    parser.add_argument(
        "--conditions",
        default=",".join(SUPPORTED_CONDITIONS),
        help=(
            "Comma-separated list of conditions to run (in this order). "
            f"Supported: {','.join(SUPPORTED_CONDITIONS)}."
        ),
    )
    parser.add_argument(
        "--rolling-window",
        type=int,
        default=10,
        help="Window size for cross-day rolling-mean correction curves.",
    )
    parser.add_argument(
        "--repeat-tasks",
        action="store_true",
        default=False,
        help="Use the same task set for every day instead of disjoint batches.",
    )
    args = parser.parse_args()

    summary_path = asyncio.run(_run(args))
    print(f"day-run complete. summary written to {summary_path}")


if __name__ == "__main__":
    main()
