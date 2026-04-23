"""Run a SkillMap-only incremental eval in day-sized batches.

Usage:
    env PYTHONPATH=. python eval/scripts/run_skillmap_days.py --profile kimi_k25_v1

Behavior:
  - If the profile does not exist, generate it first with Stage 1.
  - Sample disjoint LiveCodeBench tasks with a fixed per-day difficulty mix.
  - Run only the SkillMap condition.
  - Reuse the same SkillMap state across day1/day2/day3.
  - Persist a per-day stream JSON plus an aggregate summary JSON.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from skillmap_eval.conditions import SkillMapCondition
from skillmap_eval.preferences import PreferenceElicitor
from skillmap_eval.runner import InteractionLoop
from skillmap_eval.simulator import UserSimulator
from skillmap_eval.tasks import LiveCodeBenchLoader
from skillmap_eval.types import EvalTask, PreferenceProfile, StreamRun


@dataclass
class DaySummary:
    day_index: int
    n_tasks: int
    task_ids: list[str]
    total_corrections: int
    avg_corrections_per_task: float
    acceptance_rate: float
    avg_retrieved_skills_at_start: float
    retrieval_nonempty_rate: float
    total_skills_after_day: int
    confirmed_skills_after_day: int
    tentative_skills_after_day: int
    pending_split_skills_after_day: int
    deprecated_skills_after_day: int


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


async def _ensure_profile(cfg: dict[str, Any], profile_id: str) -> Path:
    path = _profiles_dir(cfg) / f"{profile_id}.json"
    if path.exists():
        return path

    llm = cfg["llm"]
    model = llm.get("dev_override_a") or llm["llm_a_model"]
    region = llm.get("region", "us-east-1")
    retry_max = int(cfg["preferences"]["retry_max"])
    n_preferences = int(cfg["preferences"]["n_preferences"])

    loader = LiveCodeBenchLoader(cache_dir=cfg["tasks"]["cache_dir"])
    all_tasks = await loader.load()
    task_examples = _sample_day_batches(
        tasks=all_tasks,
        days=1,
        tasks_per_day=3,
        difficulty_mix={k: float(v) for k, v in cfg["tasks"]["difficulty_mix"].items()},
        seed=int(cfg["tasks"]["seed"]),
    )[0]

    elicitor = PreferenceElicitor(
        llm_a_model=model,
        region=region,
        retry_max=retry_max,
    )
    profile = await elicitor.run(
        task_type="python_coding",
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
) -> list[list[EvalTask]]:
    import random

    total = days * tasks_per_day
    overall_counts = _integer_allocation(difficulty_mix, total)
    per_day_counts = _integer_allocation(difficulty_mix, tasks_per_day)

    by_diff: dict[str, list[EvalTask]] = {diff: [] for diff in difficulty_mix}
    for task in tasks:
        if task.difficulty in by_diff:
            by_diff[task.difficulty].append(task)

    rng = random.Random(seed)
    for diff, items in by_diff.items():
        rng.shuffle(items)
        if len(items) < overall_counts[diff]:
            raise ValueError(
                f"not enough {diff} tasks for {days} days: need {overall_counts[diff]}, have {len(items)}"
            )

    selected = {
        diff: by_diff[diff][: overall_counts[diff]]
        for diff in overall_counts
    }

    batches: list[list[EvalTask]] = []
    for day_idx in range(days):
        batch: list[EvalTask] = []
        for diff in ("easy", "medium", "hard"):
            start = day_idx * per_day_counts[diff]
            end = start + per_day_counts[diff]
            batch.extend(selected[diff][start:end])
        batches.append(batch)
    return batches


def _integer_allocation(mix: dict[str, float], total: int) -> dict[str, int]:
    raw = {k: v * total for k, v in mix.items()}
    floor = {k: int(v) for k, v in raw.items()}
    deficit = total - sum(floor.values())
    fracs = sorted(((raw[k] - floor[k], k) for k in raw), reverse=True)
    for _, key in fracs[:deficit]:
        floor[key] += 1
    return floor


def _make_day_summary(day_index: int, run: StreamRun, condition: SkillMapCondition) -> DaySummary:
    interactions = run.interactions
    n_tasks = len(interactions)
    total_corrections = sum(item.correction_count for item in interactions)
    accepted = sum(1 for item in interactions if item.completion_reason == "user_accepted")
    total_retrieved = sum(len(item.retrieved_skill_ids_at_start) for item in interactions)
    retrieved_nonempty = sum(1 for item in interactions if item.retrieved_skill_ids_at_start)

    skills = condition.skill_map.list_skills() if condition.skill_map is not None else []
    return DaySummary(
        day_index=day_index,
        n_tasks=n_tasks,
        task_ids=run.task_stream,
        total_corrections=total_corrections,
        avg_corrections_per_task=(total_corrections / n_tasks) if n_tasks else 0.0,
        acceptance_rate=(accepted / n_tasks) if n_tasks else 0.0,
        avg_retrieved_skills_at_start=(total_retrieved / n_tasks) if n_tasks else 0.0,
        retrieval_nonempty_rate=(retrieved_nonempty / n_tasks) if n_tasks else 0.0,
        total_skills_after_day=len(skills),
        confirmed_skills_after_day=len(skills),
        tentative_skills_after_day=0,
        pending_split_skills_after_day=0,
        deprecated_skills_after_day=0,
    )


def _load_existing_run(day_path: Path) -> StreamRun | None:
    if not day_path.exists():
        return None
    return StreamRun.model_validate_json(day_path.read_text(encoding="utf-8"))


async def _run(args: argparse.Namespace) -> Path:
    cfg = _load_cfg(args.config)
    profile_path = await _ensure_profile(cfg, args.profile)
    profile = _load_profile(profile_path)

    loader = LiveCodeBenchLoader(cache_dir=cfg["tasks"]["cache_dir"])
    all_tasks = await loader.load()
    day_batches = _sample_day_batches(
        tasks=all_tasks,
        days=args.days,
        tasks_per_day=args.tasks_per_day,
        difficulty_mix={k: float(v) for k, v in cfg["tasks"]["difficulty_mix"].items()},
        seed=int(cfg["tasks"]["seed"]),
    )

    llm = cfg["llm"]
    region = llm.get("region", "us-east-1")
    llm_a_model = llm.get("dev_override_a") or llm["llm_a_model"]
    llm_b_model = llm.get("dev_override_b") or llm["llm_b_model"]

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
    condition = SkillMapCondition(
        llm_b_model=llm_b_model,
        region=region,
        temperature=float(llm.get("temperature_b", 0.7)),
        storage_root=str(_storage_root()),
        resume_existing=True,
    )
    await condition.setup(profile_id=profile.profile_id, run_id=session_id)

    day_summaries: list[DaySummary] = []
    try:
        for day_index, tasks_for_day in enumerate(day_batches, start=1):
            day_path = results_root / f"day_{day_index:02d}.json"
            existing_run = _load_existing_run(day_path)
            if existing_run is not None:
                run = existing_run
                if run.completed_at is not None and len(run.interactions) >= len(tasks_for_day):
                    day_summaries.append(_make_day_summary(day_index, run, condition))
                    continue
            else:
                run = StreamRun(
                    run_id=f"{session_id}_day{day_index:02d}",
                    profile_id=profile.profile_id,
                    condition_name="skillmap",
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
                    idx for idx, candidate in enumerate(tasks_for_day) if candidate.task_id == task.task_id
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
                        f"[run_skillmap_days] day {day_index} task {task.task_id} "
                        f"failed ({type(e).__name__}): {e}. Skipping.",
                        file=sys.stderr,
                        flush=True,
                    )
                    continue
                run.interactions.append(interaction)
                day_path.write_text(run.model_dump_json(indent=2), encoding="utf-8")

            run.completed_at = datetime.now(timezone.utc)
            day_path.write_text(run.model_dump_json(indent=2), encoding="utf-8")
            day_summaries.append(_make_day_summary(day_index, run, condition))
    finally:
        await condition.teardown()

    summary = {
        "session_id": session_id,
        "profile_id": profile.profile_id,
        "days": args.days,
        "tasks_per_day": args.tasks_per_day,
        "storage_path": str(_storage_root() / profile.profile_id / session_id / "skill_map.json"),
        "day_summaries": [asdict(item) for item in day_summaries],
    }
    summary_path = results_root / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", default="kimi_k25_livecodebench_v1")
    parser.add_argument("--config", default=str(ROOT / "eval_config.yaml"))
    parser.add_argument("--days", type=int, default=3)
    parser.add_argument("--tasks-per-day", type=int, default=20)
    parser.add_argument("--session-id", default=None)
    args = parser.parse_args()

    summary_path = asyncio.run(_run(args))
    print(f"skillmap day-run complete. summary written to {summary_path}")


if __name__ == "__main__":
    main()
