"""EvalOrchestrator: glue.

Responsibilities:
  1. Load a PreferenceProfile from disk.
  2. Load + sample LiveCodeBench tasks.
  3. For each condition, run a StreamRunner.
  4. Compute all metrics (correction curve, preference recovery,
     generalization, correctness sanity).
  5. Persist an EvalReport JSON.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from skillmap_eval.analysis.aggregate import aggregate_runs
from skillmap_eval.conditions import (
    Condition,
    DeclarativeMemoryCondition,
    SkillMapCondition,
    StatelessCondition,
)
from skillmap_eval.metrics.preference_recovery import PreferenceRecoveryJudge
from skillmap_eval.runner import InteractionLoop, StreamRunner
from skillmap_eval.simulator import UserSimulator
from skillmap_eval.tasks import LiveCodeBenchLoader
from skillmap_eval.types import (
    EvalReport,
    PreferenceProfile,
    PreferenceRecoveryResult,
    StreamRun,
)


@dataclass
class EvalConfig:
    llm_a_model: str
    llm_b_model: str
    llm_judge_model: str
    region: str
    temperature_a: float
    temperature_b: float
    n_stream: int
    n_held_out: int
    difficulty_mix: dict[str, float]
    seed: int
    cache_dir: str
    max_turns_per_task: int
    give_up_threshold_repeats: int
    conditions: list[str]
    sanity_timeout_s: float
    profiles_dir: str
    results_dir: str


class EvalOrchestrator:
    def __init__(self, cfg: EvalConfig) -> None:
        self.cfg = cfg

    async def run_full_eval(self, profile_id: str) -> EvalReport:
        profile = self._load_profile(profile_id)

        # 1. Load + sample tasks.
        loader = LiveCodeBenchLoader(self.cfg.cache_dir)
        all_tasks = await loader.load()
        stream_tasks, held_out = loader.sample_stream(
            all_tasks,
            n_stream=self.cfg.n_stream,
            n_held_out=self.cfg.n_held_out,
            difficulty_mix=self.cfg.difficulty_mix,
            seed=self.cfg.seed,
        )

        # 2. Run each condition.
        simulator = UserSimulator(
            llm_a_model=self.cfg.llm_a_model,
            profile=profile,
            region=self.cfg.region,
            max_turns_per_task=self.cfg.max_turns_per_task,
            give_up_threshold_repeats=self.cfg.give_up_threshold_repeats,
        )
        interaction_loop = InteractionLoop(
            max_turns=self.cfg.max_turns_per_task,
            sanity_timeout_s=self.cfg.sanity_timeout_s,
        )
        stream_runner = StreamRunner(interaction_loop, results_dir=self.cfg.results_dir)

        runs: list[StreamRun] = []
        skillmap_condition: SkillMapCondition | None = None
        for cond_name in self.cfg.conditions:
            cond = self._build_condition(cond_name)
            if isinstance(cond, SkillMapCondition):
                skillmap_condition = cond
            run = await stream_runner.run_stream(
                profile=profile,
                stream_tasks=stream_tasks,
                held_out_tasks=held_out,
                condition=cond,
                simulator=simulator,
            )
            runs.append(run)

        # 3. Preference recovery (SkillMap condition only).
        recovery_results: list[PreferenceRecoveryResult] = []
        if skillmap_condition is not None and skillmap_condition.skill_map is not None:
            judge = PreferenceRecoveryJudge(
                judge_model=self.cfg.llm_judge_model, region=self.cfg.region
            )
            result = await judge.judge(
                profile=profile,
                induced_skills=skillmap_condition.skill_map.list_skills(),
                condition_name="skillmap",
            )
            recovery_results.append(result)

        # 4. Assemble + persist report.
        report = aggregate_runs(
            profile_id=profile.profile_id,
            runs=runs,
            preference_recovery=recovery_results,
        )
        self._persist_report(profile.profile_id, report)
        return report

    # ------------------------------------------------------------------

    def _load_profile(self, profile_id: str) -> PreferenceProfile:
        path = Path(self.cfg.profiles_dir) / f"{profile_id}.json"
        if not path.exists():
            raise FileNotFoundError(f"preference profile not found: {path}")
        return PreferenceProfile.model_validate_json(path.read_text(encoding="utf-8"))

    def _build_condition(self, name: str) -> Condition:
        if name == "stateless":
            return StatelessCondition(
                llm_b_model=self.cfg.llm_b_model,
                region=self.cfg.region,
                temperature=self.cfg.temperature_b,
            )
        if name == "declarative_memory":
            return DeclarativeMemoryCondition(
                llm_b_model=self.cfg.llm_b_model,
                region=self.cfg.region,
                temperature=self.cfg.temperature_b,
            )
        if name == "skillmap":
            return SkillMapCondition(
                llm_b_model=self.cfg.llm_b_model,
                region=self.cfg.region,
                temperature=self.cfg.temperature_b,
            )
        raise ValueError(f"unknown condition: {name!r}")

    def _persist_report(self, profile_id: str, report: EvalReport) -> None:
        out = Path(self.cfg.results_dir)
        out.mkdir(parents=True, exist_ok=True)
        path = out / f"eval_report_{profile_id}.json"
        path.write_text(report.model_dump_json(indent=2), encoding="utf-8")


def load_eval_config(yaml_path: str | Path) -> EvalConfig:
    import yaml

    data: dict[str, Any] = yaml.safe_load(Path(yaml_path).read_text(encoding="utf-8"))
    llm = data["llm"]
    # Honor dev overrides if present.
    a = llm.get("dev_override_a") or llm["llm_a_model"]
    b = llm.get("dev_override_b") or llm["llm_b_model"]
    judge = llm.get("dev_override_judge") or llm["llm_judge_model"]
    return EvalConfig(
        llm_a_model=a,
        llm_b_model=b,
        llm_judge_model=judge,
        region=llm.get("region", "us-east-1"),
        temperature_a=float(llm.get("temperature_a", 0.0)),
        temperature_b=float(llm.get("temperature_b", 0.7)),
        n_stream=int(data["tasks"]["n_stream"]),
        n_held_out=int(data["tasks"]["n_held_out"]),
        difficulty_mix={k: float(v) for k, v in data["tasks"]["difficulty_mix"].items()},
        seed=int(data["tasks"]["seed"]),
        cache_dir=data["tasks"]["cache_dir"],
        max_turns_per_task=int(data["interaction"]["max_turns_per_task"]),
        give_up_threshold_repeats=int(data["interaction"]["give_up_threshold_repeats"]),
        conditions=list(data["conditions"]),
        sanity_timeout_s=float(data["sanity_check"]["timeout_per_test_seconds"]),
        profiles_dir=data["paths"]["profiles_dir"],
        results_dir=data["paths"]["results_dir"],
    )
