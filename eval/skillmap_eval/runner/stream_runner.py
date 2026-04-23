"""StreamRunner: run a sequential stream of tasks for one condition."""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

from skillmap_eval.conditions.base import Condition
from skillmap_eval.runner.interaction_loop import InteractionLoop
from skillmap_eval.simulator.user_simulator import UserSimulator
from skillmap_eval.types import (
    ConditionName,
    EvalTask,
    PreferenceProfile,
    StreamRun,
)


class StreamRunner:
    def __init__(
        self,
        interaction_loop: InteractionLoop,
        results_dir: str = "./results",
    ) -> None:
        self.interaction_loop = interaction_loop
        self.results_dir = Path(results_dir)

    async def run_stream(
        self,
        profile: PreferenceProfile,
        stream_tasks: list[EvalTask],
        held_out_tasks: list[EvalTask],
        condition: Condition,
        simulator: UserSimulator,
    ) -> StreamRun:
        run_id = f"run_{uuid.uuid4().hex[:8]}"
        await condition.setup(profile_id=profile.profile_id, run_id=run_id)

        run = StreamRun(
            run_id=run_id,
            profile_id=profile.profile_id,
            condition_name=_cast_condition_name(condition.name),
            task_stream=[t.task_id for t in stream_tasks],
            held_out_task_ids=[t.task_id for t in held_out_tasks],
            started_at=datetime.now(timezone.utc),
        )

        for idx, task in enumerate(stream_tasks):
            interaction = await self.interaction_loop.run_single_task(
                task=task,
                task_index=idx,
                condition=condition,
                simulator=simulator,
            )
            run.interactions.append(interaction)
            self._persist_incremental(run)

        for idx, task in enumerate(held_out_tasks):
            interaction = await self.interaction_loop.run_single_task(
                task=task,
                task_index=idx,
                condition=condition,
                simulator=simulator,
            )
            run.held_out_interactions.append(interaction)
            self._persist_incremental(run)

        run.completed_at = datetime.now(timezone.utc)
        await condition.teardown()
        self._persist_incremental(run)
        return run

    # ------------------------------------------------------------------

    def _persist_incremental(self, run: StreamRun) -> None:
        self.results_dir.mkdir(parents=True, exist_ok=True)
        path = self.results_dir / f"stream_{run.condition_name}_{run.run_id}.json"
        path.write_text(run.model_dump_json(indent=2), encoding="utf-8")


def _cast_condition_name(name: str) -> ConditionName:
    if name not in ("stateless", "declarative_memory", "skillmap"):
        raise ValueError(f"unexpected condition name {name!r}")
    return name  # type: ignore[return-value]
