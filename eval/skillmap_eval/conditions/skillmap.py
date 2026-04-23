"""SkillMap condition: three-stage flat experience library.

Stage A (per task): CorrectionSummarizer extracts correction summaries.
Stage B (every 20 summaries): SkillConsolidator writes/updates skills.
Stage C (per task, first turn): CatalogSelector picks ≤ 2 skills.
"""

from __future__ import annotations

from pathlib import Path

from skillmap.induction import CorrectionSummarizer, SkillConsolidator
from skillmap.llm.client import LLMConfig, configure_default_client
from skillmap.orchestrator import Orchestrator
from skillmap.retrieval import CatalogSelector
from skillmap.runtime import Assistant
from skillmap.storage import JSONPersistence, SkillMap
from skillmap.types import SkillMapState

from skillmap_eval.conditions.base import render_conversation_for_llm_b
from skillmap_eval.types import EvalTask, SimulatedTurn, TaskInteraction


class SkillMapCondition:
    name = "skillmap"

    def __init__(
        self,
        llm_b_model: str,
        region: str = "us-east-1",
        temperature: float = 0.7,
        storage_root: str = "./data/skillmap_eval_runs",
        resume_existing: bool = False,
        structured_model: str = "us.anthropic.claude-haiku-4-5-20251001-v1:0",
    ) -> None:
        self.llm_b_model = llm_b_model
        self.region = region
        self.temperature = temperature
        self.storage_root = Path(storage_root)
        self.resume_existing = resume_existing
        self.structured_model = structured_model
        self._orchestrator: Orchestrator | None = None
        self._skill_map: SkillMap | None = None

    async def setup(self, profile_id: str, run_id: str) -> None:
        configure_default_client(
            LLMConfig(
                provider="bedrock",
                model=self.llm_b_model,
                region=self.region,
                extra_inference={"temperature": self.temperature},
            )
        )

        run_dir = self.storage_root / profile_id / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        persistence = JSONPersistence(run_dir / "skill_map.json", user_id=run_id)
        state = persistence.load() if self.resume_existing else SkillMapState(user_id=run_id)
        self._skill_map = SkillMap(state, persistence)

        haiku = self.structured_model
        self._orchestrator = Orchestrator(
            skill_map=self._skill_map,
            summarizer=CorrectionSummarizer(llm_model=haiku, region=self.region),
            consolidator=SkillConsolidator(llm_model=haiku, region=self.region),
            selector=CatalogSelector(llm_model=haiku, region=self.region),
            assistant=Assistant(),
        )

    async def handle_user_message(
        self,
        task: EvalTask,
        conversation_so_far: list[SimulatedTurn],
        user_message: str,
    ) -> tuple[str, list[str]]:
        assert self._orchestrator is not None, "setup() not called"

        if not conversation_so_far:
            return await self._orchestrator.handle_first_turn(user_message)
        else:
            response = await self._orchestrator.handle_followup_turn(user_message)
            return response, []

    async def finalize_task(
        self,
        task: EvalTask,
        interaction: TaskInteraction,
    ) -> None:
        assert self._orchestrator is not None
        await self._orchestrator.finalize_task(task_id=task.task_id)

    async def teardown(self) -> None:
        self._orchestrator = None
        self._skill_map = None

    @property
    def skill_map(self) -> SkillMap | None:
        return self._skill_map
