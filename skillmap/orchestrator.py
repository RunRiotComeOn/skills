"""Orchestrator: wires Stage A/B/C pipeline.

handle_query  → Stage C (select) → Assistant (LLM-B call) → return response
finalize_task → Stage A (summarize) → append to buffer
              → if buffer >= trigger: Stage B (consolidate)
"""

from __future__ import annotations

CONSOLIDATION_TRIGGER = 20

from skillmap.induction.consolidator import SkillConsolidator
from skillmap.induction.summarizer import CorrectionSummarizer
from skillmap.retrieval.selector import CatalogSelector
from skillmap.runtime.assistant import Assistant
from skillmap.storage.skill_map import SkillMap
from skillmap.types import Skill


class Orchestrator:
    def __init__(
        self,
        skill_map: SkillMap,
        summarizer: CorrectionSummarizer | None = None,
        consolidator: SkillConsolidator | None = None,
        selector: CatalogSelector | None = None,
        assistant: Assistant | None = None,
    ) -> None:
        self.skill_map = skill_map
        self.summarizer = summarizer or CorrectionSummarizer()
        self.consolidator = consolidator or SkillConsolidator()
        self.selector = selector or CatalogSelector()
        self.assistant = assistant or Assistant()

        # State carried between handle_query and finalize_task for one task.
        self._pending_selected: list[Skill] = []
        self._pending_conversation: list[dict] = []

    # ------------------------------------------------------------------
    # Runtime path (called per turn from the condition)
    # ------------------------------------------------------------------

    async def handle_first_turn(
        self, user_query: str
    ) -> tuple[str, list[str]]:
        """Stage C + LLM-B call for the first user turn of a task."""
        selected = await self.selector.select(user_query, self.skill_map)
        self._pending_selected = selected
        self._pending_conversation = []

        response = await self.assistant.run(
            user_query,
            selected_skills=selected,
            conversation_history=None,
        )
        self._pending_conversation = [
            {"role": "user", "content": user_query},
            {"role": "assistant", "content": response},
        ]
        return response, [s.id for s in selected]

    async def handle_followup_turn(self, user_message: str) -> str:
        """LLM-B call for subsequent turns (skills already injected via system prompt)."""
        response = await self.assistant.run(
            user_message,
            selected_skills=self._pending_selected,
            conversation_history=self._pending_conversation,
        )
        self._pending_conversation.append({"role": "user", "content": user_message})
        self._pending_conversation.append({"role": "assistant", "content": response})
        return response

    # ------------------------------------------------------------------
    # Induction path (called once per task after completion)
    # ------------------------------------------------------------------

    async def finalize_task(self, task_id: str) -> None:
        """Stage A: summarize corrections; Stage B: consolidate if buffer full."""
        trajectory = list(self._pending_conversation)
        self._pending_selected = []
        self._pending_conversation = []

        if not trajectory:
            return

        # Stage A
        try:
            summaries = await self.summarizer.run(trajectory, task_id)
            for s in summaries:
                self.skill_map.append_summary(s)
        except Exception as exc:
            print(f"[orchestrator] summarizer error for task {task_id}: {exc}")

        # Stage B
        if self.skill_map.pending_summary_count >= CONSOLIDATION_TRIGGER:
            try:
                await self.consolidator.run(self.skill_map)
            except Exception as exc:
                print(f"[orchestrator] consolidator error: {exc}")
