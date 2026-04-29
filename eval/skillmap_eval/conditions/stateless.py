"""Stateless baseline: no memory. Each turn, call LLM-B with current conversation."""

from __future__ import annotations

from skillmap.llm.client import LLMClient, LLMConfig

from skillmap_eval.conditions.base import (
    SYSTEM_PROMPT_BASE,
    format_task_hint,
    render_conversation_for_llm_b,
)
from skillmap_eval.types import EvalTask, SimulatedTurn, TaskInteraction


class StatelessCondition:
    name = "stateless"

    def __init__(
        self,
        llm_b_model: str,
        region: str = "us-east-1",
        temperature: float = 0.7,
    ) -> None:
        self._client = LLMClient(
            LLMConfig(
                provider="bedrock",
                model=llm_b_model,
                region=region,
                extra_inference={"temperature": temperature},
            )
        )

    async def setup(self, profile_id: str, run_id: str) -> None:
        return None

    async def handle_user_message(
        self,
        task: EvalTask,
        conversation_so_far: list[SimulatedTurn],
        user_message: str,
    ) -> tuple[str, list[str]]:
        messages = render_conversation_for_llm_b(conversation_so_far, user_message)
        system = SYSTEM_PROMPT_BASE
        hint = format_task_hint(task)
        if hint:
            system = f"{system}\n\n{hint}"
        text = await self._client.call(messages=messages, system=system)
        return (text if isinstance(text, str) else str(text)), []

    async def finalize_task(
        self,
        task: EvalTask,
        interaction: TaskInteraction,
    ) -> None:
        return None

    async def teardown(self) -> None:
        return None
