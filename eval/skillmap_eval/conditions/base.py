"""Shared Condition protocol. All three implementations share LLM-B and the
outer loop; the ONLY difference is what gets injected into context.
"""

from __future__ import annotations

from typing import Protocol

from skillmap_eval.types import EvalTask, SimulatedTurn, TaskInteraction


class Condition(Protocol):
    name: str

    async def setup(self, profile_id: str, run_id: str) -> None: ...

    async def handle_user_message(
        self,
        task: EvalTask,
        conversation_so_far: list[SimulatedTurn],
        user_message: str,
    ) -> tuple[str, list[str]]:
        """Return (assistant_response, retrieved_skill_ids).

        retrieved_skill_ids is empty for non-skillmap conditions.
        """
        ...

    async def finalize_task(
        self,
        task: EvalTask,
        interaction: TaskInteraction,
    ) -> None: ...

    async def teardown(self) -> None: ...


SYSTEM_PROMPT_BASE = (
    "You are an expert Python coding assistant. Given a user task, respond "
    "with a working solution and a short explanation. When the user "
    "corrects you, adjust and continue without apologizing excessively."
)


def render_conversation_for_llm_b(
    turns: list[SimulatedTurn], latest_user_message: str
) -> list[dict]:
    """Shared helper: convert SimulatedTurn history + new user message into
    Bedrock Converse `messages` payload. Handles alternation.
    """
    messages: list[dict] = []
    for t in turns:
        messages.append({"role": t.role, "content": t.content})
    messages.append({"role": "user", "content": latest_user_message})
    return messages
