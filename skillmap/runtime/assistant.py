"""Assistant: inject pre-selected skills and call LLM-B."""

from __future__ import annotations

from skillmap.llm import call_llm
from skillmap.llm.prompts import CONSTRAINT_INJECTION_TEMPLATE
from skillmap.types import Skill


class Assistant:
    async def run(
        self,
        query: str,
        selected_skills: list[Skill],
        conversation_history: list[dict] | None = None,
    ) -> str:
        system_prompt = self._render_system_prompt(selected_skills)
        messages = list(conversation_history or [])
        messages.append({"role": "user", "content": query})
        response = await call_llm(messages, system=system_prompt)
        return response if isinstance(response, str) else str(response)

    @staticmethod
    def _render_system_prompt(skills: list[Skill]) -> str:
        base = (
            "You are a concise Python coding assistant. Respond with the final "
            "solution directly. Do not produce long deliberation or step-by-step "
            "reasoning. Keep any explanation brief."
        )
        if not skills:
            return base
        skills_block = "\n".join(
            f"{i+1}. {s.title}\n"
            f"   When: {s.catalog_trigger}\n"
            f"   Do: {s.guidance}"
            for i, s in enumerate(skills)
        )
        return base + "\n\n" + CONSTRAINT_INJECTION_TEMPLATE.format(skills_block=skills_block)
