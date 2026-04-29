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
        task_hint: str = "",
    ) -> str:
        system_prompt = self._render_system_prompt(selected_skills, task_hint)
        messages = list(conversation_history or [])
        messages.append({"role": "user", "content": query})
        response = await call_llm(messages, system=system_prompt)
        return response if isinstance(response, str) else str(response)

    @staticmethod
    def _render_system_prompt(skills: list[Skill], task_hint: str = "") -> str:
        base = (
            "You are an expert Python coding assistant. Given a user task, respond "
            "with a working solution and a short explanation. When the user "
            "corrects you, adjust and continue without apologizing excessively."
        )
        parts = [base]
        if task_hint:
            parts.append(task_hint)
        if skills:
            skills_block = "\n".join(
                f"{i+1}. [{s.axis}] {s.title}\n"
                f"   When: {s.catalog_trigger}\n"
                f"   Do: {s.guidance}"
                for i, s in enumerate(skills)
            )
            parts.append(
                CONSTRAINT_INJECTION_TEMPLATE.format(skills_block=skills_block)
            )
        return "\n\n".join(parts)
