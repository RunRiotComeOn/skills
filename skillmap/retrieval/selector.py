"""Stage C: select ≤ MAX_SKILLS_PER_TASK skills from catalog for the current task."""

from __future__ import annotations

import os
from typing import Any

from pydantic import BaseModel

from skillmap.llm.client import LLMClient, LLMConfig
from skillmap.llm.prompts import CATALOG_SELECTOR_PROMPT
from skillmap.storage.skill_map import SkillMap
from skillmap.types import CatalogEntry, Skill


MAX_SKILLS_PER_TASK = int(os.environ.get("SKILLMAP_TOP_K", "5"))


class _SelectorResponse(BaseModel):
    selected_ids: list[str]


class CatalogSelector:
    def __init__(
        self,
        llm_model: str = "us.anthropic.claude-haiku-4-5-20251001-v1:0",
        region: str = "us-east-1",
        retry_max: int = 3,
    ) -> None:
        self._client = LLMClient(
            LLMConfig(
                provider="bedrock",
                model=llm_model,
                region=region,
                max_tokens=256,
                extra_inference={"temperature": 0.0},
            )
        )
        self.retry_max = retry_max

    async def select(
        self,
        task_description: str,
        skill_map: SkillMap,
    ) -> list[Skill]:
        """Return ≤ MAX_SKILLS_PER_TASK skills relevant to task_description.

        The catalog is rendered with axis labels so the selector LLM can
        balance preference vs correctness picks.
        """
        catalog = skill_map.get_catalog()
        if not catalog:
            return []

        catalog_text = _render_catalog(catalog)
        prompt = CATALOG_SELECTOR_PROMPT.format(
            task=task_description,
            catalog=catalog_text,
            max_skills=MAX_SKILLS_PER_TASK,
        )

        last_exc: Exception | None = None
        for _ in range(self.retry_max):
            try:
                raw: Any = await self._client.call(
                    messages=[{"role": "user", "content": prompt}],
                    response_schema=_SelectorResponse,
                )
                ids = raw.get("selected_ids", [])[:MAX_SKILLS_PER_TASK]
                skills = []
                for sid in ids:
                    try:
                        skills.append(skill_map.get_skill(sid))
                    except Exception:
                        pass  # stale id — skip silently
                return skills
            except Exception as exc:
                last_exc = exc
        # On failure return empty — better to run without skills than crash
        return []


def _render_catalog(catalog: list[CatalogEntry]) -> str:
    return "\n".join(
        f"[{e.axis}] [{e.id}] {e.title}\n  trigger: {e.catalog_trigger}"
        for e in catalog
    )
