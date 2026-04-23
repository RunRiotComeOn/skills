"""Traverser: category root -> frontier.

Algorithm (spec §3.3):
  1. Classify query into a category (LLM).
  2. Seed frontier_candidates with category roots + "general" roots.
  3. For each level (BFS), LLM-judge applicability of every node.
  4. For matches, recurse into children; if any child matches, the parent
     drops off the frontier.
  5. Return maximal matching set s.t. no descendant also matches.

Invariants:
  - `tentative` skills are skipped by default (only `confirmed` traversed).
  - Deduplication across category + general roots.
"""

from __future__ import annotations

import json
from typing import Any

from skillmap.llm.client import LLMClient, LLMConfig, _extract_json
from skillmap.llm.prompts import (
    CATEGORY_CLASSIFICATION_PROMPT,
    TRAVERSAL_JUDGMENT_PROMPT,
)
from skillmap.storage.skill_map import SkillMap
from skillmap.types import Skill
from pydantic import BaseModel


class _CategoryClassificationResponse(BaseModel):
    category: str


class _TraversalJudgmentResponse(BaseModel):
    applies: bool


class Traverser:
    def __init__(
        self,
        categories: list[str],
        llm_model: str = "us.anthropic.claude-haiku-4-5-20251001-v1:0",
        region: str = "us-east-1",
        retry_max: int = 3,
    ) -> None:
        self.categories = categories
        self.retry_max = retry_max
        self._client = LLMClient(
            LLMConfig(
                provider="bedrock",
                model=llm_model,
                region=region,
                max_tokens=256,
                extra_inference={"temperature": 0.0},
            )
        )

    async def run(
        self,
        query: str,
        skill_map: SkillMap,
        include_tentative: bool = False,
    ) -> list[Skill]:
        category = await self._classify(query)
        seeds = list(skill_map.get_category_roots(category))
        if category != "general":
            for s in skill_map.get_category_roots("general"):
                if s.id not in {x.id for x in seeds}:
                    seeds.append(s)

        frontier: dict[str, Skill] = {}
        # Traverse from all roots regardless of their own status — roots
        # define structure. Status filter applies only when adding to frontier.
        level = list(seeds)
        visited: set[str] = set()

        while level:
            # Batch judge all nodes at this level.
            judgments = await self._judge_batch(query, level)
            next_level: list[Skill] = []
            for skill, applies in zip(level, judgments):
                visited.add(skill.id)
                if not applies:
                    continue
                # Only add to frontier if status qualifies; always expand
                # children so confirmed descendants of tentative nodes are reached.
                if include_tentative or skill.status == "confirmed":
                    frontier[skill.id] = skill
                children = [
                    c for c in skill_map.get_children(skill.id)
                    if c.id not in visited
                ]
                next_level.extend(children)

            if not next_level:
                break

            child_judgments = await self._judge_batch(query, next_level)
            surviving_children: list[Skill] = []
            for child, applies in zip(next_level, child_judgments):
                visited.add(child.id)
                if not applies:
                    continue
                surviving_children.append(child)
                # Demote confirmed parent and add confirmed child to frontier.
                if include_tentative or child.status == "confirmed":
                    for parent in skill_map.get_parents(child.id):
                        frontier.pop(parent.id, None)
                    frontier[child.id] = child

            # Next iteration: expand children of surviving children.
            level = []
            for child in surviving_children:
                for grandchild in skill_map.get_children(child.id):
                    if grandchild.id not in visited:
                        level.append(grandchild)

        return list(frontier.values())

    # ------------------------------------------------------------------
    # LLM judges
    # ------------------------------------------------------------------

    async def _classify(self, query: str) -> str:
        parsed = await self._call_json(
            CATEGORY_CLASSIFICATION_PROMPT.format(
                query=query,
                categories=", ".join(self.categories),
            ),
            _CategoryClassificationResponse,
        )
        cat = parsed.get("category", "general")
        return cat if cat in self.categories else "general"

    async def _judge_batch(self, query: str, skills: list[Skill]) -> list[bool]:
        """Concurrent per-skill judgments. v0: sequential for simplicity;
        Phase 4+ may swap to asyncio.gather for latency.
        """
        out: list[bool] = []
        for s in skills:
            parsed = await self._call_json(
                TRAVERSAL_JUDGMENT_PROMPT.format(
                    query=query,
                    skill_name=s.name,
                    skill_triggering_context=s.triggering_context,
                    skill_correction_target=s.correction_target,
                ),
                _TraversalJudgmentResponse,
            )
            out.append(bool(parsed.get("applies", False)))
        return out

    async def _call_json(self, prompt: str, schema: type[BaseModel]) -> dict[str, Any]:
        strict_suffix = (
            "\n\nCRITICAL: Return ONLY valid minified JSON. "
            "No prose, no markdown fence, no explanation."
        )
        last_error: Exception | None = None
        for attempt in range(self.retry_max):
            try:
                raw = await self._client.call(
                    messages=[
                        {
                            "role": "user",
                            "content": prompt if attempt == 0 else prompt + strict_suffix,
                        }
                    ],
                    response_schema=schema,
                )
                return _parse_json_obj(raw)
            except (json.JSONDecodeError, ValueError, RuntimeError) as e:
                last_error = e
                continue
        raise RuntimeError(f"traverser failed to return valid JSON after retries: {last_error}")

def _parse_json_obj(raw: Any) -> dict:
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        return json.loads(_extract_json(raw))
    return raw or {}
