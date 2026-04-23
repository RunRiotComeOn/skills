"""SkillProposer: propose a Skill from a correction point, with abstraction check.

Returns None if the LLM's covered_failures field has < 2 entries (the
abstraction check documented in spec §3.2).
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from typing import Any

from skillmap.llm.client import LLMClient, LLMConfig, _extract_json
from skillmap.llm.prompts import SKILL_PROPOSAL_PROMPT
from skillmap.types import CorrectionPoint, Episode, Skill
from pydantic import BaseModel


class _SkillProposalResponse(BaseModel):
    should_create: bool
    name: str
    triggering_context: str
    correction_target: str
    covered_failures: list[str]
    candidate_prereq_ids: list[str]


class SkillProposer:
    def __init__(
        self,
        llm_model: str = "us.anthropic.claude-haiku-4-5-20251001-v1:0",
        region: str = "us-east-1",
        retry_max: int = 3,
    ) -> None:
        self.retry_max = retry_max
        self._client = LLMClient(
            LLMConfig(
                provider="bedrock",
                model=llm_model,
                region=region,
                max_tokens=768,
                extra_inference={"temperature": 0.0},
            )
        )

    async def run(
        self,
        episode: Episode,
        correction: CorrectionPoint,
        existing_skills_in_category: list[Skill],
    ) -> tuple[Skill, list[str]] | None:
        """Return (proposed_skill, candidate_prereq_ids) or None on abstraction failure."""
        catalog = _render_catalog(existing_skills_in_category)
        trajectory_text = _render_trajectory(episode)
        correction_text = f"[turn {correction.turn_index}] {correction.correction_type}: {correction.correction_content}"

        parsed = await self._call_json(
            SKILL_PROPOSAL_PROMPT.format(
                trajectory=trajectory_text,
                correction=correction_text,
                existing_skill_names=catalog,
            ),
            _SkillProposalResponse,
        )
        parsed = _parse_proposal(parsed)
        if parsed is None:
            return None

        covered_failures = parsed.get("covered_failures") or []
        if len(covered_failures) < 2:
            return None  # abstraction check failed

        now = datetime.now(timezone.utc)
        skill = Skill(
            id=str(uuid.uuid4()),
            name=parsed["name"],
            status="tentative",
            category=episode.task_category,
            triggering_context=parsed["triggering_context"],
            correction_target=parsed["correction_target"],
            covered_failures=covered_failures,
            episode_ids=[episode.id],
            created_at=now,
            updated_at=now,
        )
        candidate_prereq_ids = list(parsed.get("candidate_prereq_ids") or [])
        return skill, candidate_prereq_ids

    async def _call_json(self, prompt: str, schema: type[BaseModel]) -> dict[str, Any]:
        strict_suffix = (
            "\n\nCRITICAL: Return ONLY valid JSON. "
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
        raise RuntimeError(f"skill proposer failed to return valid JSON after retries: {last_error}")


def _render_catalog(skills: list[Skill]) -> str:
    if not skills:
        return "(no existing skills in this category)"
    return "\n".join(f"- {s.id}: {s.name} - {s.triggering_context}" for s in skills)


def _render_trajectory(episode: Episode) -> str:
    return "\n".join(
        f"[{i}] {t.role}: {t.content}" for i, t in enumerate(episode.trajectory)
    )


def _parse_proposal(raw: Any) -> dict | None:
    data = _parse_json_obj(raw)
    if not bool(data.get("should_create", False)):
        return None
    return data


def _parse_json_obj(raw: Any) -> dict:
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        return json.loads(_extract_json(raw))
    return raw or {}
