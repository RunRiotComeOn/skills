"""MatcherInserter: two-gate matching + branch to merge / split / insert.

Spec §3.2 branches:
  - Gate 1 fail                 -> insert_skill (brand-new skill)
  - Gate 1 pass, Gate 2 pass    -> merge_episode_into_skill
  - Gate 1 pass, Gate 2 fail    -> flag_pending_split (or commit_split
                                   once split_disambiguation_threshold is hit)
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from typing import Any

from skillmap.llm.client import LLMClient, LLMConfig, _extract_json
from skillmap.llm.prompts import MATCHING_GATE_1_PROMPT, MATCHING_GATE_2_PROMPT
from skillmap.storage.skill_map import SkillMap
from skillmap.types import Episode, Skill
from pydantic import BaseModel


class _Gate1Response(BaseModel):
    same_context: bool


class _Gate2Response(BaseModel):
    same_target: bool


class MatcherInserter:
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
                max_tokens=256,
                extra_inference={"temperature": 0.0},
            )
        )

    async def run(
        self,
        candidate: Skill,
        candidate_prereq_ids: list[str],
        episode: Episode,
        skill_map: SkillMap,
    ) -> str:
        """Decide the fate of `candidate` and return the skill_id the episode ends up in."""
        same_category = skill_map.list_skills(category=candidate.category)
        # Candidate has not been inserted yet - exclude it if present.
        same_category = [s for s in same_category if s.id != candidate.id]

        for existing in same_category:
            if existing.status == "deprecated":
                continue
            gate1 = await self._gate_1(candidate, existing)
            if not gate1:
                continue

            gate2 = await self._gate_2(candidate, existing)
            if gate2:
                # Both gates pass: merge evidence into existing skill.
                skill_map.merge_episode_into_skill(existing.id, episode.id)
                return existing.id

            # Gate 1 passed, Gate 2 failed -> ambiguity. Flag or split.
            skill_map.flag_pending_split(existing.id, episode.id)
            if (
                skill_map.pending_split_progress(existing.id)
                >= skill_map.split_disambiguation_threshold
            ):
                return await self._commit_split(existing, candidate, episode, skill_map)
            return existing.id

        # No match: insert as a brand-new skill.
        new_id, _ = skill_map.insert_skill(candidate, candidate_prereq_ids)
        return new_id

    # ------------------------------------------------------------------
    # Gate judges
    # ------------------------------------------------------------------

    async def _gate_1(self, candidate: Skill, existing: Skill) -> bool:
        parsed = await self._call_json(
            MATCHING_GATE_1_PROMPT.format(
                candidate_triggering_context=candidate.triggering_context,
                existing_triggering_context=existing.triggering_context,
            ),
            _Gate1Response,
        )
        return bool(parsed.get("same_context", False))

    async def _gate_2(self, candidate: Skill, existing: Skill) -> bool:
        parsed = await self._call_json(
            MATCHING_GATE_2_PROMPT.format(
                candidate_correction_target=candidate.correction_target,
                existing_correction_target=existing.correction_target,
            ),
            _Gate2Response,
        )
        return bool(parsed.get("same_target", False))

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
        raise RuntimeError(f"matcher inserter failed to return valid JSON after retries: {last_error}")

    # ------------------------------------------------------------------
    # Split construction
    # ------------------------------------------------------------------

    async def _commit_split(
        self,
        parent: Skill,
        latest_candidate: Skill,
        episode: Episode,
        skill_map: SkillMap,
    ) -> str:
        """Construct two sub-skills and commit the split.

        v0 simplification: sub_skill_a inherits the parent's original fields
        (with a fresh id and reset episode list); sub_skill_b adopts the
        latest_candidate's fields. Episode reassignment: all prior episodes
        go to sub_skill_a, the current inducing episode goes to sub_skill_b.
        Phase 5+ may replace this with an LLM-driven re-clustering.
        """
        now = datetime.now(timezone.utc)
        sub_a = Skill(
            id=str(uuid.uuid4()),
            name=parent.name + " (A)",
            status="confirmed" if len(parent.episode_ids) >= skill_map.confirmation_episode_threshold else "tentative",
            category=parent.category,
            triggering_context=parent.triggering_context,
            correction_target=parent.correction_target,
            covered_failures=list(parent.covered_failures),
            episode_ids=[],
            created_at=now,
            updated_at=now,
        )
        sub_b = Skill(
            id=str(uuid.uuid4()),
            name=latest_candidate.name + " (B)",
            status="tentative",
            category=parent.category,
            triggering_context=latest_candidate.triggering_context,
            correction_target=latest_candidate.correction_target,
            covered_failures=list(latest_candidate.covered_failures),
            episode_ids=[],
            created_at=now,
            updated_at=now,
        )
        reassignment: dict[str, str] = {eid: "a" for eid in parent.episode_ids}
        # The current episode is the one that completed the disambiguation.
        # Route it to sub_b by default.
        reassignment[episode.id] = "b"

        a_id, b_id = skill_map.commit_split(parent.id, sub_a, sub_b, reassignment)
        return b_id


def _parse_json_obj(raw: Any) -> dict:
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        return json.loads(_extract_json(raw))
    return raw or {}
