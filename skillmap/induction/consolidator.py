"""Stage B: consolidate CorrectionSummary buffer into Skill entries."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel

from skillmap.llm.client import LLMClient, LLMConfig
from skillmap.llm.prompts import (
    SKILL_CANDIDATE_EXTRACTION_PROMPT,
    SKILL_RECONCILIATION_PROMPT,
)
from skillmap.storage.skill_map import SkillMap
from skillmap.types import CorrectionSummary, Skill


MIN_SUPPORT = 3


# ---------------------------------------------------------------------------
# Pydantic schemas for LLM responses
# ---------------------------------------------------------------------------

class _CandidateItem(BaseModel):
    title: str
    catalog_trigger: str
    guidance: str
    supporting_summary_ids: list[str]


class _CandidatesResponse(BaseModel):
    candidates: list[_CandidateItem]


class _DecisionItem(BaseModel):
    proposed_index: int
    action: str                        # "discard" | "update" | "add"
    existing_skill_id: str | None
    updated_guidance: str | None


class _ReconciliationResponse(BaseModel):
    decisions: list[_DecisionItem]


# ---------------------------------------------------------------------------
# Consolidator
# ---------------------------------------------------------------------------

class SkillConsolidator:
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
                max_tokens=2048,
                extra_inference={"temperature": 0.0},
            )
        )
        self.retry_max = retry_max

    async def run(self, skill_map: SkillMap) -> None:
        """Consume the full summary buffer and update skill_map in-place."""
        summaries = skill_map.get_pending_summaries()
        if not summaries:
            return

        # Step 1: extract candidates from buffer
        candidates = await self._extract_candidates(summaries)
        if not candidates:
            skill_map.clear_summaries()
            return

        # Step 2: reconcile against existing skills (skip if map is empty)
        existing_catalog = skill_map.get_catalog()
        if existing_catalog:
            decisions = await self._reconcile(candidates, existing_catalog)
        else:
            decisions = [
                {"proposed_index": i, "action": "add",
                 "existing_skill_id": None, "updated_guidance": None}
                for i in range(len(candidates))
            ]

        now = datetime.now(timezone.utc)
        summary_ids = [s.id for s in summaries]

        for dec in decisions:
            idx = dec["proposed_index"]
            if idx >= len(candidates):
                continue
            cand = candidates[idx]
            action = dec.get("action", "add")

            if action == "discard":
                continue

            elif action == "update":
                existing_id = dec.get("existing_skill_id")
                new_guidance = dec.get("updated_guidance") or cand.guidance
                if existing_id and existing_id in {s.id for s in skill_map.list_skills()}:
                    new_ids = [
                        sid for sid in cand.supporting_summary_ids
                        if sid in summary_ids
                    ]
                    skill_map.update_skill(existing_id, new_guidance, new_ids)

            else:  # "add"
                valid_ids = [
                    sid for sid in cand.supporting_summary_ids
                    if sid in summary_ids
                ]
                if len(valid_ids) < MIN_SUPPORT:
                    continue
                skill = Skill(
                    id=str(uuid.uuid4()),
                    title=cand.title,
                    catalog_trigger=cand.catalog_trigger,
                    guidance=cand.guidance,
                    support_count=len(valid_ids),
                    supporting_summary_ids=valid_ids,
                    created_at=now,
                    updated_at=now,
                )
                skill_map.insert_skill(skill)

        skill_map.clear_summaries()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _extract_candidates(
        self, summaries: list[CorrectionSummary]
    ) -> list[_CandidateItem]:
        formatted = _format_summaries(summaries)
        prompt = SKILL_CANDIDATE_EXTRACTION_PROMPT.format(
            summaries=formatted,
            min_support=MIN_SUPPORT,
        )
        last_exc: Exception | None = None
        for _ in range(self.retry_max):
            try:
                raw: Any = await self._client.call(
                    messages=[{"role": "user", "content": prompt}],
                    response_schema=_CandidatesResponse,
                )
                return [_CandidateItem(**c) for c in raw.get("candidates", [])]
            except Exception as exc:
                last_exc = exc
        raise RuntimeError(f"candidate extraction failed after retries: {last_exc}") from last_exc

    async def _reconcile(
        self,
        candidates: list[_CandidateItem],
        existing_catalog: list,
    ) -> list[dict]:
        catalog_text = "\n".join(
            f"{i+1}. [{e.id}] {e.title} | {e.catalog_trigger}"
            for i, e in enumerate(existing_catalog)
        )
        proposed_text = "\n".join(
            f"{i}. title={c.title!r} | trigger={c.catalog_trigger!r} | guidance={c.guidance!r}"
            for i, c in enumerate(candidates)
        )
        prompt = SKILL_RECONCILIATION_PROMPT.format(
            existing_catalog=catalog_text,
            proposed_skills=proposed_text,
        )
        last_exc: Exception | None = None
        for _ in range(self.retry_max):
            try:
                raw: Any = await self._client.call(
                    messages=[{"role": "user", "content": prompt}],
                    response_schema=_ReconciliationResponse,
                )
                return raw.get("decisions", [])
            except Exception as exc:
                last_exc = exc
        raise RuntimeError(f"reconciliation failed after retries: {last_exc}") from last_exc


def _format_summaries(summaries: list[CorrectionSummary]) -> str:
    parts = []
    for s in summaries:
        parts.append(
            f"ID: {s.id}\n"
            f"  situation: {s.triggering_situation}\n"
            f"  wrong: {s.what_was_wrong}\n"
            f"  wanted: {s.what_user_wanted}\n"
            f"  quote: {s.correction_quote}"
        )
    return "\n\n".join(parts)
