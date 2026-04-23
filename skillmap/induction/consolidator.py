"""Stage B: consolidate CorrectionSummary buffer into Skill entries."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel

from skillmap.llm.client import LLMClient, LLMConfig
from skillmap.llm.prompts import (
    SKILL_CANDIDATE_EXTRACTION_PROMPT,
    SKILL_COMPACTION_PROMPT,
    SKILL_DEDUP_PROMPT,
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


class _MergedItem(BaseModel):
    title: str
    catalog_trigger: str
    guidance: str
    source_indices: list[int]


class _DedupResponse(BaseModel):
    merged: list[_MergedItem]


class _DecisionItem(BaseModel):
    proposed_index: int
    action: str  # "discard" | "update" | "replace" | "add"
    existing_skill_id: str | None
    updated_guidance: str | None


class _ReconciliationResponse(BaseModel):
    decisions: list[_DecisionItem]


class _CompactionGroup(BaseModel):
    keep_id: str
    title: str
    catalog_trigger: str
    guidance: str
    discard_ids: list[str]


class _CompactionResponse(BaseModel):
    groups: list[_CompactionGroup]


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
                max_tokens=4096,
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

        # Step 1b: merge near-duplicates within the batch
        if len(candidates) > 1:
            candidates = await self._dedup_candidates(candidates)

        # Step 2: reconcile against existing skills
        existing_skills = skill_map.list_skills()
        if existing_skills:
            decisions = await self._reconcile(candidates, existing_skills)
        else:
            decisions = [
                {"proposed_index": i, "action": "add",
                 "existing_skill_id": None, "updated_guidance": None}
                for i in range(len(candidates))
            ]

        now = datetime.now(timezone.utc)
        summary_ids = {s.id for s in summaries}
        existing_ids = {s.id for s in skill_map.list_skills()}

        for dec in decisions:
            idx = dec["proposed_index"] if isinstance(dec, dict) else dec.proposed_index
            if idx >= len(candidates):
                continue
            cand = candidates[idx]
            action = dec.get("action") if isinstance(dec, dict) else dec.action

            if action == "discard":
                continue

            elif action in ("update", "replace"):
                existing_id = (
                    dec.get("existing_skill_id") if isinstance(dec, dict)
                    else dec.existing_skill_id
                )
                new_guidance = (
                    (dec.get("updated_guidance") if isinstance(dec, dict) else dec.updated_guidance)
                    or cand.guidance
                )
                if existing_id and existing_id in existing_ids:
                    new_ids = [sid for sid in cand.supporting_summary_ids if sid in summary_ids]
                    skill_map.update_skill(existing_id, new_guidance, new_ids)

            else:  # "add"
                valid_ids = [sid for sid in cand.supporting_summary_ids if sid in summary_ids]
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

    async def compact(self, skill_map: SkillMap) -> None:
        """Full catalog compaction: merge overlapping or contradictory skills."""
        skills = skill_map.list_skills()
        if len(skills) <= 1:
            return

        formatted = "\n\n".join(
            f"id: {s.id}\ntitle: {s.title!r}\ntrigger: {s.catalog_trigger!r}\n"
            f"guidance: {s.guidance!r}\nsupport_count: {s.support_count}"
            for s in skills
        )
        prompt = SKILL_COMPACTION_PROMPT.format(skills=formatted)

        last_exc: Exception | None = None
        for _ in range(self.retry_max):
            try:
                raw: Any = await self._client.call(
                    messages=[{"role": "user", "content": prompt}],
                    response_schema=_CompactionResponse,
                )
                groups = raw.get("groups", [])
                current_ids = {s.id for s in skill_map.list_skills()}

                for group in groups:
                    keep_id = group.get("keep_id") if isinstance(group, dict) else group.keep_id
                    title = group.get("title") if isinstance(group, dict) else group.title
                    trigger = group.get("catalog_trigger") if isinstance(group, dict) else group.catalog_trigger
                    guidance = group.get("guidance") if isinstance(group, dict) else group.guidance
                    discard_ids = group.get("discard_ids", []) if isinstance(group, dict) else group.discard_ids

                    if not keep_id or keep_id not in current_ids:
                        continue
                    valid_discards = [d for d in discard_ids if d != keep_id and d in current_ids]
                    try:
                        skill_map.merge_skill(
                            primary_id=keep_id,
                            updated_title=title,
                            updated_trigger=trigger,
                            updated_guidance=guidance,
                            absorb_ids=valid_discards,
                        )
                        current_ids -= set(valid_discards)
                    except Exception as exc:
                        print(f"[compactor] merge failed for {keep_id}: {exc}")
                return
            except Exception as exc:
                last_exc = exc
        print(f"[compactor] compaction failed after {self.retry_max} retries: {last_exc}")

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

    async def _dedup_candidates(
        self, candidates: list[_CandidateItem]
    ) -> list[_CandidateItem]:
        """Merge near-duplicate candidates within a single batch before reconciliation."""
        formatted = "\n".join(
            f"{i}. title={c.title!r} | trigger={c.catalog_trigger!r} | guidance={c.guidance!r}"
            for i, c in enumerate(candidates)
        )
        prompt = SKILL_DEDUP_PROMPT.format(candidates=formatted)

        last_exc: Exception | None = None
        for _ in range(self.retry_max):
            try:
                raw: Any = await self._client.call(
                    messages=[{"role": "user", "content": prompt}],
                    response_schema=_DedupResponse,
                )
                merged_items = raw.get("merged", [])
                result: list[_CandidateItem] = []
                covered: set[int] = set()

                for m in merged_items:
                    src = m.get("source_indices", []) if isinstance(m, dict) else m.source_indices
                    if not src:
                        continue
                    # Union of supporting summary IDs from all source candidates
                    combined_ids: list[str] = []
                    seen_ids: set[str] = set()
                    for idx in src:
                        if 0 <= idx < len(candidates):
                            covered.add(idx)
                            for sid in candidates[idx].supporting_summary_ids:
                                if sid not in seen_ids:
                                    combined_ids.append(sid)
                                    seen_ids.add(sid)
                    title = m.get("title") if isinstance(m, dict) else m.title
                    trigger = m.get("catalog_trigger") if isinstance(m, dict) else m.catalog_trigger
                    guidance = m.get("guidance") if isinstance(m, dict) else m.guidance
                    result.append(_CandidateItem(
                        title=title,
                        catalog_trigger=trigger,
                        guidance=guidance,
                        supporting_summary_ids=combined_ids,
                    ))

                # Pass through any candidates the LLM didn't mention
                for i, c in enumerate(candidates):
                    if i not in covered:
                        result.append(c)
                return result
            except Exception as exc:
                last_exc = exc

        # Fallback: return original list unchanged
        print(f"[consolidator] dedup failed, skipping: {last_exc}")
        return candidates

    async def _reconcile(
        self,
        candidates: list[_CandidateItem],
        existing_skills: list[Skill],
    ) -> list[dict]:
        catalog_text = "\n\n".join(
            f"{i+1}. [{s.id}]\n"
            f"   title: {s.title}\n"
            f"   trigger: {s.catalog_trigger}\n"
            f"   guidance: {s.guidance}"
            for i, s in enumerate(existing_skills)
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
