"""Stage B: consolidate CorrectionSummary buffer into Skill entries.

Two-axis processing
-------------------
The buffer holds summaries from BOTH axes (preference and correctness).
The consolidator splits the buffer by axis and runs the full Stage B
pipeline (extract → dedup → reconcile) ONCE PER AXIS. Reconciliation
operates within an axis: a preference skill is never compared against or
replaced by a correctness skill (and vice versa). The two axes track
different metrics and must remain independently inspectable.

Containment / conflict handling (done at reconcile time, not batch-end)
-----------------------------------------------------------------------
When a candidate covers the same habit as an existing skill (same or
containment relationship), the LLM issues "update" and the existing skill
is updated in-place with the best-of-both guidance.

When a candidate CONTRADICTS an existing PREFERENCE skill, the LLM issues
"conflict": the old skill is deprecated (status → "past", removed from
catalog) and the new skill is inserted as active.

For CORRECTNESS contradictions the LLM issues "replace": the existing
skill is updated in-place with the new guidance (no "past" archival —
one of the two was objectively wrong, so there is no value in keeping it).

Per-axis MIN_SUPPORT
--------------------
The default MIN_SUPPORT is shared across axes for now, but the knob is
per-axis so that correctness can be tightened (it carries a higher false-
positive cost — a bad correctness skill nudges the model toward an
incorrect "fix") without affecting preference recall.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel

from skillmap.llm.client import LLMClient, LLMConfig
from skillmap.llm.prompts import (
    AXIS_GUIDANCE_CORRECTNESS,
    AXIS_GUIDANCE_PREFERENCE,
    SKILL_CANDIDATE_EXTRACTION_PROMPT,
    SKILL_DEDUP_PROMPT,
    SKILL_RECONCILIATION_PROMPT,
)
from skillmap.storage.skill_map import SkillMap
from skillmap.types import CorrectionSummary, Skill, SkillAxis


# Per-axis MIN_SUPPORT. Correctness is held to the same bar as preference
# by default; raise the correctness threshold if false-positive bug-class
# skills become a problem in eval.
MIN_SUPPORT_BY_AXIS: dict[SkillAxis, int] = {
    "preference": 3,
    "correctness": 3,
}

_AXIS_GUIDANCE: dict[SkillAxis, str] = {
    "preference": AXIS_GUIDANCE_PREFERENCE,
    "correctness": AXIS_GUIDANCE_CORRECTNESS,
}


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
        """Consume the full summary buffer and update skill_map in-place.

        Processes each axis independently — see module docstring.
        """
        summaries = skill_map.get_pending_summaries()
        if not summaries:
            return

        by_axis: dict[SkillAxis, list[CorrectionSummary]] = {
            "preference": [s for s in summaries if s.correction_type == "preference"],
            "correctness": [s for s in summaries if s.correction_type == "correctness"],
        }
        for axis, axis_summaries in by_axis.items():
            if axis_summaries:
                await self._run_axis(skill_map, axis, axis_summaries)

        skill_map.clear_summaries()

    async def _run_axis(
        self,
        skill_map: SkillMap,
        axis: SkillAxis,
        summaries: list[CorrectionSummary],
    ) -> None:
        min_support = MIN_SUPPORT_BY_AXIS[axis]

        # Step 1: extract candidates from this axis's slice of the buffer
        candidates = await self._extract_candidates(axis, summaries, min_support)
        if not candidates:
            return

        # Step 1b: merge near-duplicates within the batch
        if len(candidates) > 1:
            candidates = await self._dedup_candidates(axis, candidates)

        # Step 2: reconcile against existing skills OF THE SAME AXIS only
        existing_axis_skills = [s for s in skill_map.list_skills() if s.axis == axis]
        if existing_axis_skills:
            decisions = await self._reconcile(axis, candidates, existing_axis_skills)
        else:
            decisions = [
                {"proposed_index": i, "action": "add",
                 "existing_skill_id": None, "updated_guidance": None}
                for i in range(len(candidates))
            ]

        now = datetime.now(timezone.utc)
        summary_ids = {s.id for s in summaries}
        existing_axis_ids = {s.id for s in existing_axis_skills}

        for dec in decisions:
            idx = dec["proposed_index"] if isinstance(dec, dict) else dec.proposed_index
            if idx >= len(candidates):
                continue
            cand = candidates[idx]
            action = dec.get("action") if isinstance(dec, dict) else dec.action

            if action == "discard":
                continue

            elif action == "update":
                # Same habit, containment, or correctness-axis override — all
                # update existing skill in place. Cross-axis updates forbidden.
                existing_id = (
                    dec.get("existing_skill_id") if isinstance(dec, dict)
                    else dec.existing_skill_id
                )
                new_guidance = (
                    (dec.get("updated_guidance") if isinstance(dec, dict) else dec.updated_guidance)
                    or cand.guidance
                )
                if existing_id and existing_id in existing_axis_ids:
                    new_ids = [sid for sid in cand.supporting_summary_ids if sid in summary_ids]
                    skill_map.update_skill(existing_id, new_guidance, new_ids)

            elif action == "conflict":
                # Preference-only: user's preference has changed. Deprecate the
                # old skill (archive as "past") and insert the new one as active.
                # Using "conflict" for correctness is a prompt error — fall back
                # to "replace" semantics (update in place) to avoid data loss.
                existing_id = (
                    dec.get("existing_skill_id") if isinstance(dec, dict)
                    else dec.existing_skill_id
                )
                new_guidance = (
                    (dec.get("updated_guidance") if isinstance(dec, dict) else dec.updated_guidance)
                    or cand.guidance
                )
                if existing_id and existing_id in existing_axis_ids:
                    if axis == "preference":
                        skill_map.deprecate_skill(existing_id)
                        valid_ids = [sid for sid in cand.supporting_summary_ids if sid in summary_ids]
                        if len(valid_ids) >= min_support:
                            skill = Skill(
                                id=str(uuid.uuid4()),
                                title=cand.title,
                                catalog_trigger=cand.catalog_trigger,
                                guidance=new_guidance,
                                axis=axis,
                                support_count=len(valid_ids),
                                supporting_summary_ids=valid_ids,
                                created_at=now,
                                updated_at=now,
                            )
                            skill_map.insert_skill(skill)
                    else:
                        new_ids = [sid for sid in cand.supporting_summary_ids if sid in summary_ids]
                        skill_map.update_skill(existing_id, new_guidance, new_ids)

            else:  # "add"
                valid_ids = [sid for sid in cand.supporting_summary_ids if sid in summary_ids]
                if len(valid_ids) < min_support:
                    continue
                skill = Skill(
                    id=str(uuid.uuid4()),
                    title=cand.title,
                    catalog_trigger=cand.catalog_trigger,
                    guidance=cand.guidance,
                    axis=axis,
                    support_count=len(valid_ids),
                    supporting_summary_ids=valid_ids,
                    created_at=now,
                    updated_at=now,
                )
                skill_map.insert_skill(skill)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _extract_candidates(
        self,
        axis: SkillAxis,
        summaries: list[CorrectionSummary],
        min_support: int,
    ) -> list[_CandidateItem]:
        formatted = _format_summaries(summaries)
        prompt = SKILL_CANDIDATE_EXTRACTION_PROMPT.format(
            axis=axis,
            axis_guidance=_AXIS_GUIDANCE[axis],
            summaries=formatted,
            min_support=min_support,
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
        raise RuntimeError(
            f"candidate extraction failed (axis={axis}) after retries: {last_exc}"
        ) from last_exc

    async def _dedup_candidates(
        self,
        axis: SkillAxis,
        candidates: list[_CandidateItem],
    ) -> list[_CandidateItem]:
        """Merge near-duplicate candidates within a single batch before reconciliation."""
        formatted = "\n".join(
            f"{i}. title={c.title!r} | trigger={c.catalog_trigger!r} | guidance={c.guidance!r}"
            for i, c in enumerate(candidates)
        )
        prompt = SKILL_DEDUP_PROMPT.format(axis=axis, candidates=formatted)

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
        print(f"[consolidator:{axis}] dedup failed, skipping: {last_exc}")
        return candidates

    async def _reconcile(
        self,
        axis: SkillAxis,
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
            axis=axis,
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
        raise RuntimeError(
            f"reconciliation failed (axis={axis}) after retries: {last_exc}"
        ) from last_exc


def _format_summaries(summaries: list[CorrectionSummary]) -> str:
    parts = []
    for s in summaries:
        block = (
            f"ID: {s.id}\n"
            f"  situation: {s.triggering_situation}\n"
            f"  wrong: {s.what_was_wrong}\n"
            f"  wanted: {s.what_user_wanted}\n"
            f"  quote: {s.correction_quote}"
        )
        if s.correction_type == "correctness" and s.verification_evidence:
            block += f"\n  evidence: {s.verification_evidence}"
        parts.append(block)
    return "\n\n".join(parts)
