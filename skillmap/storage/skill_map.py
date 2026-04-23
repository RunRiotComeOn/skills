"""SkillMap: flat experience library. No DAG, no edges."""

from __future__ import annotations

from datetime import datetime, timezone

from skillmap.storage.persistence import PersistenceBackend
from skillmap.types import (
    CatalogEntry,
    CorrectionSummary,
    Skill,
    SkillMapState,
    UnknownSkillError,
)

_UTC_NOW = lambda: datetime.now(timezone.utc)  # noqa: E731


class SkillMap:
    def __init__(self, state: SkillMapState, persistence: PersistenceBackend) -> None:
        self._state = state
        self._persistence = persistence

    # ------------------------------------------------------------------
    # Summary buffer
    # ------------------------------------------------------------------

    def append_summary(self, summary: CorrectionSummary) -> None:
        self._state.summary_buffer.append(summary)
        self._persist()

    @property
    def pending_summary_count(self) -> int:
        return len(self._state.summary_buffer)

    def get_pending_summaries(self) -> list[CorrectionSummary]:
        return list(self._state.summary_buffer)

    def clear_summaries(self) -> None:
        self._state.summary_buffer = []
        self._persist()

    # ------------------------------------------------------------------
    # Skills
    # ------------------------------------------------------------------

    def insert_skill(self, skill: Skill) -> str:
        self._state.skills[skill.id] = skill
        self._rebuild_catalog()
        self._persist()
        return skill.id

    def update_skill(
        self,
        skill_id: str,
        updated_guidance: str,
        new_summary_ids: list[str],
    ) -> None:
        skill = self._require_skill(skill_id)
        existing = set(skill.supporting_summary_ids)
        for sid in new_summary_ids:
            if sid not in existing:
                skill.supporting_summary_ids.append(sid)
                existing.add(sid)
        skill.support_count = len(skill.supporting_summary_ids)
        skill.guidance = updated_guidance
        skill.updated_at = _UTC_NOW()
        self._rebuild_catalog()
        self._persist()

    def get_skill(self, skill_id: str) -> Skill:
        return self._require_skill(skill_id)

    def list_skills(self) -> list[Skill]:
        return list(self._state.skills.values())

    def get_catalog(self) -> list[CatalogEntry]:
        return list(self._state.catalog)

    def delete_skill(self, skill_id: str) -> None:
        if skill_id not in self._state.skills:
            raise UnknownSkillError(f"unknown skill_id: {skill_id!r}")
        del self._state.skills[skill_id]
        self._rebuild_catalog()
        self._persist()

    def merge_skill(
        self,
        primary_id: str,
        updated_title: str,
        updated_trigger: str,
        updated_guidance: str,
        absorb_ids: list[str],
    ) -> None:
        """Update primary skill content and absorb (delete) others, merging their evidence."""
        primary = self._require_skill(primary_id)
        primary.title = updated_title
        primary.catalog_trigger = updated_trigger
        primary.guidance = updated_guidance
        primary.updated_at = _UTC_NOW()

        existing_ids = set(primary.supporting_summary_ids)
        for aid in absorb_ids:
            absorbed = self._state.skills.pop(aid, None)
            if absorbed:
                for sid in absorbed.supporting_summary_ids:
                    if sid not in existing_ids:
                        primary.supporting_summary_ids.append(sid)
                        existing_ids.add(sid)
        primary.support_count = len(primary.supporting_summary_ids)
        self._rebuild_catalog()
        self._persist()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _require_skill(self, skill_id: str) -> Skill:
        skill = self._state.skills.get(skill_id)
        if skill is None:
            raise UnknownSkillError(f"unknown skill_id: {skill_id!r}")
        return skill

    def _rebuild_catalog(self) -> None:
        self._state.catalog = [
            CatalogEntry(id=s.id, title=s.title, catalog_trigger=s.catalog_trigger)
            for s in sorted(
                self._state.skills.values(),
                key=lambda s: s.support_count,
                reverse=True,
            )
        ]

    def _persist(self) -> None:
        self._persistence.save(self._state)
