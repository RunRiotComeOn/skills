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

    def deprecate_skill(self, skill_id: str) -> None:
        """Mark a preference skill as past (superseded by a conflicting newer skill).

        Past skills are retained for audit but excluded from the catalog and
        never selected at task time.
        """
        skill = self._require_skill(skill_id)
        skill.status = "past"
        skill.updated_at = _UTC_NOW()
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
            CatalogEntry(
                id=s.id,
                title=s.title,
                catalog_trigger=s.catalog_trigger,
                axis=s.axis,
            )
            for s in sorted(
                (s for s in self._state.skills.values() if s.status == "active"),
                key=lambda s: s.support_count,
                reverse=True,
            )
        ]

    def _persist(self) -> None:
        self._persistence.save(self._state)
