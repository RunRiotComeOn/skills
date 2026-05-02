"""Phase 1 gate tests: SkillMap structural operations + persistence round-trip."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from pathlib import Path

import pytest

from skillmap.storage import JSONPersistence, SkillMap
from skillmap.types import Skill, SkillMapState, UnknownSkillError


def _make_skill(
    title: str,
    axis: str = "preference",
    summary_ids: list[str] | None = None,
) -> Skill:
    now = datetime.now(timezone.utc)
    ids = list(summary_ids or [str(uuid.uuid4())])
    return Skill(
        id=str(uuid.uuid4()),
        title=title,
        catalog_trigger=f"when {title}",
        guidance=f"do {title} correctly",
        axis=axis,
        support_count=len(ids),
        supporting_summary_ids=ids,
        created_at=now,
        updated_at=now,
    )


@pytest.fixture
def sm(tmp_path: Path) -> SkillMap:
    persistence = JSONPersistence(tmp_path / "sm.json", user_id="u1")
    return SkillMap(persistence.load(), persistence)


def test_insert_skill_appears_in_catalog(sm: SkillMap) -> None:
    skill = _make_skill("state complexity")
    sm.insert_skill(skill)
    catalog_ids = [e.id for e in sm.get_catalog()]
    assert skill.id in catalog_ids


def test_update_skill_accumulates_evidence(sm: SkillMap) -> None:
    skill = _make_skill("name pattern", summary_ids=["s1", "s2", "s3"])
    sm.insert_skill(skill)
    sm.update_skill(skill.id, "updated guidance", ["s4", "s5"])
    updated = sm.get_skill(skill.id)
    assert updated.support_count == 5
    assert set(updated.supporting_summary_ids) == {"s1", "s2", "s3", "s4", "s5"}
    assert updated.guidance == "updated guidance"


def test_update_skill_deduplicates_summary_ids(sm: SkillMap) -> None:
    skill = _make_skill("edge cases", summary_ids=["s1", "s2"])
    sm.insert_skill(skill)
    sm.update_skill(skill.id, "same guidance", ["s2", "s3"])
    updated = sm.get_skill(skill.id)
    assert updated.support_count == 3
    assert updated.supporting_summary_ids.count("s2") == 1


def test_delete_skill_removes_from_catalog(sm: SkillMap) -> None:
    skill = _make_skill("delete me")
    sm.insert_skill(skill)
    sm.delete_skill(skill.id)
    catalog_ids = [e.id for e in sm.get_catalog()]
    assert skill.id not in catalog_ids
    with pytest.raises(UnknownSkillError):
        sm.get_skill(skill.id)


def test_delete_unknown_skill_raises(sm: SkillMap) -> None:
    with pytest.raises(UnknownSkillError):
        sm.delete_skill("does-not-exist")


def test_deprecate_skill_removes_from_catalog_but_keeps_in_store(sm: SkillMap) -> None:
    skill = _make_skill("old preference")
    sm.insert_skill(skill)
    sm.deprecate_skill(skill.id)
    # catalog must not include it
    catalog_ids = [e.id for e in sm.get_catalog()]
    assert skill.id not in catalog_ids
    # but it's still retrievable with status=past
    stored = sm.get_skill(skill.id)
    assert stored.status == "past"


def test_catalog_only_shows_active_skills(sm: SkillMap) -> None:
    active = _make_skill("active skill")
    past = _make_skill("past skill")
    sm.insert_skill(active)
    sm.insert_skill(past)
    sm.deprecate_skill(past.id)
    catalog_ids = [e.id for e in sm.get_catalog()]
    assert active.id in catalog_ids
    assert past.id not in catalog_ids


def test_catalog_sorted_by_support_count_descending(sm: SkillMap) -> None:
    low = _make_skill("low support", summary_ids=["s1", "s2"])
    high = _make_skill("high support", summary_ids=["s1", "s2", "s3", "s4"])
    sm.insert_skill(low)
    sm.insert_skill(high)
    catalog_ids = [e.id for e in sm.get_catalog()]
    assert catalog_ids.index(high.id) < catalog_ids.index(low.id)


def test_persistence_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "sm.json"
    persistence = JSONPersistence(path, user_id="u1")
    sm = SkillMap(persistence.load(), persistence)

    a = _make_skill("skill a")
    b = _make_skill("skill b", axis="correctness")
    sm.insert_skill(a)
    sm.insert_skill(b)
    sm.deprecate_skill(a.id)

    sm2 = SkillMap(JSONPersistence(path, user_id="u1").load(), persistence)
    assert {s.id for s in sm2.list_skills()} == {a.id, b.id}
    assert sm2.get_skill(a.id).status == "past"
    assert [e.id for e in sm2.get_catalog()] == [b.id]
