"""Phase 1 gate tests: SkillMap structural operations + persistence round-trip."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from pathlib import Path

import pytest

from skillmap.storage import JSONPersistence, SkillMap
from skillmap.types import (
    DAGIntegrityError,
    Skill,
    SkillMapState,
    UnknownSkillError,
)


def _make_skill(name: str, category: str = "coding", episode_ids: list[str] | None = None) -> Skill:
    now = datetime.now(timezone.utc)
    return Skill(
        id=str(uuid.uuid4()),
        name=name,
        status="tentative",
        category=category,
        triggering_context=f"when {name}",
        correction_target=f"avoid failing at {name}",
        covered_failures=[f"failure_a_{name}", f"failure_b_{name}"],
        episode_ids=list(episode_ids or []),
        created_at=now,
        updated_at=now,
    )


@pytest.fixture
def sm(tmp_path: Path) -> SkillMap:
    persistence = JSONPersistence(tmp_path / "sm.json", user_id="u1")
    return SkillMap(persistence.load(), persistence)


def test_insert_skill_registers_and_indexes_root(sm: SkillMap) -> None:
    skill = _make_skill("a")
    new_id, prereqs = sm.insert_skill(skill, candidate_prereq_ids=[])
    assert new_id == skill.id
    assert prereqs == []
    roots = sm.get_category_roots("coding")
    assert [r.id for r in roots] == [skill.id]


def test_insert_with_prereq_adds_edge_and_updates_roots(sm: SkillMap) -> None:
    a = _make_skill("a")
    sm.insert_skill(a, candidate_prereq_ids=[])
    b = _make_skill("b", episode_ids=["ep-b-1"])
    sm.insert_skill(b, candidate_prereq_ids=[a.id])

    children = sm.get_children(a.id)
    assert [c.id for c in children] == [b.id]

    # `b` is no longer a root of "coding" since it has an incoming edge.
    roots = sm.get_category_roots("coding")
    assert [r.id for r in roots] == [a.id]


def test_unknown_prereq_raises(sm: SkillMap) -> None:
    skill = _make_skill("x")
    with pytest.raises(UnknownSkillError):
        sm.insert_skill(skill, candidate_prereq_ids=["does-not-exist"])


def test_cycle_rejected(sm: SkillMap) -> None:
    a = _make_skill("a")
    b = _make_skill("b")
    sm.insert_skill(a, candidate_prereq_ids=[])
    sm.insert_skill(b, candidate_prereq_ids=[a.id])
    # Now try inserting c with prereq b, then manually force a cycle by
    # inserting a new skill with b as prereq AND making it a parent of a.
    # We exercise the cycle detector via a direct edge attempt.
    c = _make_skill("c")
    sm.insert_skill(c, candidate_prereq_ids=[b.id])

    # Attempt: insert a skill whose id collides with a's ancestor chain.
    # The easiest way to trigger DAGIntegrityError in v0 is to re-insert a
    # (already present) with c as a prereq. Since _edge_would_create_cycle
    # looks at existing edges, this would close the loop c -> a -> b -> c.
    looped = Skill(
        id=a.id,  # overwrite a; semantics: "add prereq c to a"
        name=a.name,
        status=a.status,
        category=a.category,
        triggering_context=a.triggering_context,
        correction_target=a.correction_target,
        covered_failures=a.covered_failures,
        episode_ids=a.episode_ids,
        created_at=a.created_at,
        updated_at=a.updated_at,
    )
    with pytest.raises(DAGIntegrityError):
        sm.insert_skill(looped, candidate_prereq_ids=[c.id])


def test_transitive_reduction_drops_redundant_prereqs(sm: SkillMap) -> None:
    a = _make_skill("a")
    b = _make_skill("b")
    c = _make_skill("c")
    sm.insert_skill(a, candidate_prereq_ids=[])
    sm.insert_skill(b, candidate_prereq_ids=[a.id])
    # Insert c with BOTH a and b as candidates. Since a reaches b, a is
    # redundant and must be dropped.
    _, validated = sm.insert_skill(c, candidate_prereq_ids=[a.id, b.id])
    assert validated == [b.id]


def test_merge_episode_promotes_to_confirmed(sm: SkillMap) -> None:
    s = _make_skill("a", episode_ids=["ep-1"])
    sm.insert_skill(s, candidate_prereq_ids=[])
    assert sm.get_skill(s.id).status == "tentative"
    sm.merge_episode_into_skill(s.id, "ep-2")
    assert sm.get_skill(s.id).status == "confirmed"
    assert sm.get_skill(s.id).episode_ids == ["ep-1", "ep-2"]


def test_record_application_increments_mastery_only_on_clean_success(sm: SkillMap) -> None:
    s = _make_skill("a")
    sm.insert_skill(s, candidate_prereq_ids=[])
    sm.record_application(s.id, "ep-x", "success_no_correction")
    assert sm.get_skill(s.id).mastery_count == 1
    sm.record_application(s.id, "ep-y", "success_with_correction")
    assert sm.get_skill(s.id).mastery_count == 1
    sm.record_application(s.id, "ep-z", "failure")
    assert sm.get_skill(s.id).mastery_count == 1


def test_persistence_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "sm.json"
    persistence = JSONPersistence(path, user_id="u1")
    sm = SkillMap(persistence.load(), persistence)

    a = _make_skill("a")
    b = _make_skill("b", episode_ids=["ep-b-1"])
    sm.insert_skill(a, [])
    sm.insert_skill(b, [a.id])

    # Reload from disk; verify identical structure.
    sm2 = SkillMap(JSONPersistence(path, user_id="u1").load(), persistence)
    assert {s.id for s in sm2.list_skills()} == {s.id for s in sm.list_skills()}
    assert [c.id for c in sm2.get_children(a.id)] == [b.id]


def test_flag_and_commit_split(sm: SkillMap) -> None:
    parent = _make_skill("p", episode_ids=["ep-1", "ep-2"])
    sm.insert_skill(parent, [])
    sm.flag_pending_split(parent.id, "ep-3")
    sm.flag_pending_split(parent.id, "ep-4")
    sm.flag_pending_split(parent.id, "ep-5")
    assert sm.pending_split_progress(parent.id) == 3
    assert sm.get_skill(parent.id).status == "pending_split"

    a = _make_skill("p_a")
    b = _make_skill("p_b")
    reassignment = {eid: "a" for eid in parent.episode_ids}
    a_id, b_id = sm.commit_split(parent.id, a, b, reassignment)
    assert sm.get_skill(parent.id).status == "deprecated"
    assert sm.get_skill(a_id).category == "coding"
    assert sm.get_skill(b_id).category == "coding"
