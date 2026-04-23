"""Phase 3 gate: induction pipeline integration.

These tests are SKELETONS - they require a working LLM backend (Phase 2)
and are marked xfail until the LLM client is wired up. They describe the
contract that finalize_task must meet.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.xfail(
    reason="Phase 3 gate requires LLM client (see skillmap/llm/client.py TODO)",
    strict=False,
)


@pytest.mark.asyncio
async def test_finalize_task_on_fixture_episode_inserts_skill() -> None:
    """Feed a hand-crafted multi-turn dialogue with one correction; verify
    a skill with non-empty fields appears in the map after finalize_task.

    Phase 3 gate in spec §6.
    """
    raise NotImplementedError("Phase 3 skeleton")


@pytest.mark.asyncio
async def test_proposer_rejects_over_specific_correction() -> None:
    """If covered_failures < 2 the SkillProposer must return None."""
    raise NotImplementedError("Phase 3 skeleton")


@pytest.mark.asyncio
async def test_matcher_merges_on_both_gates_passing() -> None:
    """Gate 1 + Gate 2 both pass -> merge_episode_into_skill, no new skill."""
    raise NotImplementedError("Phase 3 skeleton")


@pytest.mark.asyncio
async def test_matcher_flags_split_when_gate_2_fails() -> None:
    """Gate 1 pass + Gate 2 fail -> flag_pending_split; after
    split_disambiguation_threshold -> commit_split."""
    raise NotImplementedError("Phase 3 skeleton")
