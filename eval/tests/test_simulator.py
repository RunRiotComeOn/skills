"""Phase 3 gate: user simulator. Live-LLM tests are xfail by default."""

from __future__ import annotations

from pathlib import Path

import pytest

from skillmap_eval.simulator.correction_generator import (
    count_pref_violations,
    render_conversation,
    render_profile,
)
from skillmap_eval.types import PreferenceProfile, SimulatedTurn


def test_render_profile_orders_by_priority() -> None:
    fixture = Path(__file__).parent / "fixtures" / "sample_preference_profile.json"
    profile = PreferenceProfile.model_validate_json(fixture.read_text())
    rendered = render_profile(profile)
    # pref_01 has priority 1, pref_10 has priority 10; pref_01 must appear first.
    assert rendered.index("[pref_01]") < rendered.index("[pref_10]")


def test_render_conversation_handles_empty() -> None:
    assert render_conversation([]) == "(no messages yet)"


def test_count_pref_violations_tallies_user_turns_only() -> None:
    turns = [
        SimulatedTurn(role="user", content="a", triggered_preferences=["pref_01"]),
        SimulatedTurn(role="assistant", content="x", violated_preferences=["pref_01"]),
        SimulatedTurn(role="user", content="b", triggered_preferences=["pref_01", "pref_02"]),
    ]
    assert count_pref_violations(turns) == {"pref_01": 2, "pref_02": 1}


@pytest.mark.xfail(reason="Phase 3 gate requires Bedrock API key + LLM-A call", strict=False)
@pytest.mark.asyncio
async def test_simulator_decisions_match_expectations_on_scripted_responses() -> None:
    """Phase 3 gate: feed 5 scripted assistant responses (3 violating, 2
    compliant) and verify simulator's decisions match expected
    (correct/correct/correct/accept/accept)."""
    raise NotImplementedError("requires live LLM-A")


@pytest.mark.xfail(reason="requires live LLM-A", strict=False)
@pytest.mark.asyncio
async def test_simulator_give_up_requires_turn_gate_and_repeat_gate() -> None:
    """give_up only fires when turn count > 5 AND a pref has repeated >= threshold."""
    raise NotImplementedError("requires live LLM-A")
