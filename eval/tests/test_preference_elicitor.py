"""Phase 1 gate: preference elicitation. Live-LLM tests are xfail by default."""

from __future__ import annotations

from pathlib import Path

import pytest

from skillmap_eval.preferences import PreferenceElicitor
from skillmap_eval.types import EvalTask, PreferenceProfile


class _FakeClient:
    def __init__(self) -> None:
        self.messages = None

    async def call(self, messages: list[dict[str, str]]) -> str:
        self.messages = messages
        return '{"preferences":[]}'


def test_profile_schema_accepts_valid_fixture() -> None:
    """Non-LLM sanity: a hand-crafted profile JSON parses."""
    fixture = Path(__file__).parent / "fixtures" / "sample_preference_profile.json"
    profile = PreferenceProfile.model_validate_json(fixture.read_text())
    assert len(profile.preferences) == 10
    assert {p.category for p in profile.preferences} >= {
        "communication_style",
        "code_style",
        "diagnostic_approach",
    }
    for p in profile.preferences:
        assert p.expected_correction_trigger.strip()


@pytest.mark.asyncio
async def test_elicitor_prompt_includes_livecodebench_examples() -> None:
    elicitor = PreferenceElicitor(llm_a_model="fake-model")
    fake_client = _FakeClient()
    elicitor._client = fake_client

    tasks = [
        EvalTask(
            task_id="lcb_easy_001",
            source="livecodebench",
            problem_statement="Read integers from stdin and print their running maximum.",
            difficulty="easy",
        ),
        EvalTask(
            task_id="lcb_medium_002",
            source="livecodebench",
            problem_statement="Given a tree, answer path xor queries efficiently.",
            difficulty="medium",
        ),
        EvalTask(
            task_id="lcb_hard_003",
            source="livecodebench",
            problem_statement="Maintain dynamic connectivity under edge deletions.",
            difficulty="hard",
        ),
    ]

    await elicitor._call_llm(
        task_type="python_coding",
        n_preferences=10,
        task_examples=tasks,
    )

    assert fake_client.messages is not None
    prompt = fake_client.messages[0]["content"]
    assert "Task examples:" in prompt
    assert "lcb_easy_001" in prompt
    assert "lcb_medium_002" in prompt
    assert "lcb_hard_003" in prompt
    assert "running maximum" in prompt


@pytest.mark.xfail(reason="Phase 1 gate requires Bedrock API key + LLM-A call", strict=False)
@pytest.mark.asyncio
async def test_elicitor_returns_valid_profile() -> None:
    """Contract: PreferenceElicitor.run() returns a PreferenceProfile with
    exactly n_preferences entries, each with a non-empty correction trigger,
    and at least 3 distinct categories.

    Manual human gate: read the output, agree that >=8/N preferences are
    opinionated and non-default.
    """
    raise NotImplementedError("requires live LLM-A")
