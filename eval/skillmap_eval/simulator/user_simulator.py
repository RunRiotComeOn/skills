"""UserSimulator: LLM-A plays the user, driven by a ground-truth preference profile.

Temperature is intentionally low (not zero) so corrections vary in phrasing
while remaining consistent on *which* preferences to flag.

Judgment scope: LLM-A judges ONLY preference adherence - never whether the
code passes tests.
"""

from __future__ import annotations

import json
import random
from typing import Any

from skillmap.llm.client import LLMClient, LLMConfig, _extract_json
from pydantic import BaseModel

from skillmap_eval.simulator.correction_generator import (
    count_pref_violations,
    render_conversation,
    render_profile,
    sample_correction_style_hint,
)
from skillmap_eval.simulator.prompts import (
    USER_INITIATE_PROMPT,
    USER_RESPONSE_PROMPT,
)
from skillmap_eval.types import (
    EvalTask,
    Preference,
    PreferenceProfile,
    SimulatedTurn,
    SimulatorAction,
)


class _UserSimulatorResponse(BaseModel):
    violated_prefs: list[str]
    decision: SimulatorAction
    response: str


class UserSimulator:
    def __init__(
        self,
        llm_a_model: str,
        profile: PreferenceProfile,
        region: str = "us-east-1",
        max_turns_per_task: int = 8,
        give_up_threshold_repeats: int = 3,
        retry_max: int = 2,
        simulator_temperature: float = 0.4,
        seed: int = 42,
    ) -> None:
        self.profile = profile
        self.max_turns_per_task = max_turns_per_task
        self.give_up_threshold_repeats = give_up_threshold_repeats
        self.retry_max = retry_max
        self._rng = random.Random(seed)
        self._client = LLMClient(
            LLMConfig(
                provider="bedrock",
                model=llm_a_model,
                region=region,
                max_tokens=512,
                extra_inference={"temperature": simulator_temperature},
            )
        )
        self._pref_by_id: dict[str, Preference] = {p.id: p for p in profile.preferences}

    async def initiate_task(self, task: EvalTask) -> str:
        prompt = USER_INITIATE_PROMPT.format(task_problem=task.problem_statement)
        result = await self._client.call(messages=[{"role": "user", "content": prompt}])
        if not isinstance(result, str):
            raise RuntimeError("expected string response from LLMClient")
        return result.strip()

    async def respond_to_assistant(
        self,
        task: EvalTask,
        conversation_so_far: list[SimulatedTurn],
        latest_assistant_message: str,
    ) -> tuple[str, list[str], SimulatorAction]:
        prompt = USER_RESPONSE_PROMPT.format(
            preference_profile_rendered=render_profile(self.profile),
            task_problem=task.problem_statement,
            conversation_rendered=render_conversation(conversation_so_far),
            latest_assistant_message=latest_assistant_message,
            give_up_threshold=self.give_up_threshold_repeats,
            correction_style_hint=sample_correction_style_hint(self._rng),
        )
        parsed = await self._call_and_parse_json(prompt)

        violated = [pid for pid in parsed.get("violated_prefs", []) if pid in self._pref_by_id]
        decision: SimulatorAction = parsed.get("decision", "accept")  # type: ignore[assignment]
        response = parsed.get("response", "")

        # Force give_up when the same pref has already been corrected
        # give_up_threshold_repeats times — the user is out of patience.
        forced = self._must_give_up(conversation_so_far, violated)
        if forced:
            decision = "give_up"
            if parsed.get("decision") != "give_up":
                response = (
                    "We've gone in circles on the same point too many times. "
                    "Let's drop it."
                )
        elif decision == "give_up" and not self._can_give_up(conversation_so_far, violated):
            decision = "correct" if violated else "accept"

        return response, violated, decision

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    async def _call_and_parse_json(self, prompt: str) -> dict[str, Any]:
        last_error: Exception | None = None
        strict_suffix = (
            "\n\nCRITICAL: Return ONLY valid minified JSON. "
            'No markdown fence, no prose, no backticks. '
            'The JSON must be an object with keys "violated_prefs", "decision", and "response".'
        )
        for attempt in range(self.retry_max):
            raw = await self._client.call(
                messages=[
                    {
                        "role": "user",
                        "content": prompt if attempt == 0 else prompt + strict_suffix,
                    }
                ],
                response_schema=_UserSimulatorResponse,
            )
            try:
                parsed = self._parse(raw)
                self._validate_response_shape(parsed)
                return parsed
            except (json.JSONDecodeError, ValueError) as e:
                last_error = e
                continue
        raise RuntimeError(f"user simulator failed to return valid JSON after retries: {last_error}")

    def _parse(self, raw: str | dict[str, Any]) -> dict[str, Any]:
        if isinstance(raw, dict):
            data = raw
        else:
            data = json.loads(_extract_json(raw))
        if not isinstance(data, dict):
            raise ValueError(f"user simulator expected JSON object, got {type(data).__name__}")
        return data

    @staticmethod
    def _validate_response_shape(parsed: dict[str, Any]) -> None:
        if "violated_prefs" not in parsed or "decision" not in parsed or "response" not in parsed:
            raise ValueError(f"user simulator missing required keys in response: {parsed!r}")
        violated = parsed["violated_prefs"]
        if not isinstance(violated, list) or not all(isinstance(item, str) for item in violated):
            raise ValueError(f"user simulator expected violated_prefs to be list[str], got {violated!r}")
        if parsed["decision"] not in ("correct", "accept", "give_up"):
            raise ValueError(f"user simulator returned invalid decision: {parsed['decision']!r}")
        if not isinstance(parsed["response"], str):
            raise ValueError(f"user simulator expected response to be str, got {type(parsed['response']).__name__}")

    def _can_give_up(
        self, conversation_so_far: list[SimulatedTurn], violated: list[str]
    ) -> bool:
        if len(conversation_so_far) <= 5:
            return False
        tally = count_pref_violations(conversation_so_far)
        return any(tally.get(pid, 0) + 1 >= self.give_up_threshold_repeats for pid in violated)

    def _must_give_up(
        self, conversation_so_far: list[SimulatedTurn], violated: list[str]
    ) -> bool:
        """Override the LLM and force give_up: the same pref has already been
        corrected give_up_threshold_repeats times and is being violated again."""
        if not violated:
            return False
        tally = count_pref_violations(conversation_so_far)
        return any(
            tally.get(pid, 0) >= self.give_up_threshold_repeats
            for pid in violated
        )
