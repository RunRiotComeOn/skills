"""PostHocAnalyzer: after an episode finalizes, record application outcomes.

For every skill in `episode.retrieved_skills_at_start`, decide whether the
assistant actually applied it (success_no_correction / success_with_correction
/ failure) via APPLICATION_JUDGMENT_PROMPT, then call
`skill_map.record_application`.
"""

from __future__ import annotations

import json
from typing import Any

from skillmap.llm.client import LLMClient, LLMConfig, _extract_json
from skillmap.llm.prompts import APPLICATION_JUDGMENT_PROMPT
from skillmap.storage.skill_map import SkillMap
from skillmap.types import ApplicationOutcome, Episode
from pydantic import BaseModel


class _ApplicationJudgmentResponse(BaseModel):
    outcome: str


class PostHocAnalyzer:
    def __init__(
        self,
        llm_model: str = "us.anthropic.claude-haiku-4-5-20251001-v1:0",
        region: str = "us-east-1",
        retry_max: int = 3,
    ) -> None:
        self.retry_max = retry_max
        self._client = LLMClient(
            LLMConfig(
                provider="bedrock",
                model=llm_model,
                region=region,
                max_tokens=256,
                extra_inference={"temperature": 0.0},
            )
        )

    async def run(self, episode: Episode, skill_map: SkillMap) -> None:
        if not episode.retrieved_skills_at_start:
            return

        trajectory_text = _render_trajectory(episode)
        corrections_text = _render_corrections(episode)

        for skill_id in episode.retrieved_skills_at_start:
            try:
                skill = skill_map.get_skill(skill_id)
            except Exception:
                # Skill may have been deprecated/split between retrieval and
                # finalization; skip silently.
                continue

            outcome = await self._judge(
                skill_name=skill.name,
                triggering_context=skill.triggering_context,
                correction_target=skill.correction_target,
                trajectory_text=trajectory_text,
                corrections_text=corrections_text,
            )
            skill_map.record_application(skill_id, episode.id, outcome)

    async def _judge(
        self,
        skill_name: str,
        triggering_context: str,
        correction_target: str,
        trajectory_text: str,
        corrections_text: str,
    ) -> ApplicationOutcome:
        parsed = await self._call_json(
            APPLICATION_JUDGMENT_PROMPT.format(
                skill_name=skill_name,
                skill_triggering_context=triggering_context,
                skill_correction_target=correction_target,
                trajectory=trajectory_text,
                corrections=corrections_text,
            ),
            _ApplicationJudgmentResponse,
        )
        outcome = parsed.get("outcome", "success_no_correction")
        if outcome not in ("success_no_correction", "success_with_correction", "failure"):
            outcome = "success_no_correction"
        return outcome  # type: ignore[return-value]

    async def _call_json(self, prompt: str, schema: type[BaseModel]) -> dict[str, Any]:
        strict_suffix = (
            "\n\nCRITICAL: Return ONLY valid minified JSON. "
            "No prose, no markdown fence, no explanation."
        )
        last_error: Exception | None = None
        for attempt in range(self.retry_max):
            try:
                raw = await self._client.call(
                    messages=[
                        {
                            "role": "user",
                            "content": prompt if attempt == 0 else prompt + strict_suffix,
                        }
                    ],
                    response_schema=schema,
                )
                return _parse_json_obj(raw)
            except (json.JSONDecodeError, ValueError, RuntimeError) as e:
                last_error = e
                continue
        raise RuntimeError(f"post hoc analyzer failed to return valid JSON after retries: {last_error}")


def _render_trajectory(episode: Episode) -> str:
    return "\n".join(
        f"[{i}] {t.role}: {t.content}" for i, t in enumerate(episode.trajectory)
    )


def _render_corrections(episode: Episode) -> str:
    if not episode.correction_points:
        return "(none)"
    return "\n".join(
        f"- turn {cp.turn_index} ({cp.correction_type}): {cp.correction_content}"
        for cp in episode.correction_points
    )


def _parse_json_obj(raw: Any) -> dict:
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        return json.loads(_extract_json(raw))
    return raw or {}
