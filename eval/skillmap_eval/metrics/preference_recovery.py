"""Metric 2: preference recovery (SkillMap condition only).

LLM-as-judge: for each ground-truth preference, ask whether any induced
skill captures it.
"""

from __future__ import annotations

from skillmap.llm.client import LLMClient, LLMConfig
from pydantic import BaseModel
from skillmap.types import Skill

from skillmap_eval.types import (
    ConditionName,
    PreferenceProfile,
    PreferenceRecoveryResult,
)


_JUDGE_PROMPT = """\
Preference: {preference_description}
Expected correction style: {preference_trigger}

Candidate induced skills:
{candidates}

Does any of these skills capture the preference? A skill captures a
preference if a person holding this preference would be satisfied that the
skill would prevent the assistant from violating this preference.

Output ONLY a JSON object:
{{"matched_skill_id": "skill_xxx" | null, "reasoning": "..."}}
"""


class _PreferenceRecoveryJudgeResponse(BaseModel):
    matched_skill_id: str | None
    reasoning: str


class PreferenceRecoveryJudge:
    def __init__(self, judge_model: str, region: str = "us-east-1") -> None:
        self._client = LLMClient(
            LLMConfig(
                provider="bedrock",
                model=judge_model,
                region=region,
                extra_inference={"temperature": 0.0},
            )
        )

    async def judge(
        self,
        profile: PreferenceProfile,
        induced_skills: list[Skill],
        condition_name: ConditionName = "skillmap",
    ) -> PreferenceRecoveryResult:
        recovered: list[str] = []
        reasoning: dict[str, str] = {}
        candidates_block = self._render_candidates(induced_skills)

        for pref in profile.preferences:
            prompt = _JUDGE_PROMPT.format(
                preference_description=pref.description,
                preference_trigger=pref.expected_correction_trigger,
                candidates=candidates_block or "(no induced skills)",
            )
            try:
                parsed = await self._client.call(
                    messages=[{"role": "user", "content": prompt}],
                    response_schema=_PreferenceRecoveryJudgeResponse,
                )
            except Exception:
                parsed = {"matched_skill_id": None, "reasoning": "parse_error"}
            matched = parsed.get("matched_skill_id")
            reasoning[pref.id] = str(parsed.get("reasoning", ""))[:500]
            if matched:
                recovered.append(pref.id)

        return PreferenceRecoveryResult(
            condition_name=condition_name,
            total_preferences=len(profile.preferences),
            recovered_preferences=recovered,
            recovery_rate=(len(recovered) / len(profile.preferences)) if profile.preferences else 0.0,
            judge_reasoning=reasoning,
        )

    @staticmethod
    def _render_candidates(skills: list[Skill]) -> str:
        lines: list[str] = []
        for s in skills:
            if s.status == "deprecated":
                continue
            lines.append(
                f"- {s.id}: {s.name}\n"
                f"    triggering_context: {s.triggering_context}\n"
                f"    correction_target: {s.correction_target}"
            )
        return "\n".join(lines)
