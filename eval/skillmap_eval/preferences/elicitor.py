"""PreferenceElicitor: LLM-A generates a preference profile.

Validates:
  - exactly n_preferences entries
  - each has non-empty expected_correction_trigger
  - diversity: at least 3 distinct categories
Retries up to retry_max times on validation failure.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from typing import Any

from skillmap.llm.client import LLMClient, LLMConfig, _extract_json
from pydantic import BaseModel

from skillmap_eval.preferences.prompts import PREFERENCE_ELICITATION_PROMPT
from skillmap_eval.types import EvalTask, Preference, PreferenceProfile


_CATEGORIES_BY_TASK_TYPE: dict[str, list[str]] = {
    "python_coding": [
        "communication_style",
        "diagnostic_approach",
        "code_style",
        "tool_usage",
        "detail_level",
    ],
    "math_competition": [
        "communication_style",
        "diagnostic_approach",
        "mathematical_approach",
        "problem_solving_style",
        "detail_level",
    ],
}
_CATEGORIES_DEFAULT = [
    "communication_style",
    "diagnostic_approach",
    "solution_approach",
    "presentation_style",
    "detail_level",
]

_PERSONA_BY_TASK_TYPE: dict[str, str] = {
    "python_coding": "software engineer with strong opinions about how AI coding assistants should behave",
    "math_competition": "math competition trainer with strong opinions about how AI tutors should approach competition math",
}
_PERSONA_DEFAULT = "domain expert with strong opinions about how AI assistants should behave"


def _profile_schema(categories: list[str]) -> str:
    cats = ", ".join(categories)
    return f"""{{\n  "preferences": [\n    {{\n      "id": "pref_01",\n      "description": "...",\n      "priority": 1,\n      "expected_correction_trigger": "...",\n      "category": "one of: {cats}"\n    }}\n  ]\n}}"""


class ElicitationError(RuntimeError):
    pass


class _PreferenceProfileResponse(BaseModel):
    preferences: list[Preference]


class PreferenceElicitor:
    def __init__(self, llm_a_model: str, region: str = "us-east-1", retry_max: int = 3) -> None:
        self.llm_a_model = llm_a_model
        self.retry_max = retry_max
        self._client = LLMClient(
            LLMConfig(
                provider="bedrock",
                model=llm_a_model,
                region=region,
                extra_inference={"temperature": 0.0},
            )
        )

    async def run(
        self,
        task_type: str = "python_coding",
        n_preferences: int = 10,
        task_examples: list[EvalTask] | None = None,
    ) -> PreferenceProfile:
        last_err: str | None = None
        for attempt in range(self.retry_max):
            try:
                raw = await self._call_llm(task_type, n_preferences, task_examples=task_examples)
                prefs = self._parse_preferences(raw)
                self._validate(prefs, n_preferences)
                return PreferenceProfile(
                    profile_id=f"profile_{task_type}_{uuid.uuid4().hex[:8]}",
                    generator_model=self.llm_a_model,
                    task_type=task_type,
                    preferences=prefs,
                    created_at=datetime.now(timezone.utc),
                )
            except (ValueError, json.JSONDecodeError) as e:
                last_err = f"attempt {attempt}: {e}"
                continue
        raise ElicitationError(f"preference elicitation failed after {self.retry_max} attempts: {last_err}")

    async def _call_llm(
        self,
        task_type: str,
        n_preferences: int,
        task_examples: list[EvalTask] | None = None,
    ) -> dict[str, Any]:
        categories = _CATEGORIES_BY_TASK_TYPE.get(task_type, _CATEGORIES_DEFAULT)
        persona = _PERSONA_BY_TASK_TYPE.get(task_type, _PERSONA_DEFAULT)
        prompt = PREFERENCE_ELICITATION_PROMPT.format(
            n_preferences=n_preferences,
            task_type=task_type,
            persona=persona,
            categories_list=", ".join(categories),
            schema=_profile_schema(categories),
            task_examples_section=self._format_task_examples(task_examples or []),
        )
        result = await self._client.call(
            messages=[{"role": "user", "content": prompt}],
            response_schema=_PreferenceProfileResponse,
        )
        if not isinstance(result, dict):
            raise ValueError(f"expected dict from LLMClient, got {type(result).__name__}")
        return result

    @staticmethod
    def _format_task_examples(task_examples: list[EvalTask]) -> str:
        if not task_examples:
            return "Task examples: none provided."

        lines = ["Task examples:"]
        for idx, task in enumerate(task_examples, start=1):
            problem = " ".join(task.problem_statement.split())
            if len(problem) > 700:
                problem = problem[:697] + "..."
            lines.extend(
                [
                    f"{idx}. [{task.difficulty}] task_id={task.task_id}",
                    f"   {problem}",
                ]
            )
        return "\n".join(lines)

    def _parse_preferences(self, raw: Any) -> list[Preference]:
        if isinstance(raw, dict):
            data = raw
        else:
            data = json.loads(_extract_json(raw))
        if isinstance(data, dict) and "preferences" in data:
            items = data["preferences"]
        elif isinstance(data, list):
            items = data
        else:
            raise ValueError(f"unexpected shape: {data!r}")
        return [Preference.model_validate(item) for item in items]

    def _validate(self, prefs: list[Preference], expected_n: int) -> None:
        if len(prefs) != expected_n:
            raise ValueError(f"expected {expected_n} preferences, got {len(prefs)}")
        for p in prefs:
            if not p.expected_correction_trigger.strip():
                raise ValueError(f"preference {p.id!r} has empty correction trigger")
        categories = {p.category for p in prefs}
        if len(categories) < 3:
            raise ValueError(f"need >=3 distinct categories, got {sorted(categories)}")
        priorities = [p.priority for p in prefs]
        if len(set(priorities)) != len(priorities):
            raise ValueError(f"priorities must be distinct; got {priorities}")
