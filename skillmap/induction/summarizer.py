"""Stage A: extract CorrectionSummary records from a completed task trajectory."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel

from skillmap.llm.client import LLMClient, LLMConfig
from skillmap.llm.prompts import CORRECTION_SUMMARIZER_PROMPT
from skillmap.types import CorrectionSummary


class _SummaryItem(BaseModel):
    triggering_situation: str
    what_was_wrong: str
    what_user_wanted: str
    correction_quote: str


class _SummaryResponse(BaseModel):
    summaries: list[_SummaryItem]


class CorrectionSummarizer:
    def __init__(
        self,
        llm_model: str = "us.anthropic.claude-haiku-4-5-20251001-v1:0",
        region: str = "us-east-1",
        retry_max: int = 3,
    ) -> None:
        self._client = LLMClient(
            LLMConfig(
                provider="bedrock",
                model=llm_model,
                region=region,
                max_tokens=1024,
                extra_inference={"temperature": 0.0},
            )
        )
        self.retry_max = retry_max

    async def run(
        self,
        trajectory: list[dict[str, str]],
        task_id: str,
    ) -> list[CorrectionSummary]:
        """Extract 0-5 CorrectionSummary records from a task trajectory."""
        conversation = _render_conversation(trajectory)
        prompt = CORRECTION_SUMMARIZER_PROMPT.format(conversation=conversation)

        last_exc: Exception | None = None
        for _ in range(self.retry_max):
            try:
                raw: Any = await self._client.call(
                    messages=[{"role": "user", "content": prompt}],
                    response_schema=_SummaryResponse,
                )
                items = raw.get("summaries", [])
                now = datetime.now(timezone.utc)
                return [
                    CorrectionSummary(
                        id=str(uuid.uuid4()),
                        task_id=task_id,
                        created_at=now,
                        triggering_situation=item["triggering_situation"],
                        what_was_wrong=item["what_was_wrong"],
                        what_user_wanted=item["what_user_wanted"],
                        correction_quote=item["correction_quote"],
                    )
                    for item in items
                ]
            except Exception as exc:
                last_exc = exc
        raise RuntimeError(f"summarizer failed after retries: {last_exc}") from last_exc


def _render_conversation(trajectory: list[dict[str, str]]) -> str:
    lines = []
    for i, turn in enumerate(trajectory):
        role = turn.get("role", "unknown").upper()
        content = turn.get("content", "")
        lines.append(f"[Turn {i}] {role}:\n{content}")
    return "\n\n".join(lines)
