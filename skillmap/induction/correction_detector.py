"""CorrectionDetector: identify and classify correction points in an episode.

LLM-based. Prompt: CORRECTION_DETECTION_PROMPT in skillmap/llm/prompts.py.
"""

from __future__ import annotations

import json
from typing import Any

from skillmap.llm.client import LLMClient, LLMConfig, _extract_json
from skillmap.llm.prompts import CORRECTION_DETECTION_PROMPT
from skillmap.types import CorrectionPoint, Episode
from pydantic import BaseModel


class _CorrectionPointPayload(BaseModel):
    turn_index: int
    correction_content: str
    correction_type: str


class _CorrectionDetectorResponse(BaseModel):
    corrections: list[_CorrectionPointPayload]


class CorrectionDetector:
    def __init__(
        self,
        llm_model: str = "minimax.minimax-m2.5",
        region: str = "us-east-1",
        retry_max: int = 3,
    ) -> None:
        self.retry_max = retry_max
        self._client = LLMClient(
            LLMConfig(
                provider="bedrock",
                model=llm_model,
                region=region,
                max_tokens=1536,
                extra_inference={"temperature": 0.0},
            )
        )

    async def run(self, episode: Episode) -> list[CorrectionPoint]:
        trajectory_text = _render_trajectory(episode)
        prompt = CORRECTION_DETECTION_PROMPT.format(trajectory=trajectory_text)
        strict_suffix = (
            "\n\nCRITICAL: Return ONLY a valid JSON array. "
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
                    response_schema=_CorrectionDetectorResponse,
                )
                return _parse_corrections(raw)
            except (json.JSONDecodeError, ValueError, RuntimeError) as e:
                last_error = e
                continue
        raise RuntimeError(
            f"correction detector failed to return valid JSON after retries: {last_error}"
        )


def _render_trajectory(episode: Episode) -> str:
    lines = []
    for i, t in enumerate(episode.trajectory):
        lines.append(f"[{i}] {t.role}: {t.content}")
    return "\n".join(lines)


def _parse_corrections(raw: Any) -> list[CorrectionPoint]:
    """Parse the LLM's JSON array into CorrectionPoint objects."""
    if isinstance(raw, dict):
        data = raw
    elif isinstance(raw, str):
        try:
            data = json.loads(_extract_json(raw))
        except ValueError:
            data = _recover_truncated_corrections(raw)
    else:
        data = raw
    if data is None:
        return []
    items = _normalize_correction_items(data)
    return [CorrectionPoint.model_validate(item) for item in items]


def _normalize_correction_items(data: Any) -> list[dict[str, Any]]:
    """Accept a few common JSON shapes and normalize to a list of objects."""
    if isinstance(data, list):
        return [item for item in data if isinstance(item, dict)]

    if isinstance(data, dict):
        # Common wrapper: {"corrections": [...]}
        corrections = data.get("corrections")
        if isinstance(corrections, list):
            return [item for item in corrections if isinstance(item, dict)]

        # Single-object fallback: {"turn_index": ..., ...}
        if {
            "turn_index",
            "correction_content",
            "correction_type",
        }.issubset(data.keys()):
            return [data]

    raise ValueError(f"unexpected correction detector JSON shape: {data!r}")


def _recover_truncated_corrections(raw: str) -> list[dict[str, Any]]:
    """Recover complete leading objects from a truncated JSON array response."""
    from skillmap.llm.client import _clean_json_text

    text = _clean_json_text(raw)
    start = text.find("[")
    if start == -1:
        raise ValueError(f"could not recover correction list from output: {raw!r}")

    decoder = json.JSONDecoder()
    idx = start + 1
    items: list[dict[str, Any]] = []
    length = len(text)

    while idx < length:
        while idx < length and text[idx] in " \r\n\t,":
            idx += 1
        if idx >= length or text[idx] == "]":
            break
        try:
            value, next_idx = decoder.raw_decode(text, idx)
        except json.JSONDecodeError:
            break
        if isinstance(value, dict):
            items.append(value)
        idx = next_idx

    if items:
        return items
    raise ValueError(f"could not recover correction list from output: {raw!r}")
