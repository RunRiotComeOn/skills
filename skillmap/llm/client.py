"""LLM client abstraction. Default backend: Amazon Bedrock (Converse API).

Single entry point: `call_llm(messages, response_schema=None, system=None)`.
If a `response_schema` (Pydantic model class) is passed, the LLM's text is
parsed into that schema and returned as a dict; otherwise the raw text is
returned.

Components MUST NOT import provider SDKs directly.

Supported Bedrock model ids (us-east-1 unless noted) and pricing per 1M
tokens (input / output):
  Anthropic  us.anthropic.claude-haiku-4-5-20251001-v1:0    $1.00 / $5.00
             us.anthropic.claude-sonnet-4-6                 $3.00 / $15.00
             us.anthropic.claude-opus-4-6-v1                $5.00 / $25.00
  Moonshot   moonshotai.kimi-k2.5                           $0.60 / $3.00
             moonshotai.kimi-k2-thinking                    $0.60 / $2.50
  MiniMax    minimax.minimax-m2.5                           $0.30 / $1.20   (default)
  DeepSeek   deepseek.deepseek-v3-2                         $0.62 / $1.85
  Qwen       qwen.qwen3-32b-v1:0                            $0.20 / $0.78
             qwen.qwen3-235b-a22b-2507-v1:0  (us-east-2)    $0.11 / $0.45
             qwen.qwen3-coder-next
"""

from __future__ import annotations

import asyncio
import csv
import json
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Type

import requests
from pydantic import BaseModel


def _load_dotenv_once() -> None:
    """Populate os.environ from a repo-root .env file. No-op if absent or
    already loaded. Kept tiny (no python-dotenv dependency)."""
    if getattr(_load_dotenv_once, "_done", False):
        return
    _load_dotenv_once._done = True  # type: ignore[attr-defined]
    root = Path(__file__).resolve().parents[2]
    env_path = root / ".env"
    if not env_path.exists():
        return
    for raw in env_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


_load_dotenv_once()


@dataclass
class LLMConfig:
    provider: str = "bedrock"
    model: str = "minimax.minimax-m2.5"
    region: str = "us-east-1"
    max_tokens: int = 4096
    api_key_env: str = "BEDROCK_API_KEY"
    # Fallback CSV source. Header must include the column "API key".
    api_key_csv: str | None = "./bedrock-long-term-api-key.csv"
    # Networking.
    timeout_s: float = 120.0
    retry_max: int = 3
    retry_backoff_s: float = 2.0
    # Extra inference parameters forwarded into `inferenceConfig`.
    extra_inference: dict = field(default_factory=dict)


class LLMClient:
    """Thin provider-agnostic wrapper around Bedrock's Converse REST endpoint."""

    def __init__(self, config: LLMConfig | None = None) -> None:
        self.config = config or LLMConfig()
        self._api_key: str | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def call(
        self,
        messages: list[dict[str, str]],
        response_schema: Type[BaseModel] | None = None,
        system: str | None = None,
    ) -> str | dict:
        if self.config.provider != "bedrock":
            raise NotImplementedError(f"provider {self.config.provider!r} not supported in v0")

        region = _resolve_region(self.config.model, self.config.region)
        endpoint = (
            f"https://bedrock-runtime.{region}.amazonaws.com/"
            f"model/{self.config.model}/converse"
        )
        api_key = self._load_api_key()

        payload: dict[str, Any] = {
            "messages": [_to_bedrock_message(m) for m in messages],
            "inferenceConfig": {
                "maxTokens": self.config.max_tokens,
                **self.config.extra_inference,
            },
        }
        if system:
            payload["system"] = [{"text": system}]
        if response_schema is not None and _supports_structured_output(self.config.model):
            payload["outputConfig"] = {
                "textFormat": {
                    "type": "json_schema",
                    "structure": {
                        "jsonSchema": {
                            "schema": json.dumps(
                                _bedrock_json_schema(response_schema.model_json_schema())
                            ),
                            "name": response_schema.__name__,
                            "description": f"Structured output for {response_schema.__name__}",
                        }
                    },
                }
            }

        text = await asyncio.to_thread(
            _post_converse,
            endpoint,
            api_key,
            payload,
            self.config.timeout_s,
            self.config.retry_max,
            self.config.retry_backoff_s,
        )

        if response_schema is not None:
            extracted = _extract_json(text)
            return response_schema.model_validate_json(extracted).model_dump()
        return text

    # ------------------------------------------------------------------
    # Auth helpers
    # ------------------------------------------------------------------

    def _load_api_key(self) -> str:
        if self._api_key:
            return self._api_key

        env_key = os.environ.get(self.config.api_key_env) if self.config.api_key_env else None
        if env_key:
            self._api_key = env_key
            return env_key

        if self.config.api_key_csv:
            csv_path = Path(self.config.api_key_csv)
            if not csv_path.is_absolute():
                # Resolve relative to project root (2 levels up from this file).
                root = Path(__file__).resolve().parents[2]
                csv_path = root / self.config.api_key_csv
            if csv_path.exists():
                with csv_path.open(newline="") as f:
                    reader = csv.DictReader(f)
                    row = next(reader, None)
                    if row and "API key" in row:
                        self._api_key = row["API key"]
                        return self._api_key

        raise RuntimeError(
            f"Bedrock API key not found. Set ${self.config.api_key_env} or "
            f"place a CSV at {self.config.api_key_csv!r} with an 'API key' column."
        )


# ---------------------------------------------------------------------------
# Module-level convenience
# ---------------------------------------------------------------------------

_default_client: LLMClient | None = None


def _get_default_client() -> LLMClient:
    global _default_client
    if _default_client is None:
        _default_client = LLMClient()
    return _default_client


def configure_default_client(config: LLMConfig) -> None:
    """Override the process-wide default client. Called once at startup."""
    global _default_client
    _default_client = LLMClient(config)


async def call_llm(
    messages: list[dict[str, str]],
    response_schema: Type[BaseModel] | None = None,
    system: str | None = None,
) -> str | dict:
    return await _get_default_client().call(
        messages, response_schema=response_schema, system=system
    )


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _resolve_region(model_id: str, default_region: str) -> str:
    # Qwen3 235B is only exposed in us-east-2 on this account.
    if model_id.startswith("qwen.qwen3-235b") and default_region == "us-east-1":
        return "us-east-2"
    return default_region


def _supports_structured_output(model_id: str) -> bool:
    supported_markers = (
        "claude-haiku-4-5",
        "claude-sonnet-4-5",
        "claude-opus-4-5",
        "claude-opus-4-6",
    )
    return any(marker in model_id for marker in supported_markers)


def _bedrock_json_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Normalize a Pydantic JSON schema to satisfy Bedrock structured-output requirements."""
    if isinstance(schema, dict):
        out: dict[str, Any] = {}
        for key, value in schema.items():
            out[key] = _bedrock_json_schema(value) if isinstance(value, (dict, list)) else value

        if out.get("type") == "object":
            out.setdefault("additionalProperties", False)
        return out

    if isinstance(schema, list):
        return [_bedrock_json_schema(item) if isinstance(item, (dict, list)) else item for item in schema]

    return schema


def _to_bedrock_message(msg: dict[str, Any]) -> dict[str, Any]:
    """Normalize a `{role, content: str}` dict into Bedrock Converse shape.

    Bedrock expects `content` to be a list of content-block dicts. If the
    caller already provides the list form, pass through unchanged.
    """
    role = msg["role"]
    if role not in ("user", "assistant"):
        # Bedrock Converse only accepts user/assistant in messages; system
        # goes at the top level.
        raise ValueError(f"unexpected role in messages: {role!r}")
    content = msg["content"]
    if isinstance(content, str):
        return {"role": role, "content": [{"text": content}]}
    return {"role": role, "content": content}


def _post_converse(
    endpoint: str,
    api_key: str,
    payload: dict[str, Any],
    timeout_s: float,
    retry_max: int,
    retry_backoff_s: float,
) -> str:
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    last_error: Exception | None = None
    for attempt in range(retry_max):
        try:
            resp = requests.post(endpoint, headers=headers, json=payload, timeout=timeout_s)
            if resp.status_code != 200:
                raise RuntimeError(f"Bedrock {resp.status_code}: {resp.text}")
            result = resp.json()
            break
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
            last_error = e
            if attempt == retry_max - 1:
                raise
            time.sleep(retry_backoff_s * (attempt + 1))
    else:
        raise RuntimeError(f"Bedrock request failed after retries: {last_error}")
    try:
        content_blocks = result["output"]["message"]["content"]
        if not isinstance(content_blocks, list):
            raise TypeError("content is not a list")
        text_parts = [
            block["text"]
            for block in content_blocks
            if isinstance(block, dict) and isinstance(block.get("text"), str)
        ]
        if text_parts:
            return "".join(text_parts)
        reasoning_parts = []
        for block in content_blocks:
            if not isinstance(block, dict):
                continue
            reasoning = block.get("reasoningContent")
            if not isinstance(reasoning, dict):
                continue
            reasoning_text = reasoning.get("reasoningText")
            if not isinstance(reasoning_text, dict):
                continue
            text = reasoning_text.get("text")
            if isinstance(text, str):
                reasoning_parts.append(text)
        if reasoning_parts:
            return "\n".join(reasoning_parts)
        raise KeyError("no text or reasoningText block found in content")
    except (KeyError, IndexError, TypeError) as e:
        raise RuntimeError(f"unexpected Bedrock response shape: {result!r}") from e


_JSON_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)


def _clean_json_text(text: str) -> str:
    """Remove common markdown wrapping and surrounding whitespace."""
    return _JSON_FENCE_RE.sub("", text).strip()


def _extract_json(text: str) -> str:
    """Best-effort JSON extraction from model text.

    Steps:
      1. Strip markdown fences like ```json ... ```
      2. Try parsing the cleaned text directly
      3. If that fails, extract the first `{...}` or `[...]` span and retry
    """
    stripped = _clean_json_text(text)
    try:
        json.loads(stripped)
        return stripped
    except json.JSONDecodeError:
        pass

    # Fallback: grab the first object span, then array span.
    for open_c, close_c in (("{", "}"), ("[", "]")):
        start = stripped.find(open_c)
        end = stripped.rfind(close_c)
        if start == -1 or end == -1 or end <= start:
            continue
        candidate = stripped[start : end + 1].strip()
        try:
            json.loads(candidate)
            return candidate
        except json.JSONDecodeError:
            continue
    raise ValueError(f"could not extract JSON from model output: {text!r}")
