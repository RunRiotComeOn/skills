"""Mem0-style declarative memory baseline.

After each task, LLM-B summarizes the interaction into a short list of
"facts about what this user seems to prefer". Facts accumulate across
tasks. On each new turn, all stored facts are prepended to the system
prompt.

v1 simplification: inject ALL stored facts, not an embedding-filtered top-k.
For a 30-task stream the fact list stays short enough that full injection
is an acceptable approximation of a Mem0-style retriever. Flag the
simplification in comments so a future patch can swap in real embedding
retrieval without reshaping the condition API.
"""

from __future__ import annotations

import json
from pathlib import Path

from skillmap.llm.client import LLMClient, LLMConfig
from pydantic import BaseModel

from skillmap_eval.conditions.base import (
    SYSTEM_PROMPT_BASE,
    format_task_hint,
    render_conversation_for_llm_b,
)
from skillmap_eval.types import EvalTask, SimulatedTurn, TaskInteraction


_FACT_EXTRACTION_PROMPT = """\
You have just completed a coding task with a user. Based on how the user
corrected you (if at all), extract 0-3 SHORT, ACTIONABLE facts about the
user's preferences. Keep each fact to one sentence. Do not invent; only
state what was evidenced. If there is nothing new, return an empty list.

Conversation:
{conversation}

Existing known facts (do not duplicate):
{existing_facts}

Output ONLY a JSON object:
{{"facts": ["The user prefers ...", "The user dislikes ..."]}}
"""


class DeclarativeMemoryCondition:
    name = "declarative_memory"

    def __init__(
        self,
        llm_b_model: str,
        region: str = "us-east-1",
        temperature: float = 0.7,
        storage_root: str | None = None,
        resume_existing: bool = False,
    ) -> None:
        self._client = LLMClient(
            LLMConfig(
                provider="bedrock",
                model=llm_b_model,
                region=region,
                extra_inference={"temperature": temperature},
            )
        )
        # Separate client for fact extraction; deterministic.
        self._summarizer = LLMClient(
            LLMConfig(
                provider="bedrock",
                model=llm_b_model,
                region=region,
                extra_inference={"temperature": 0.0},
            )
        )
        self._facts: list[str] = []
        # Optional on-disk persistence so multi-day SLURM runs that get
        # interrupted resume from the same fact list rather than starting
        # over. Mirrors SkillMapCondition.resume_existing.
        self._storage_root = Path(storage_root) if storage_root else None
        self._resume_existing = resume_existing
        self._facts_path: Path | None = None

    async def setup(self, profile_id: str, run_id: str) -> None:
        if self._storage_root is not None:
            run_dir = self._storage_root / profile_id / run_id
            run_dir.mkdir(parents=True, exist_ok=True)
            self._facts_path = run_dir / "declarative_memory.json"
            if self._resume_existing and self._facts_path.exists():
                try:
                    self._facts = json.loads(self._facts_path.read_text(encoding="utf-8"))
                    return
                except (json.JSONDecodeError, OSError):
                    pass
        self._facts = []

    async def handle_user_message(
        self,
        task: EvalTask,
        conversation_so_far: list[SimulatedTurn],
        user_message: str,
    ) -> tuple[str, list[str]]:
        system = self._render_system_prompt(format_task_hint(task))
        messages = render_conversation_for_llm_b(conversation_so_far, user_message)
        text = await self._client.call(messages=messages, system=system)
        return (text if isinstance(text, str) else str(text)), []

    async def finalize_task(
        self,
        task: EvalTask,
        interaction: TaskInteraction,
    ) -> None:
        conversation = "\n".join(
            f"{t.role.upper()}: {t.content}" for t in interaction.turns
        )
        existing = "\n".join(f"- {f}" for f in self._facts) or "(none)"
        prompt = _FACT_EXTRACTION_PROMPT.format(
            conversation=conversation, existing_facts=existing
        )
        raw = await self._summarizer.call(
            messages=[{"role": "user", "content": prompt}],
            response_schema=_FactsResponse,
        )
        if not isinstance(raw, dict):
            return
        try:
            new_facts = raw.get("facts", [])
        except Exception:
            return
        if not isinstance(new_facts, list):
            return
        for fact in new_facts:
            if isinstance(fact, str) and fact.strip() and fact not in self._facts:
                self._facts.append(fact.strip())
        self._persist_facts()

    async def teardown(self) -> None:
        return None

    def _persist_facts(self) -> None:
        if self._facts_path is None:
            return
        try:
            self._facts_path.write_text(
                json.dumps(self._facts, indent=2), encoding="utf-8"
            )
        except OSError:
            pass

    # ------------------------------------------------------------------

    def _render_system_prompt(self, task_hint: str = "") -> str:
        parts = [SYSTEM_PROMPT_BASE]
        if task_hint:
            parts.append(task_hint)
        if self._facts:
            facts_block = "\n".join(f"- {f}" for f in self._facts)
            parts.append(
                f"Known preferences for this user (follow them):\n{facts_block}"
            )
        return "\n\n".join(parts)


class _FactsResponse(BaseModel):
    facts: list[str]
