"""Shared Condition protocol. All three implementations share LLM-B and the
outer loop; the ONLY difference is what gets injected into context.
"""

from __future__ import annotations

from typing import Protocol

from skillmap_eval.types import EvalTask, SimulatedTurn, TaskInteraction


class Condition(Protocol):
    name: str

    async def setup(self, profile_id: str, run_id: str) -> None: ...

    async def handle_user_message(
        self,
        task: EvalTask,
        conversation_so_far: list[SimulatedTurn],
        user_message: str,
    ) -> tuple[str, list[str]]:
        """Return (assistant_response, retrieved_skill_ids).

        retrieved_skill_ids is empty for non-skillmap conditions.
        """
        ...

    async def finalize_task(
        self,
        task: EvalTask,
        interaction: TaskInteraction,
    ) -> None: ...

    async def teardown(self) -> None: ...


SYSTEM_PROMPT_BASE = (
    "You are an expert Python coding assistant. Given a user task, respond "
    "with a working solution and a short explanation. When the user "
    "corrects you, adjust and continue without apologizing excessively."
)


def format_task_hint(task: EvalTask) -> str:
    """Render an I/O-format hint that grounds the assistant on whether to
    write a `class Solution` (functional / LeetCode-style) or a
    stdin-reading script (competitive-programming style).

    The hint is derived from `task.test_cases[*].testtype`. It is empty when
    the task has no test cases, mixes both styles, or carries an unknown
    testtype — in those cases we don't constrain format and let the model
    infer from the problem statement. Returns "" for the empty case so
    callers can safely concatenate without a separator check.
    """
    if not task.test_cases:
        return ""

    types = {tc.get("testtype") for tc in task.test_cases if isinstance(tc, dict)}
    has_func = "functional" in types
    has_stdin = "stdin" in types

    if has_func and not has_stdin:
        starter = (task.reference_solution or "").strip()
        starter_block = (
            f"\nStarter code (preserve the class and method signature):\n"
            f"```python\n{starter}\n```"
            if starter else ""
        )
        return (
            "I/O FORMAT (do not deviate): this task uses a LeetCode-style "
            "functional API. Define a `class Solution` with the method "
            "named in the starter code (do not rename it). Do NOT read "
            "from stdin or print results — the test harness instantiates "
            "`Solution()` and calls the method directly with the parsed "
            "Python literals as positional arguments, then compares the "
            "return value to the expected output."
            + starter_block
        )
    if has_stdin and not has_func:
        return (
            "I/O FORMAT (do not deviate): this task uses competitive-"
            "programming I/O. Read input via `input()` or `sys.stdin`, "
            "parse it according to the problem statement, and print the "
            "result(s) to stdout. Do NOT define a `class Solution`; the "
            "test harness runs the script as `python solution.py`, feeds "
            "stdin, and compares the captured stdout to the expected "
            "output (trailing whitespace stripped)."
        )
    return ""


def render_conversation_for_llm_b(
    turns: list[SimulatedTurn], latest_user_message: str
) -> list[dict]:
    """Shared helper: convert SimulatedTurn history + new user message into
    Bedrock Converse `messages` payload. Handles alternation.
    """
    messages: list[dict] = []
    for t in turns:
        messages.append({"role": t.role, "content": t.content})
    messages.append({"role": "user", "content": latest_user_message})
    return messages
