"""Metric 4: test-case pass-rate sanity check.

Run the assistant's final code in a sandboxed subprocess against each
test case. Failure to extract code => pass rate 0, NOT an error. Never
block the eval on sanity-check failures.
"""

from __future__ import annotations

import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from skillmap_eval.types import (
    ConditionName,
    CorrectnessSanity,
    EvalTask,
    StreamRun,
)


_CODE_FENCE_RE = re.compile(r"```(?:python|py)?\s*(.*?)```", re.DOTALL)


def extract_python_code(message: str) -> str | None:
    matches = _CODE_FENCE_RE.findall(message)
    if matches:
        return matches[-1].strip()  # prefer last (most recent) code block
    # Fallback: heuristic "starts with def" scan.
    lines = message.splitlines()
    for i, line in enumerate(lines):
        if line.lstrip().startswith(("def ", "class ", "import ", "from ")):
            return "\n".join(lines[i:]).strip()
    return None


def run_sanity_check_for_task(
    task: EvalTask,
    assistant_message: str,
    timeout_s: float = 10.0,
) -> float:
    """Return a pass rate in [0, 1]. Returns 0.0 on any failure; never raises."""
    code = extract_python_code(assistant_message)
    if not code or not task.test_cases:
        return 0.0

    passed = 0
    for tc in task.test_cases:
        ok = _run_one_test(code, tc, timeout_s)
        if ok:
            passed += 1
    return passed / len(task.test_cases)


def _run_one_test(code: str, tc: dict[str, Any], timeout_s: float) -> bool:
    """Very conservative: write `code` + a small runner into a temp file,
    execute with CPython, compare stdout to expected.

    `tc` shape (LCB-ish):
      {"input": "...", "output": "...", "testtype": "stdin"}
    We only handle the stdin/stdout variant; any other shape -> False.
    """
    stdin = tc.get("input")
    expected = tc.get("output")
    if not isinstance(stdin, str) or not isinstance(expected, str):
        return False

    with tempfile.TemporaryDirectory() as tmpdir:
        script = Path(tmpdir) / "solution.py"
        script.write_text(code, encoding="utf-8")
        try:
            result = subprocess.run(
                [sys.executable, str(script)],
                input=stdin,
                capture_output=True,
                text=True,
                timeout=timeout_s,
            )
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False
        return result.stdout.strip() == expected.strip()


def compute_correctness_sanity(run: StreamRun) -> CorrectnessSanity:
    all_interactions = list(run.interactions) + list(run.held_out_interactions)
    rates = [i.test_case_pass_rate for i in all_interactions if i.test_case_pass_rate is not None]
    avg_pass = (sum(rates) / len(rates)) if rates else 0.0
    completed = sum(1 for i in all_interactions if i.completion_reason == "user_accepted")
    completion_rate = completed / len(all_interactions) if all_interactions else 0.0
    return CorrectnessSanity(
        condition_name=_cast(run.condition_name),
        avg_test_pass_rate=avg_pass,
        task_completion_rate=completion_rate,
    )


def _cast(name: str) -> ConditionName:
    if name not in ("stateless", "declarative_memory", "skillmap"):
        raise ValueError(f"unexpected condition: {name!r}")
    return name  # type: ignore[return-value]
