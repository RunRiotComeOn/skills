"""Test-case pass-rate trajectory — PRIMARY metric for the correctness-skill axis.

Run the assistant's final code in a sandboxed subprocess against each
test case. Never block the eval on subprocess failures (we return None
for "couldn't run", which gets dropped from the average).

The trajectory metric splits the stream into an early window and a late
window and reports the average pass rate of each, plus the held-out
average. Reading the result:

  • SkillMap should show late > early if the induced correctness skills
    are actually moving the model toward fewer bugs across tasks.
  • Stateless and declarative_memory baselines should be roughly flat —
    they have no per-user procedural memory for bug classes.
  • Held-out gap (late vs held-out) shows how much of the lift is genuine
    generalization vs in-stream memorization.

Handles two LCB testtype variants:
  - "stdin"      : feed input via stdin, compare stdout (only attempted when
                   the extracted code actually reads from stdin)
  - "functional" : parse inputs as Python literals, call Solution().method(),
                   compare return value to expected literal
"""

from __future__ import annotations

import ast
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from skillmap_eval.types import (
    ConditionName,
    CorrectnessTrajectory,
    EvalTask,
    StreamRun,
    TaskInteraction,
)


_CODE_FENCE_RE = re.compile(r"```(?:python|py)?\s*(.*?)```", re.DOTALL)
_SOLUTION_METHOD_RE = re.compile(
    r"class\s+Solution\b.*?def\s+([a-zA-Z_]\w*)\s*\(self",
    re.DOTALL,
)

# AIME answers are integers 0–999.
_BOXED_RE = re.compile(r"\\boxed\{(\d{1,3})\}")
_ANSWER_PHRASE_RE = re.compile(
    r"(?:the\s+)?answer\s+(?:is\s+)?[:\s]*(\d{1,3})\b", re.IGNORECASE
)
_TRAILING_INT_RE = re.compile(r"\b(\d{1,3})\b")


def extract_aime_answer(text: str) -> str | None:
    """Extract the final integer answer (0–999) from an assistant response.

    Priority: \\boxed{N} > "the answer is N" > last standalone integer in range.
    """
    m = _BOXED_RE.search(text)
    if m:
        return m.group(1)
    m = _ANSWER_PHRASE_RE.search(text)
    if m:
        return m.group(1)
    candidates = [m for m in _TRAILING_INT_RE.findall(text) if 0 <= int(m) <= 999]
    return candidates[-1] if candidates else None


def extract_python_code(message: str) -> str | None:
    matches = _CODE_FENCE_RE.findall(message)
    if matches:
        return matches[-1].strip()
    # Fallback: heuristic line scan
    lines = message.splitlines()
    for i, line in enumerate(lines):
        if line.lstrip().startswith(("def ", "class ", "import ", "from ")):
            return "\n".join(lines[i:]).strip()
    return None


def run_sanity_check_for_task(
    task: EvalTask,
    assistant_message: str,
    timeout_s: float = 10.0,
) -> float | None:
    """Return pass rate in [0, 1], or None when the answer cannot be evaluated.

    For AIME tasks: extracts the final integer from the response and compares
    to the reference answer. Returns 1.0 (correct), 0.0 (wrong), or None
    (no answer found).

    For LCB tasks: runs Python code against test cases. None means "couldn't
    run"; 0.0 means "ran but all tests failed".
    """
    if task.source == "aime":
        return _check_aime_answer(task, assistant_message)

    code = extract_python_code(assistant_message)
    if not code or not task.test_cases:
        return None

    func_cases = [tc for tc in task.test_cases if tc.get("testtype") == "functional"]
    stdin_cases = [tc for tc in task.test_cases if tc.get("testtype") == "stdin"]

    results: list[bool] = []

    # --- functional: Solution().method(*args) ---
    if func_cases:
        method = _find_solution_method(code)
        if method:
            for tc in func_cases:
                ok = _run_functional_test(code, tc, method, timeout_s)
                if ok is not None:
                    results.append(ok)

    # --- stdin: only attempt if the code reads from stdin ---
    if stdin_cases and _reads_stdin(code):
        for tc in stdin_cases:
            ok = _run_stdin_test(code, tc, timeout_s)
            if ok is not None:
                results.append(ok)

    return (sum(results) / len(results)) if results else None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _check_aime_answer(task: EvalTask, assistant_message: str) -> float | None:
    if not task.test_cases:
        return None
    expected = str(task.test_cases[0].get("answer", "")).strip()
    if not expected:
        return None
    extracted = extract_aime_answer(assistant_message)
    if extracted is None:
        return None
    return 1.0 if extracted == expected else 0.0


def _reads_stdin(code: str) -> bool:
    return "input(" in code or "sys.stdin" in code or "stdin.read" in code


def _find_solution_method(code: str) -> str | None:
    m = _SOLUTION_METHOD_RE.search(code)
    return m.group(1) if m else None


def _run_functional_test(
    code: str,
    tc: dict[str, Any],
    method: str,
    timeout_s: float,
) -> bool | None:
    input_str = tc.get("input", "")
    expected_str = tc.get("output", "")
    if not isinstance(input_str, str) or not isinstance(expected_str, str):
        return None

    # Parse each line of input as a Python literal
    try:
        args = [
            ast.literal_eval(line.strip())
            for line in input_str.strip().splitlines()
            if line.strip()
        ]
        expected = ast.literal_eval(expected_str.strip())
    except (ValueError, SyntaxError):
        return None

    runner = (
        f"{code}\n\n"
        f"import sys\n"
        f"try:\n"
        f"    _sol = Solution()\n"
        f"    _result = _sol.{method}(*{args!r})\n"
        f"    print(repr(_result))\n"
        f"except Exception as _e:\n"
        f"    print('ERROR:', _e, file=sys.stderr)\n"
        f"    sys.exit(1)\n"
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        script = Path(tmpdir) / "solution.py"
        script.write_text(runner, encoding="utf-8")
        try:
            proc = subprocess.run(
                [sys.executable, str(script)],
                capture_output=True,
                text=True,
                timeout=timeout_s,
            )
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False
        if proc.returncode != 0:
            return False
        try:
            actual = ast.literal_eval(proc.stdout.strip())
            return actual == expected
        except (ValueError, SyntaxError):
            # Fallback: string comparison
            return proc.stdout.strip() == expected_str.strip()


def _run_stdin_test(
    code: str,
    tc: dict[str, Any],
    timeout_s: float,
) -> bool | None:
    stdin = tc.get("input")
    expected = tc.get("output")
    if not isinstance(stdin, str) or not isinstance(expected, str):
        return None

    with tempfile.TemporaryDirectory() as tmpdir:
        script = Path(tmpdir) / "solution.py"
        script.write_text(code, encoding="utf-8")
        try:
            proc = subprocess.run(
                [sys.executable, str(script)],
                input=stdin,
                capture_output=True,
                text=True,
                timeout=timeout_s,
            )
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False
        return proc.stdout.strip() == expected.strip()


def format_test_results_for_llm(
    task: EvalTask,
    assistant_message: str,
    max_cases: int = 2,
    timeout_s: float = 5.0,
) -> str:
    """Return a short human-readable result string to show the simulator LLM.

    For AIME: checks whether the extracted integer answer is correct. Does NOT
    reveal the expected answer, so the simulator cannot simply relay it to the
    model — it can only say "that's wrong, try again."

    For LCB: runs test cases and reports input/output mismatches.

    Returns empty string when no answer/code was found or no test cases exist.
    """
    if task.source == "aime":
        return _format_aime_result_for_llm(task, assistant_message)

    code = extract_python_code(assistant_message)
    if not code or not task.test_cases:
        return ""

    func_cases = [tc for tc in task.test_cases if tc.get("testtype") == "functional"][:max_cases]
    stdin_cases = [tc for tc in task.test_cases if tc.get("testtype") == "stdin"][:max_cases]

    lines: list[str] = []

    if func_cases:
        method = _find_solution_method(code)
        if method:
            for i, tc in enumerate(func_cases, 1):
                result = _run_functional_test_with_output(code, tc, method, timeout_s)
                lines.append(_format_result_line(i, tc.get("output", ""), result))

    if stdin_cases and _reads_stdin(code):
        offset = len(lines)
        for i, tc in enumerate(stdin_cases, offset + 1):
            result = _run_stdin_test_with_output(code, tc, timeout_s)
            lines.append(_format_result_line(i, tc.get("output", ""), result))

    if not lines:
        return ""

    passed = sum(1 for l in lines if l.startswith("  ✓"))
    header = f"Test results ({passed}/{len(lines)} sample cases passed):"
    return header + "\n" + "\n".join(lines)


def _format_aime_result_for_llm(task: EvalTask, assistant_message: str) -> str:
    if not task.test_cases:
        return ""
    expected = str(task.test_cases[0].get("answer", "")).strip()
    if not expected:
        return ""
    extracted = extract_aime_answer(assistant_message)
    if extracted is None:
        return ""
    if extracted == expected:
        return "Answer check: ✓ correct"
    return "Answer check: ✗ your final answer is incorrect"


def _format_result_line(idx: int, expected: str, actual_or_err: str | None) -> str:
    exp = expected.strip()[:60]
    if actual_or_err is None:
        return f"  ? Case {idx}: could not run"
    got = actual_or_err.strip()[:60]
    if got == exp:
        return f"  ✓ Case {idx}: correct"
    return f"  ✗ Case {idx}: expected {exp!r}, got {got!r}"


def _run_functional_test_with_output(
    code: str, tc: dict[str, Any], method: str, timeout_s: float
) -> str | None:
    input_str = tc.get("input", "")
    if not isinstance(input_str, str):
        return None
    try:
        args = [
            ast.literal_eval(line.strip())
            for line in input_str.strip().splitlines()
            if line.strip()
        ]
    except (ValueError, SyntaxError):
        return None

    runner = (
        f"{code}\n\n"
        f"import sys\n"
        f"try:\n"
        f"    _sol = Solution()\n"
        f"    _result = _sol.{method}(*{args!r})\n"
        f"    print(repr(_result))\n"
        f"except Exception as _e:\n"
        f"    print('ERROR:', _e, file=sys.stderr)\n"
        f"    sys.exit(1)\n"
    )
    with tempfile.TemporaryDirectory() as tmpdir:
        script = Path(tmpdir) / "solution.py"
        script.write_text(runner, encoding="utf-8")
        try:
            proc = subprocess.run(
                [sys.executable, str(script)],
                capture_output=True, text=True, timeout=timeout_s,
            )
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return "timeout"
        if proc.returncode != 0:
            return f"runtime error"
        return proc.stdout.strip()


def _run_stdin_test_with_output(
    code: str, tc: dict[str, Any], timeout_s: float
) -> str | None:
    stdin = tc.get("input")
    if not isinstance(stdin, str):
        return None
    with tempfile.TemporaryDirectory() as tmpdir:
        script = Path(tmpdir) / "solution.py"
        script.write_text(code, encoding="utf-8")
        try:
            proc = subprocess.run(
                [sys.executable, str(script)],
                input=stdin, capture_output=True, text=True, timeout=timeout_s,
            )
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return "timeout"
        return proc.stdout.strip() if proc.returncode == 0 else "runtime error"


def compute_correctness_trajectory(run: StreamRun) -> CorrectnessTrajectory:
    """Split the stream in half, average pass rate in each half, plus held-out."""
    stream = list(run.interactions)
    held_out = list(run.held_out_interactions)
    midpoint = len(stream) // 2
    early = stream[:midpoint]
    late = stream[midpoint:]

    return CorrectnessTrajectory(
        condition_name=_cast(run.condition_name),
        avg_pass_rate_early=mean_test_case_pass_rate(early)[0],
        avg_pass_rate_late=mean_test_case_pass_rate(late)[0],
        avg_pass_rate_held_out=mean_test_case_pass_rate(held_out)[0],
        task_completion_rate=task_completion_rate(stream + held_out),
    )


def mean_test_case_pass_rate(
    interactions: list[TaskInteraction],
) -> tuple[float, int]:
    """Average test_case_pass_rate over interactions whose code was MEASURABLE.

    Tasks with `test_case_pass_rate is None` (no extractable code, no
    matching test cases) are dropped — counted via the second element of
    the returned tuple so callers can distinguish "0% pass rate over 5
    measurable tasks" from "0 measurable tasks".

    See `mean_test_case_pass_rate_strict` for the variant that treats
    unmeasurable tasks as 0.0 (failed).
    """
    rates = [
        i.test_case_pass_rate for i in interactions
        if i.test_case_pass_rate is not None
    ]
    return ((sum(rates) / len(rates)) if rates else 0.0, len(rates))


def mean_test_case_pass_rate_strict(
    interactions: list[TaskInteraction],
) -> float:
    """Average test_case_pass_rate over ALL interactions; None counts as 0.

    Strict denominator = len(interactions). This is the metric to look at
    when "no extractable code" should be punished as a delivery failure
    (the assistant didn't ship runnable code) rather than ignored.
    """
    if not interactions:
        return 0.0
    total = sum(
        (i.test_case_pass_rate if i.test_case_pass_rate is not None else 0.0)
        for i in interactions
    )
    return total / len(interactions)


def task_completion_rate(interactions: list[TaskInteraction]) -> float:
    """Fraction of interactions whose simulator chose `user_accepted`."""
    if not interactions:
        return 0.0
    accepted = sum(1 for i in interactions if i.completion_reason == "user_accepted")
    return accepted / len(interactions)


def _cast(name: str) -> ConditionName:
    if name not in ("stateless", "declarative_memory", "skillmap"):
        raise ValueError(f"unexpected condition: {name!r}")
    return name  # type: ignore[return-value]
