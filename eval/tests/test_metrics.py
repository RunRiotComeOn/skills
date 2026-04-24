"""Tests for metric modules. These do not call LLMs (judge excepted)."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from skillmap_eval.metrics.correction_rate import compute_correction_curve
from skillmap_eval.metrics.correctness_sanity import (
    compute_correctness_trajectory,
    extract_python_code,
    run_sanity_check_for_task,
)
from skillmap_eval.metrics.generalization import compute_generalization
from skillmap_eval.types import (
    EvalTask,
    StreamRun,
    TaskInteraction,
)


def _make_interaction(
    idx: int,
    correction_count: int,
    pass_rate: float | None = 1.0,
    completion: str = "user_accepted",
    is_held_out: bool = False,
    pref_count: int | None = None,
    corr_count: int | None = None,
) -> TaskInteraction:
    # By default, attribute the whole correction_count to the preference axis
    # so legacy tests keep their original semantics.
    p = pref_count if pref_count is not None else correction_count
    c = corr_count if corr_count is not None else 0
    return TaskInteraction(
        task_id=f"t{idx}",
        condition_name="stateless",
        task_index_in_stream=idx,
        turns=[],
        correction_count=correction_count,
        preference_correction_count=p,
        correctness_correction_count=c,
        completion_reason=completion,  # type: ignore[arg-type]
        test_case_pass_rate=pass_rate,
        retrieved_skill_ids_at_start=[],
    )


def _make_run(
    condition: str,
    stream_counts: list[int],
    held_out_counts: list[int],
) -> StreamRun:
    return StreamRun(
        run_id="r1",
        profile_id="p1",
        condition_name=condition,  # type: ignore[arg-type]
        task_stream=[f"t{i}" for i in range(len(stream_counts))],
        held_out_task_ids=[f"h{i}" for i in range(len(held_out_counts))],
        interactions=[_make_interaction(i, c) for i, c in enumerate(stream_counts)],
        held_out_interactions=[
            _make_interaction(i, c, is_held_out=True)
            for i, c in enumerate(held_out_counts)
        ],
        started_at=datetime.now(timezone.utc),
    )


def test_correction_curve_rolling_mean_window_3() -> None:
    run = _make_run("skillmap", [3, 2, 1, 0, 0], [])
    curve = compute_correction_curve(run, window=3)
    assert curve.total_per_task == [3, 2, 1, 0, 0]
    assert curve.preference_per_task == [3, 2, 1, 0, 0]  # default attribution
    assert curve.correctness_per_task == [0, 0, 0, 0, 0]
    # Rolling means (total): [3], 2.5, 2.0, 1.0, 1/3
    assert curve.rolling_mean_window_3_total[0] == 3.0
    assert curve.rolling_mean_window_3_total[1] == 2.5
    assert curve.rolling_mean_window_3_total[2] == 2.0
    assert curve.rolling_mean_window_3_total[3] == 1.0
    assert abs(curve.rolling_mean_window_3_total[4] - 1.0 / 3) < 1e-9


def test_generalization_handles_empty_held_out() -> None:
    run = _make_run("stateless", [1, 2, 3], [])
    g = compute_generalization(run)
    assert g.held_out_avg_correction_count == 0.0
    assert g.held_out_median_correction_count == 0.0
    assert g.held_out_avg_preference_corrections == 0.0
    assert g.held_out_avg_correctness_corrections == 0.0


def test_correctness_trajectory_averages_pass_rates() -> None:
    # Stream of 4 with pass_rate=1.0; held-out of 2 with pass_rate=1.0.
    run = _make_run("stateless", [0, 0, 0, 0], [0, 0])
    t = compute_correctness_trajectory(run)
    assert t.avg_pass_rate_early == 1.0
    assert t.avg_pass_rate_late == 1.0
    assert t.avg_pass_rate_held_out == 1.0
    assert t.task_completion_rate == 1.0


def test_extract_python_code_from_fenced_block() -> None:
    message = "Sure, here you go:\n```python\ndef f():\n    return 42\n```\nThat's it."
    code = extract_python_code(message)
    assert code == "def f():\n    return 42"


def test_extract_python_code_returns_none_when_no_code() -> None:
    assert extract_python_code("just prose, no code here") is None


def test_run_sanity_check_returns_none_when_no_code() -> None:
    fixture_path = Path(__file__).parent / "fixtures" / "sample_task.json"
    task = EvalTask.model_validate_json(fixture_path.read_text())
    # No code → cannot evaluate → None (not 0.0).
    assert run_sanity_check_for_task(task, "no code at all") is None


def test_run_sanity_check_executes_passing_code() -> None:
    fixture_path = Path(__file__).parent / "fixtures" / "sample_task.json"
    task = EvalTask.model_validate_json(fixture_path.read_text())
    assistant_msg = (
        "Here is the solution:\n"
        "```python\n"
        "import sys\n"
        "nums = list(map(int, sys.stdin.read().split()))\n"
        "best = float('-inf')\n"
        "out = []\n"
        "for n in nums:\n"
        "    if n > best:\n"
        "        best = n\n"
        "    out.append(str(best))\n"
        "print(' '.join(out))\n"
        "```\n"
    )
    rate = run_sanity_check_for_task(task, assistant_msg, timeout_s=5.0)
    assert rate == 1.0
