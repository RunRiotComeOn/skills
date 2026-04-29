"""Tests for metric modules. These do not call LLMs (judge excepted)."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from skillmap_eval.metrics.correction_rate import compute_correction_curve
from skillmap_eval.metrics.correctness_sanity import (
    compute_correctness_trajectory,
    extract_python_code,
    mean_test_case_pass_rate,
    mean_test_case_pass_rate_strict,
    run_sanity_check_for_task,
)
from skillmap_eval.metrics.generalization import compute_generalization
from skillmap_eval.conditions.base import format_task_hint
from skillmap_eval.metrics.preference_acceptance import (
    compute_preference_trajectory,
    mean_first_turn_preference_acceptance_rate,
    mean_first_turn_preference_violation_count,
)
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
    acceptance_rate: float | None = 1.0,
    first_turn_acceptance_rate: float | None = None,
    first_turn_violations: int | None = None,
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
        preference_acceptance_rate=acceptance_rate,
        first_turn_preference_acceptance_rate=first_turn_acceptance_rate,
        first_turn_preference_violation_count=first_turn_violations,
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


def test_mean_test_case_pass_rate_measurable_drops_none() -> None:
    interactions = [
        _make_interaction(0, 0, pass_rate=None),
        _make_interaction(1, 0, pass_rate=0.5),
        _make_interaction(2, 0, pass_rate=1.0),
    ]
    mean, n = mean_test_case_pass_rate(interactions)
    assert n == 2
    assert mean == 0.75


def test_mean_test_case_pass_rate_strict_counts_none_as_zero() -> None:
    interactions = [
        _make_interaction(0, 0, pass_rate=None),
        _make_interaction(1, 0, pass_rate=0.5),
        _make_interaction(2, 0, pass_rate=1.0),
    ]
    # Strict: (0 + 0.5 + 1.0) / 3 = 0.5
    assert mean_test_case_pass_rate_strict(interactions) == 0.5


def test_mean_test_case_pass_rate_strict_empty_returns_zero() -> None:
    assert mean_test_case_pass_rate_strict([]) == 0.0


def test_correctness_trajectory_averages_pass_rates() -> None:
    # Stream of 4 with pass_rate=1.0; held-out of 2 with pass_rate=1.0.
    run = _make_run("stateless", [0, 0, 0, 0], [0, 0])
    t = compute_correctness_trajectory(run)
    assert t.avg_pass_rate_early == 1.0
    assert t.avg_pass_rate_late == 1.0
    assert t.avg_pass_rate_held_out == 1.0
    assert t.task_completion_rate == 1.0


def test_preference_trajectory_averages_acceptance_rates() -> None:
    # Construct a stream where late-half acceptance is higher than early-half,
    # mirroring the success case for SkillMap on the preference axis.
    run = StreamRun(
        run_id="r1",
        profile_id="p1",
        condition_name="skillmap",
        task_stream=["t0", "t1", "t2", "t3"],
        held_out_task_ids=["h0", "h1"],
        interactions=[
            _make_interaction(0, 0, acceptance_rate=0.5),
            _make_interaction(1, 0, acceptance_rate=0.5),
            _make_interaction(2, 0, acceptance_rate=1.0),
            _make_interaction(3, 0, acceptance_rate=1.0),
        ],
        held_out_interactions=[
            _make_interaction(0, 0, acceptance_rate=0.75, is_held_out=True),
            _make_interaction(1, 0, acceptance_rate=0.75, is_held_out=True),
        ],
        started_at=datetime.now(timezone.utc),
    )
    t = compute_preference_trajectory(run)
    assert t.avg_acceptance_rate_early == 0.5
    assert t.avg_acceptance_rate_late == 1.0
    assert t.avg_acceptance_rate_held_out == 0.75


def test_preference_trajectory_drops_none_rates() -> None:
    # Tasks with acceptance_rate=None (e.g., empty profile) are dropped, not
    # treated as 0.0.
    run = StreamRun(
        run_id="r1",
        profile_id="p1",
        condition_name="stateless",
        task_stream=["t0", "t1"],
        held_out_task_ids=[],
        interactions=[
            _make_interaction(0, 0, acceptance_rate=None),
            _make_interaction(1, 0, acceptance_rate=0.4),
        ],
        held_out_interactions=[],
        started_at=datetime.now(timezone.utc),
    )
    t = compute_preference_trajectory(run)
    # Early window (first half) is just t0 with None → mean over empty → 0.0.
    assert t.avg_acceptance_rate_early == 0.0
    # Late window is t1 with 0.4.
    assert t.avg_acceptance_rate_late == 0.4
    assert t.avg_acceptance_rate_held_out == 0.0


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


def test_first_turn_metrics_drop_none_and_average() -> None:
    interactions = [
        _make_interaction(0, 0, first_turn_acceptance_rate=None, first_turn_violations=None),
        _make_interaction(1, 0, first_turn_acceptance_rate=0.6, first_turn_violations=4),
        _make_interaction(2, 0, first_turn_acceptance_rate=1.0, first_turn_violations=0),
    ]
    rate_mean, rate_n = mean_first_turn_preference_acceptance_rate(interactions)
    assert rate_n == 2
    assert rate_mean == 0.8

    count_mean, count_n = mean_first_turn_preference_violation_count(interactions)
    assert count_n == 2
    assert count_mean == 2.0


def test_format_task_hint_distinguishes_io_styles() -> None:
    base = dict(task_id="t", source="livecodebench", problem_statement="x", difficulty="easy")

    functional_only = EvalTask(
        **base, test_cases=[{"testtype": "functional", "input": "[]", "output": "0"}]
    )
    stdin_only = EvalTask(
        **base, test_cases=[{"testtype": "stdin", "input": "1", "output": "1"}]
    )
    mixed = EvalTask(
        **base,
        test_cases=[
            {"testtype": "functional", "input": "[]", "output": "0"},
            {"testtype": "stdin", "input": "1", "output": "1"},
        ],
    )
    no_cases = EvalTask(**base, test_cases=[])

    func_hint = format_task_hint(functional_only)
    assert "class Solution" in func_hint
    assert "Do NOT read from stdin" in func_hint

    stdin_hint = format_task_hint(stdin_only)
    assert "stdin" in stdin_hint.lower()
    assert "Do NOT define a `class Solution`" in stdin_hint

    # Mixed and empty cases must be empty so we don't lie to the model.
    assert format_task_hint(mixed) == ""
    assert format_task_hint(no_cases) == ""


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
