"""LiveCodeBench loader + stream sampler.

Primary mode is local-cache first: if `cache_dir` already contains the raw
`test*.jsonl` files or a derived `<split>.jsonl` cache, load from there
without any network access.

Fallback mode uses Hugging Face `datasets` when available. Some newer
`datasets` releases no longer support dataset scripts, so callers should
prefer pre-populating `cache_dir` with raw jsonl files.

Test cases are used ONLY by the correctness_sanity metric; they are never
passed to the simulator.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

from skillmap_eval.types import EvalTask


_DIFFICULTY_MAP = {
    "easy": "easy",
    "medium": "medium",
    "hard": "hard",
}


class LiveCodeBenchLoader:
    def __init__(
        self,
        cache_dir: str,
        hf_name: str = "livecodebench/code_generation_lite",
        hf_split: str = "test",
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.hf_name = hf_name
        self.hf_split = hf_split

    async def load(self) -> list[EvalTask]:
        """Load the LCB split from a derived cache, local raw files, or HF."""
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file = self.cache_dir / f"{self.hf_split}.eval_tasks.jsonl"
        if cache_file.exists():
            return [
                EvalTask.model_validate_json(line)
                for line in cache_file.read_text().splitlines()
                if line
            ]

        raw_files = self._local_raw_files()
        if raw_files:
            tasks = self._load_from_local_raw_files(raw_files)
            self._write_cache(cache_file, tasks)
            return tasks

        try:
            from datasets import load_dataset  # type: ignore
        except ImportError as e:
            raise RuntimeError(
                "The 'datasets' package is required for LiveCodeBenchLoader.load(). "
                "Install with: pip install datasets, or place raw LiveCodeBench "
                f"files like test.jsonl under {self.cache_dir}"
            ) from e

        try:
            ds = load_dataset(self.hf_name, split=self.hf_split)
        except RuntimeError as e:
            msg = str(e)
            if "Dataset scripts are no longer supported" in msg:
                raise RuntimeError(
                    "LiveCodeBench HF loading failed because this datasets version no "
                    "longer supports dataset scripts. Pre-populate cache_dir with raw "
                    f"files like test.jsonl under {self.cache_dir}."
                ) from e
            raise

        tasks = self._rows_to_tasks(ds)
        self._write_cache(cache_file, tasks)
        return tasks

    def sample_stream(
        self,
        tasks: list[EvalTask],
        n_stream: int,
        n_held_out: int,
        difficulty_mix: dict[str, float],
        seed: int,
    ) -> tuple[list[EvalTask], list[EvalTask]]:
        """Return (stream_tasks, held_out_tasks).

        - disjoint sets
        - each set matches the difficulty_mix
        - stream is ordered easy -> hard across its length
        """
        _validate_mix(difficulty_mix)
        rng = random.Random(seed)

        by_diff: dict[str, list[EvalTask]] = {d: [] for d in difficulty_mix}
        for t in tasks:
            if t.difficulty in by_diff:
                by_diff[t.difficulty].append(t)
        for lst in by_diff.values():
            rng.shuffle(lst)

        total = n_stream + n_held_out
        counts = _integer_allocation(difficulty_mix, total)
        for diff, need in counts.items():
            if len(by_diff[diff]) < need:
                raise ValueError(
                    f"not enough {diff} tasks: need {need}, have {len(by_diff[diff])}"
                )

        picked: dict[str, list[EvalTask]] = {
            diff: by_diff[diff][:need] for diff, need in counts.items()
        }

        stream_counts = _integer_allocation(difficulty_mix, n_stream)
        held_counts = {d: counts[d] - stream_counts[d] for d in counts}

        stream_tasks: list[EvalTask] = []
        held_tasks: list[EvalTask] = []
        for diff in counts:
            stream_tasks.extend(picked[diff][: stream_counts[diff]])
            held_tasks.extend(picked[diff][stream_counts[diff] :])

        # Stream order: ramp easy -> medium -> hard.
        stream_tasks.sort(key=lambda t: _DIFFICULTY_ORDER[t.difficulty])

        return stream_tasks, held_tasks

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _row_to_task(row: dict[str, Any]) -> EvalTask | None:
        """Best-effort field mapping. LCB field names vary by release;
        adjust here if a new release lands.
        """
        task_id = str(row.get("question_id") or row.get("task_id") or row.get("id") or "")
        if not task_id:
            return None
        problem = (
            row.get("question_content")
            or row.get("problem_description")
            or row.get("prompt")
            or ""
        )
        raw_diff = str(row.get("difficulty") or "").lower()
        if raw_diff not in _DIFFICULTY_MAP:
            return None
        test_cases = row.get("public_test_cases") or row.get("private_test_cases") or []
        if isinstance(test_cases, str):
            try:
                test_cases = json.loads(test_cases)
            except json.JSONDecodeError:
                test_cases = []
        if not isinstance(test_cases, list):
            test_cases = []
        return EvalTask(
            task_id=task_id,
            source="livecodebench",
            problem_statement=problem,
            reference_solution=row.get("starter_code") or None,
            test_cases=test_cases,
            difficulty=_DIFFICULTY_MAP[raw_diff],
            contest_date=row.get("contest_date") or None,
        )

    def _local_raw_files(self) -> list[Path]:
        """Return raw `test*.jsonl` files under cache_dir, in numeric order."""
        def sort_key(path: Path) -> tuple[int, str]:
            name = path.name
            if name == "test.jsonl":
                return (1, name)
            suffix = name.removeprefix("test").removesuffix(".jsonl")
            try:
                return (int(suffix), name)
            except ValueError:
                return (999, name)

        files = sorted(self.cache_dir.glob("test*.jsonl"), key=sort_key)
        return [path for path in files if path.is_file() and path.name != f"{self.hf_split}.jsonl"]

    def _load_from_local_raw_files(self, raw_files: list[Path]) -> list[EvalTask]:
        tasks: list[EvalTask] = []
        for file_path in raw_files:
            with file_path.open(encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    task = self._row_to_task(json.loads(line))
                    if task is not None:
                        tasks.append(task)
        return tasks

    def _rows_to_tasks(self, rows: Any) -> list[EvalTask]:
        tasks: list[EvalTask] = []
        for row in rows:
            task = self._row_to_task(row)
            if task is not None:
                tasks.append(task)
        return tasks

    @staticmethod
    def _write_cache(cache_file: Path, tasks: list[EvalTask]) -> None:
        with cache_file.open("w", encoding="utf-8") as f:
            for task in tasks:
                f.write(task.model_dump_json() + "\n")


_DIFFICULTY_ORDER = {"easy": 0, "medium": 1, "hard": 2}


def _validate_mix(mix: dict[str, float]) -> None:
    if abs(sum(mix.values()) - 1.0) > 1e-6:
        raise ValueError(f"difficulty_mix must sum to 1.0, got {sum(mix.values())}")
    for k in mix:
        if k not in _DIFFICULTY_MAP:
            raise ValueError(f"unknown difficulty: {k!r}")


def _integer_allocation(mix: dict[str, float], total: int) -> dict[str, int]:
    """Allocate `total` items across categories by `mix`, preserving the sum."""
    raw = {k: v * total for k, v in mix.items()}
    floor = {k: int(v) for k, v in raw.items()}
    deficit = total - sum(floor.values())
    # distribute remainders by largest fractional part
    fracs = sorted(((raw[k] - floor[k], k) for k in raw), reverse=True)
    for _, k in fracs[:deficit]:
        floor[k] += 1
    return floor
