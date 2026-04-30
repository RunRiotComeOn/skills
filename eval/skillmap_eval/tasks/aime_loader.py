"""AIME (American Invitational Mathematics Examination) loader.

Primary mode: local JSONL cache under `cache_dir`.
Fallback: Hugging Face `datasets` (default: AI-MO/aimo-validation-aime).

Expected local JSONL format (one problem per line):
  {"id": "2023_I_Problem_7", "problem": "...", "answer": 42, "year": 2023}

Difficulty mapping by problem number within a given exam:
  1–5  → easy
  6–10 → medium
  11–15 → hard
Problems whose number cannot be inferred default to "medium".
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from skillmap_eval.types import EvalTask


_PROBLEM_NUM_RE = re.compile(r"\b(\d+)\b")

_DIFFICULTY_BY_NUMBER: dict[int, str] = {
    **{i: "easy" for i in range(1, 6)},
    **{i: "medium" for i in range(6, 11)},
    **{i: "hard" for i in range(11, 16)},
}


def _problem_number_from_id(task_id: str) -> int | None:
    """Extract a 1-15 problem position from an ID.

    Structured IDs like '2023_I_Problem_7' yield 7 directly. Bare sequential
    indices like '89' fold onto a within-exam position via ((N-1) % 15) + 1,
    since each AIME exam has 15 problems ordered by difficulty.
    """
    matches = [int(m) for m in _PROBLEM_NUM_RE.findall(task_id)]
    if not matches:
        return None
    in_range = [n for n in matches if 1 <= n <= 15]
    if in_range:
        return in_range[-1]
    n = matches[-1]
    return ((n - 1) % 15) + 1 if n >= 1 else None


class AIMELoader:
    def __init__(
        self,
        cache_dir: str,
        hf_name: str = "AI-MO/aimo-validation-aime",
        hf_split: str = "train",
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.hf_name = hf_name
        self.hf_split = hf_split

    async def load(self) -> list[EvalTask]:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file = self.cache_dir / "aime.eval_tasks.jsonl"
        if cache_file.exists():
            return [
                EvalTask.model_validate_json(line)
                for line in cache_file.read_text().splitlines()
                if line.strip()
            ]

        raw_files = [
            f for f in self.cache_dir.glob("*.jsonl")
            if f.name != "aime.eval_tasks.jsonl"
        ]
        if raw_files:
            tasks = self._load_from_local(raw_files)
            self._write_cache(cache_file, tasks)
            return tasks

        try:
            from datasets import load_dataset  # type: ignore
        except ImportError as e:
            raise RuntimeError(
                "The 'datasets' package is required for AIMELoader. "
                "Install with: pip install datasets, or place raw AIME "
                f"JSONL files under {self.cache_dir}"
            ) from e

        ds = load_dataset(self.hf_name, split=self.hf_split)
        tasks = [t for row in ds if (t := self._row_to_task(dict(row))) is not None]
        self._write_cache(cache_file, tasks)
        return tasks

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _load_from_local(self, files: list[Path]) -> list[EvalTask]:
        tasks: list[EvalTask] = []
        for path in files:
            with path.open(encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    task = self._row_to_task(json.loads(line))
                    if task is not None:
                        tasks.append(task)
        return tasks

    @staticmethod
    def _row_to_task(row: dict[str, Any]) -> EvalTask | None:
        task_id = str(
            row.get("id") or row.get("task_id") or row.get("ID") or ""
        ).strip()
        problem = (
            row.get("problem") or row.get("Problem")
            or row.get("question") or row.get("Question") or ""
        )
        raw_answer = row.get("answer") or row.get("Answer")
        if not task_id or not problem or raw_answer is None:
            return None

        try:
            answer_str = str(int(float(raw_answer)))
        except (ValueError, TypeError):
            answer_str = str(raw_answer).strip()

        # Problem number: explicit field or extracted from ID.
        pnum = row.get("problem_number") or row.get("Problem Number")
        if pnum is not None:
            try:
                pnum = int(pnum)
            except (ValueError, TypeError):
                pnum = None
        if pnum is None:
            pnum = _problem_number_from_id(task_id)

        difficulty = _DIFFICULTY_BY_NUMBER.get(pnum, "medium") if pnum else "medium"

        year = row.get("year") or row.get("Year")
        contest_date = str(int(year)) if year is not None else None

        return EvalTask(
            task_id=task_id,
            source="aime",
            problem_statement=str(problem),
            reference_solution=answer_str,
            test_cases=[{"answer": answer_str}],
            difficulty=difficulty,
            contest_date=contest_date,
        )

    @staticmethod
    def _write_cache(cache_file: Path, tasks: list[EvalTask]) -> None:
        with cache_file.open("w", encoding="utf-8") as f:
            for task in tasks:
                f.write(task.model_dump_json() + "\n")
