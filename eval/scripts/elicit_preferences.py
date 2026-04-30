"""Stage 1 script: generate a PreferenceProfile with LLM-A.

Usage:
    python scripts/elicit_preferences.py --task-type python_coding

Writes to: skillmap_eval/preferences/profiles/<profile_id>.json

Phase 1 gate: a human reads the generated profile and agrees that >= 8 of
the N preferences are opinionated and non-default. If not, iterate on the
prompt in skillmap_eval/preferences/prompts.py before proceeding.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

# Make `skillmap_eval` importable when run as a script.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import yaml

from skillmap_eval.preferences import PreferenceElicitor
from skillmap_eval.tasks import LiveCodeBenchLoader
from skillmap_eval.tasks.aime_loader import AIMELoader
from skillmap_eval.types import EvalTask


def _load_cfg(config_path: str | None) -> dict:
    path = Path(config_path) if config_path else ROOT / "eval_config.yaml"
    if not path.is_absolute():
        path = ROOT / path
    return yaml.safe_load(path.read_text(encoding="utf-8"))


async def _sample_task_examples(cfg: dict, n_examples: int = 3) -> list[EvalTask]:
    tasks_cfg = cfg["tasks"]
    source = tasks_cfg.get("source", "livecodebench")

    if source == "aime":
        loader = AIMELoader(cache_dir=tasks_cfg["cache_dir"])
        all_tasks = await loader.load()
        import random
        rng = random.Random(int(tasks_cfg["seed"]))
        rng.shuffle(all_tasks)
        return all_tasks[:n_examples]
    else:
        loader = LiveCodeBenchLoader(cache_dir=tasks_cfg["cache_dir"])
        all_tasks = await loader.load()
        difficulty_mix = {k: float(v) for k, v in tasks_cfg["difficulty_mix"].items()}
        stream_tasks, _ = loader.sample_stream(
            all_tasks,
            n_stream=n_examples,
            n_held_out=0,
            difficulty_mix=difficulty_mix,
            seed=int(tasks_cfg["seed"]),
        )
        return stream_tasks


async def _run(task_type: str, n_preferences: int, out_name: str | None, config_path: str | None) -> Path:
    cfg = _load_cfg(config_path)
    llm = cfg["llm"]
    model = llm.get("dev_override_a") or llm["llm_a_model"]
    region = llm.get("region", "us-east-1")
    retry_max = int(cfg["preferences"]["retry_max"])

    elicitor = PreferenceElicitor(
        llm_a_model=model, region=region, retry_max=retry_max
    )
    task_examples = await _sample_task_examples(cfg, n_examples=3)
    profile = await elicitor.run(
        task_type=task_type,
        n_preferences=n_preferences,
        task_examples=task_examples,
    )

    profiles_dir = Path(cfg["paths"]["profiles_dir"])
    if not profiles_dir.is_absolute():
        profiles_dir = ROOT / profiles_dir
    profiles_dir.mkdir(parents=True, exist_ok=True)

    profile_id = out_name or profile.profile_id
    # Force profile_id match on disk so the loader can find it by id.
    profile.profile_id = profile_id
    path = profiles_dir / f"{profile_id}.json"
    path.write_text(profile.model_dump_json(indent=2), encoding="utf-8")
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-type", default="python_coding")
    parser.add_argument("--n", type=int, default=None, help="override preferences.n_preferences")
    parser.add_argument("--name", default=None, help="explicit profile_id (filename without .json)")
    parser.add_argument("--config", default=None, help="path to eval config yaml (default: eval_config.yaml)")
    args = parser.parse_args()

    cfg = _load_cfg(args.config)
    n = args.n or int(cfg["preferences"]["n_preferences"])

    path = asyncio.run(_run(args.task_type, n, args.name, args.config))
    print(f"wrote profile -> {path}")
    print("Phase 1 gate: read the file, confirm >= N*0.8 preferences are non-default.")


if __name__ == "__main__":
    main()
