"""End-to-end demo on the synthetic stream.

Run:
    python scripts/run_demo.py

This script wires every component together and walks through the fixture
episodes in `tests/fixtures/synthetic_episodes.json`. The LLM client is
currently a skeleton (see skillmap/llm/client.py), so running this today
will raise NotImplementedError on the first call. It is structured to be
runnable once Phase 2 completes.

Phase 5 gate (spec §8): this script runs end-to-end without error and the
final SkillMap state contains >= 1 confirmed skill organized in a
non-trivial DAG.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

import yaml

from skillmap.orchestrator import Orchestrator, OrchestratorConfig
from skillmap.storage import JSONPersistence, SkillMap


ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "config.yaml"
FIXTURE_PATH = ROOT / "tests" / "fixtures" / "synthetic_episodes.json"


def _load_config() -> dict:
    return yaml.safe_load(CONFIG_PATH.read_text())


async def _run() -> None:
    cfg = _load_config()
    storage_path = Path(cfg["skill_map"]["storage_path"])
    persistence = JSONPersistence(storage_path, user_id="demo-user")
    skill_map = SkillMap(persistence.load(), persistence)
    skill_map.prereq_mastery_threshold = cfg["skill_map"]["prereq_mastery_threshold"]
    skill_map.confirmation_episode_threshold = cfg["skill_map"]["confirmation_episode_threshold"]
    skill_map.split_disambiguation_threshold = cfg["skill_map"]["split_disambiguation_threshold"]

    orchestrator = Orchestrator(
        skill_map=skill_map,
        config=OrchestratorConfig(categories=cfg["categories"]["predefined"]),
    )

    fixture = json.loads(FIXTURE_PATH.read_text())
    for i, task in enumerate(fixture["episodes"]):
        print(f"\n=== Task {i}: {task['task_category']} ===")
        print(f"User: {task['query']}")
        response = await orchestrator.handle_query(task["query"])
        print(f"Assistant: {response}")
        # Scripted corrections would be appended to the pending episode's
        # trajectory in a fuller demo. v0: we simply finalize.
        episode = await orchestrator.finalize_task(task["outcome"])
        print(
            f"Retrieved skills: {episode.retrieved_skills_at_start}; "
            f"corrections: {len(episode.correction_points)}"
        )

    confirmed = [s for s in skill_map.list_skills() if s.status == "confirmed"]
    print(f"\nFinal SkillMap: {len(skill_map.list_skills())} skills "
          f"({len(confirmed)} confirmed)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--reset",
        action="store_true",
        help="delete existing skill_map.json before running",
    )
    args = parser.parse_args()

    if args.reset:
        cfg = _load_config()
        p = Path(cfg["skill_map"]["storage_path"])
        if p.exists():
            p.unlink()

    asyncio.run(_run())


if __name__ == "__main__":
    main()
