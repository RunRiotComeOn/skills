"""One-off compaction of an existing skill_map.json.

Usage:
    env PYTHONPATH=. python eval/scripts/compact_skill_map.py \
        --run-id dayrun_17810316 \
        --profile kimi_k25_livecodebench_v1
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from skillmap.induction.consolidator import SkillConsolidator
from skillmap.llm.client import LLMConfig, configure_default_client
from skillmap.storage import JSONPersistence, SkillMap
from skillmap.types import SkillMapState


async def _run(args: argparse.Namespace) -> None:
    skill_map_path = (
        ROOT / "eval" / "data" / "skillmap_day_runs"
        / args.profile / args.run_id / "skill_map.json"
    )
    if not skill_map_path.exists():
        print(f"skill_map.json not found: {skill_map_path}", file=sys.stderr)
        sys.exit(1)

    configure_default_client(
        LLMConfig(
            provider="bedrock",
            model=args.model,
            region=args.region,
            extra_inference={"temperature": 0.0},
        )
    )

    persistence = JSONPersistence(skill_map_path, user_id=args.run_id)
    state = persistence.load()
    skill_map = SkillMap(state, persistence)

    skills_before = skill_map.list_skills()
    print(f"Skills before compaction: {len(skills_before)}")
    for s in skills_before:
        print(f"  [{s.id[:8]}] {s.title} (support={s.support_count})")

    consolidator = SkillConsolidator(llm_model=args.model, region=args.region)
    await consolidator.compact(skill_map)

    skills_after = skill_map.list_skills()
    print(f"\nSkills after compaction: {len(skills_after)}")
    for s in skills_after:
        print(f"  [{s.id[:8]}] {s.title} (support={s.support_count})")
    print(f"\nReduced {len(skills_before)} → {len(skills_after)} skills")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--profile", default="kimi_k25_livecodebench_v1")
    parser.add_argument("--model", default="us.anthropic.claude-haiku-4-5-20251001-v1:0")
    parser.add_argument("--region", default="us-east-1")
    args = parser.parse_args()
    asyncio.run(_run(args))


if __name__ == "__main__":
    main()
