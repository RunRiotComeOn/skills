"""Main entry point for the full eval.

Usage:
    python scripts/run_eval.py --profile profile_coding_v1

Phase 8 gate (spec §7): produces results/eval_report_<profile_id>.json with
all four metric sections populated, plus four PNG figures via make_figures.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from skillmap_eval.orchestrator import EvalOrchestrator, load_eval_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", required=True, help="profile_id (filename without .json)")
    parser.add_argument("--config", default=str(ROOT / "eval_config.yaml"))
    args = parser.parse_args()

    cfg = load_eval_config(args.config)
    orch = EvalOrchestrator(cfg)
    report = asyncio.run(orch.run_full_eval(profile_id=args.profile))
    print(f"eval complete. report written to {cfg.results_dir}/eval_report_{report.profile_id}.json")
    print(
        f"  - {len(report.correction_curves)} correction curves, "
        f"{len(report.preference_recovery)} recovery results, "
        f"{len(report.generalization)} generalization results"
    )


if __name__ == "__main__":
    main()
