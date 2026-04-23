"""Generate the 4 eval figures from an existing EvalReport JSON.

Usage:
    python scripts/make_figures.py --report results/eval_report_profile_coding_v1.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from skillmap_eval.analysis.plot import make_all_figures
from skillmap_eval.types import EvalReport


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", required=True)
    parser.add_argument("--out", default=None, help="output dir (default: alongside report)")
    args = parser.parse_args()

    report_path = Path(args.report)
    report = EvalReport.model_validate_json(report_path.read_text(encoding="utf-8"))
    out_dir = Path(args.out) if args.out else report_path.parent / f"figures_{report.profile_id}"
    paths = make_all_figures(report, out_dir)
    print("wrote figures:")
    for k, p in paths.items():
        print(f"  {k}: {p}")


if __name__ == "__main__":
    main()
