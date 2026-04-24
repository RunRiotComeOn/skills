#!/bin/bash -l
#SBATCH -J skillmap-3day
#SBATCH -A bgek-delta-cpu
#SBATCH -p cpu
#SBATCH -N 1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH -t 12:00:00
#SBATCH -o /u/yhuang48/SkillMap/eval/results/slurm/%x-%j.out

set -euo pipefail

cd /u/yhuang48/SkillMap
mkdir -p eval/results/slurm

PROFILE_ID="kimi_k25_livecodebench_v1"
SESSION_ID="${SESSION_ID_OVERRIDE:-dayrun_${SLURM_JOB_ID}}"

env PYTHONPATH=. PYTHONUNBUFFERED=1 python -u eval/scripts/run_skillmap_days.py \
  --profile "${PROFILE_ID}" \
  --days 3 \
  --tasks-per-day 20 \
  --session-id "${SESSION_ID}"
