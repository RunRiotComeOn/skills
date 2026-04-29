#!/bin/bash -l
#SBATCH -J skillmap-5day
#SBATCH -A bgek-delta-cpu
#SBATCH -p cpu
#SBATCH -N 1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH -t 36:00:00
#SBATCH -o /u/yhuang48/SkillMap/eval/results/slurm/%x-%j.out

set -euo pipefail

cd /u/yhuang48/SkillMap
mkdir -p eval/results/slurm

PROFILE_ID="haiku_minimax_livecodebench_v1"
SESSION_ID="${SESSION_ID_OVERRIDE:-dayrun_${SLURM_JOB_ID}}"
# Comma-separated. Each named condition runs over the SAME task stream so
# the resulting curves are directly comparable. Override via env to run a
# subset, e.g. CONDITIONS=skillmap.
CONDITIONS="${CONDITIONS:-stateless,declarative_memory,skillmap}"

env PYTHONPATH=. PYTHONUNBUFFERED=1 python -u eval/scripts/run_skillmap_days.py \
  --profile "${PROFILE_ID}" \
  --days 5 \
  --tasks-per-day 50 \
  --session-id "${SESSION_ID}" \
  --conditions "${CONDITIONS}"
