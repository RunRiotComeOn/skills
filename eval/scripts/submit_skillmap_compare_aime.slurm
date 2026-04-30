#!/bin/bash -l
#SBATCH -J skillmap-compare-aime
#SBATCH -A bgek-delta-cpu
#SBATCH -p cpu
#SBATCH -N 1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH -t 08:00:00
#SBATCH -o /u/yhuang48/SkillMap/eval/results/slurm/%x-%j.out

# AIME comparison: 3 days x 20 tasks/day, stateless vs skillmap.
# Pure Bedrock API calls — no GPU needed.

set -euo pipefail

cd /u/yhuang48/SkillMap
mkdir -p eval/results/slurm

PROFILE_ID="${PROFILE_ID:-haiku_minimax_aime_v1}"
SESSION_ID="${SESSION_ID_OVERRIDE:-compare_aime_${SLURM_JOB_ID}}"
CONDITIONS="${CONDITIONS:-stateless,skillmap}"

env PYTHONPATH=. PYTHONUNBUFFERED=1 python -u eval/scripts/run_skillmap_days.py \
  --profile "${PROFILE_ID}" \
  --config eval/eval_config_aime.yaml \
  --days 3 \
  --tasks-per-day 20 \
  --session-id "${SESSION_ID}" \
  --conditions "${CONDITIONS}"
