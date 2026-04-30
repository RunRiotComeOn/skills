#!/bin/bash -l
#SBATCH -J skillmap-compare-aime-v2
#SBATCH -A bgek-delta-gpu
#SBATCH -p gpuA40x4
#SBATCH -N 1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH -t 10:00:00
#SBATCH -o /u/yhuang48/SkillMap/eval/results/slurm/%x-%j.out

# AIME v2 comparison: 5-preference profile, top-3 skill retrieval,
# longer turn budget (12) and higher give-up threshold (5).
# 3 days x 20 tasks/day, stateless vs skillmap. Bedrock API calls only.

set -euo pipefail

cd /u/yhuang48/SkillMap
mkdir -p eval/results/slurm

PROFILE_ID="${PROFILE_ID:-haiku_minimax_aime_v2}"
SESSION_ID="${SESSION_ID_OVERRIDE:-compare_aime_v2_${SLURM_JOB_ID}}"
CONDITIONS="${CONDITIONS:-stateless,skillmap}"

env PYTHONPATH=. PYTHONUNBUFFERED=1 SKILLMAP_TOP_K=3 python -u eval/scripts/run_skillmap_days.py \
  --profile "${PROFILE_ID}" \
  --config eval/eval_config_aime_v2.yaml \
  --days 3 \
  --tasks-per-day 20 \
  --session-id "${SESSION_ID}" \
  --conditions "${CONDITIONS}"
