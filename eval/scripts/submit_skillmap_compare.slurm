#!/bin/bash -l
#SBATCH -J skillmap-compare
#SBATCH -A bgek-delta-gpu
#SBATCH -p gpuA40x4
#SBATCH -N 1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH -t 08:00:00
#SBATCH -o /u/yhuang48/SkillMap/eval/results/slurm/%x-%j.out

# Short three-condition comparison: 3 days x 20 tasks/day. The script makes
# only Bedrock API calls (no local model inference), so the GPU sits idle —
# we land on this partition because the CPU partition is unavailable.

set -euo pipefail

cd /u/yhuang48/SkillMap
mkdir -p eval/results/slurm

PROFILE_ID="${PROFILE_ID:-haiku_minimax_livecodebench_v1}"
SESSION_ID="${SESSION_ID_OVERRIDE:-compare_${SLURM_JOB_ID}}"
CONDITIONS="${CONDITIONS:-stateless,declarative_memory,skillmap}"

env PYTHONPATH=. PYTHONUNBUFFERED=1 python -u eval/scripts/run_skillmap_days.py \
  --profile "${PROFILE_ID}" \
  --days 3 \
  --tasks-per-day 20 \
  --session-id "${SESSION_ID}" \
  --conditions "${CONDITIONS}"
