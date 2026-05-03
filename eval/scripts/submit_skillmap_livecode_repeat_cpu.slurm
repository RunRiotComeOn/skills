#!/bin/bash -l
#SBATCH -J skillmap-livecode-repeat
#SBATCH -A bgek-delta-gpu
#SBATCH -p gpuA100x4
#SBATCH -N 1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH -t 12:00:00
#SBATCH -o /u/yhuang48/SkillMap/eval/results/slurm/%x-%j.out

set -euo pipefail

cd /u/yhuang48/SkillMap
mkdir -p eval/results/slurm

PROFILE_ID="${PROFILE_ID:-haiku_minimax_livecodebench_v1}"
SESSION_ID="${SESSION_ID_OVERRIDE:-livecode_repeat_${SLURM_JOB_ID}}"
CONDITIONS="${CONDITIONS:-stateless,skillmap}"

env PYTHONPATH=. PYTHONUNBUFFERED=1 SKILLMAP_TOP_K=3 python -u eval/scripts/run_skillmap_days.py \
  --profile "${PROFILE_ID}" \
  --config eval/eval_config_livecode.yaml \
  --days 3 \
  --tasks-per-day 50 \
  --repeat-tasks \
  --session-id "${SESSION_ID}" \
  --conditions "${CONDITIONS}"
