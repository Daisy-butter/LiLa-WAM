#!/usr/bin/env bash
# 8-GPU DDP training for LiLa-Dynamic (DOMINO + dual-path motion).
# All experiment outputs go under /SSD_DISK/users/wuruihan/.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
# Prefer launching from the git repo if this script is a copy on SSD.
if [[ -f /home/wuruihan/LiLa-WAM/train.py ]]; then
  REPO_ROOT=/home/wuruihan/LiLa-WAM
fi
cd "$REPO_ROOT"

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HOME="${HF_HOME:-/SSD_DISK/users/wuruihan/hf_cache}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"

PY="${PYTHON:-/SSD_DISK/users/wuruihan/conda_envs/lila/bin/python}"
SAVE_DIR="${SAVE_DIR:-/SSD_DISK/users/wuruihan/checkpoints_lila_dynamic}"
LOG_DIR="${LOG_DIR:-/SSD_DISK/users/wuruihan/logs_lila_dynamic}"
NORM_STATS="${NORM_STATS:-/SSD_DISK/users/wuruihan/lila_wam/data/norm_stats/stat-domino.json}"
CONFIG="${CONFIG:-${REPO_ROOT}/configs/lila_dynamic.yaml}"
mkdir -p "$SAVE_DIR" "$LOG_DIR"

TIMESTAMP="$(date +%Y-%m-%d_%H-%M-%S)"
LOG_FILE="${LOG_DIR}/train_8gpu_${TIMESTAMP}.log"
echo "$LOG_FILE" > "${LOG_DIR}/latest_log_path.txt"

echo "Logging to ${LOG_FILE}"
echo "per_gpu_batch from config (default 8) → global_batch = 8 * 8 = 64"

nohup "$PY" -m torch.distributed.run \
  --standalone \
  --nproc_per_node=8 \
  train.py \
  --config "$CONFIG" \
  --norm_stats_path "$NORM_STATS" \
  --save_dir "$SAVE_DIR" \
  "$@" \
  > "$LOG_FILE" 2>&1 &

echo "TRAIN_PID=$!"
echo "tail -f $LOG_FILE"
