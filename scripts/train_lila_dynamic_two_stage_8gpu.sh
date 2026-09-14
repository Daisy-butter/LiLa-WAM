#!/usr/bin/env bash
# LiLa two-stage recipe on 8 GPUs:
#   Stage-1: LR=2e-4, 12 epochs
#   Stage-2: LR=4e-5, 4 epochs, --init_from stage1 last ckpt (NOT --resume)
set -euo pipefail

REPO_ROOT=/home/wuruihan/LiLa-WAM
cd "$REPO_ROOT"

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HOME="${HF_HOME:-/SSD_DISK/users/wuruihan/hf_cache}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"

PY="${PYTHON:-/SSD_DISK/users/wuruihan/conda_envs/lila/bin/python}"
SAVE_ROOT="${SAVE_ROOT:-/SSD_DISK/users/wuruihan/checkpoints_lila_dynamic}"
LOG_DIR="${LOG_DIR:-/SSD_DISK/users/wuruihan/logs_lila_dynamic}"
NORM_STATS="${NORM_STATS:-/SSD_DISK/users/wuruihan/lila_wam/data/norm_stats/stat-domino.json}"
STAGE1_CFG="${STAGE1_CFG:-${REPO_ROOT}/configs/lila_dynamic.yaml}"
STAGE2_CFG="${STAGE2_CFG:-${REPO_ROOT}/configs/lila_dynamic_stage2.yaml}"
STAGE1_EPOCHS="${STAGE1_EPOCHS:-12}"

mkdir -p "$SAVE_ROOT" "$LOG_DIR"
TIMESTAMP="$(date +%Y-%m-%d_%H-%M-%S)"
STAGE1_SAVE="${SAVE_ROOT}/stage1_${TIMESTAMP}"
STAGE2_SAVE="${SAVE_ROOT}/stage2_${TIMESTAMP}"
LOG1="${LOG_DIR}/stage1_8gpu_${TIMESTAMP}.log"
LOG2="${LOG_DIR}/stage2_8gpu_${TIMESTAMP}.log"
echo "$LOG1" > "${LOG_DIR}/latest_log_path.txt"

echo "========== STAGE 1 (LR=2e-4, ${STAGE1_EPOCHS} epochs) =========="
echo "log: $LOG1"
echo "save: $STAGE1_SAVE"

"$PY" -m torch.distributed.run --standalone --nproc_per_node=8 \
  train.py \
  --config "$STAGE1_CFG" \
  --norm_stats_path "$NORM_STATS" \
  --save_dir "$STAGE1_SAVE" \
  2>&1 | tee "$LOG1"

# train.py creates STAGE1_SAVE/sft_<ts>/checkpoint_epoch_N.pt
CKPT="$(find "$STAGE1_SAVE" -name "checkpoint_epoch_${STAGE1_EPOCHS}.pt" | sort | tail -1)"
if [[ -z "$CKPT" || ! -f "$CKPT" ]]; then
  echo "ERROR: stage1 checkpoint_epoch_${STAGE1_EPOCHS}.pt not found under $STAGE1_SAVE"
  find "$STAGE1_SAVE" -name 'checkpoint_epoch_*.pt' | sort
  exit 1
fi

echo "========== STAGE 2 (LR=4e-5, 4 epochs, init_from=$CKPT) =========="
echo "log: $LOG2"
echo "save: $STAGE2_SAVE"
echo "$LOG2" > "${LOG_DIR}/latest_log_path.txt"

"$PY" -m torch.distributed.run --standalone --nproc_per_node=8 \
  train.py \
  --config "$STAGE2_CFG" \
  --norm_stats_path "$NORM_STATS" \
  --save_dir "$STAGE2_SAVE" \
  --init_from "$CKPT" \
  2>&1 | tee "$LOG2"

echo "DONE. Stage1 ckpt: $CKPT"
echo "Stage2 dir: $STAGE2_SAVE"
