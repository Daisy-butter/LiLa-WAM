#!/usr/bin/env bash
# Resume-aware watchdog for the current LiLa-WAM two-stage run.
# - Does NOT restart stage1 from scratch
# - If DDP dies: --resume latest epoch ckpt
# - After stage1 epoch 12: --init_from into stage2 (4 epochs)
# - After stage2: launch eval (skipped if RoboTwin is missing)
set -u

REPO_ROOT=/home/wuruihan/LiLa-WAM
cd "$REPO_ROOT"

PY="${PYTHON:-/SSD_DISK/users/wuruihan/conda_envs/lila/bin/python}"
SAVE_ROOT="${SAVE_ROOT:-/SSD_DISK/users/wuruihan/checkpoints_lila_dynamic}"
LOG_DIR="${LOG_DIR:-/SSD_DISK/users/wuruihan/logs_lila_dynamic}"
NORM_STATS="${NORM_STATS:-/SSD_DISK/users/wuruihan/lila_wam/data/norm_stats/stat-domino.json}"
STAGE1_CFG="${REPO_ROOT}/configs/lila_dynamic.yaml"
STAGE2_CFG="${REPO_ROOT}/configs/lila_dynamic_stage2.yaml"
STAGE1_EPOCHS=12
STAGE2_EPOCHS=4
STUCK_SEC="${STUCK_SEC:-600}"
MAX_RESTARTS="${MAX_RESTARTS:-8}"
NPROC=8

# Pin to the live DOMINO-VTT run unless overridden.
RUN_TS="${RUN_TS:-2026-09-14_11-06-59}"
STAGE1_SAVE="${STAGE1_SAVE:-${SAVE_ROOT}/stage1_${RUN_TS}}"
STAGE2_SAVE="${STAGE2_SAVE:-${SAVE_ROOT}/stage2_${RUN_TS}}"
STAGE1_LOG="${STAGE1_LOG:-${LOG_DIR}/stage1_8gpu_${RUN_TS}.log}"
STAGE2_LOG="${STAGE2_LOG:-${LOG_DIR}/stage2_8gpu_${RUN_TS}.log}"
WATCH_LOG="${LOG_DIR}/watchdog_${RUN_TS}.log"
STATE_FILE="${LOG_DIR}/watchdog_state_${RUN_TS}.txt"
DONE_FILE="${LOG_DIR}/pipeline_done_${RUN_TS}.flag"
LOCK_FILE="${LOG_DIR}/watchdog.lock"

mkdir -p "$LOG_DIR" "$SAVE_ROOT" "$STAGE1_SAVE" "$STAGE2_SAVE"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HOME="${HF_HOME:-/SSD_DISK/users/wuruihan/hf_cache}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"

log() { echo "[$(date '+%F %T')] $*" | tee -a "$WATCH_LOG"; }

latest_ckpt() {
  local dir="$1" epoch="${2:-}"
  if [[ -n "${epoch}" ]]; then
    find "$dir" -name "checkpoint_epoch_${epoch}.pt" 2>/dev/null | sort | tail -1
    return
  fi
  python3 - "$dir" <<'PY'
import glob, os, re, sys
root = sys.argv[1]
best, best_n = "", -1
for p in glob.glob(os.path.join(root, "**", "checkpoint_epoch_*.pt"), recursive=True):
    m = re.search(r"checkpoint_epoch_(\d+)\.pt$", p)
    if m and int(m.group(1)) >= best_n:
        best_n = int(m.group(1))
        best = p
print(best)
PY
}

ddp_pids() {
  pgrep -f "torch.distributed.run .*train.py" 2>/dev/null || true
}

wrapper_pids() {
  pgrep -f "train_lila_dynamic_two_stage_8gpu.sh" 2>/dev/null || true
}

eval_pids() {
  pgrep -f "eval_lila_dynamic.sh|eval_vla_bridge.py" 2>/dev/null || true
}

training_alive() {
  local p; p="$(ddp_pids)"
  [[ -n "$p" ]]
}

log_is_fresh() {
  local f="$1"
  [[ -f "$f" ]] || return 1
  local now mtime
  now=$(date +%s)
  mtime=$(stat -c %Y "$f")
  (( now - mtime < STUCK_SEC ))
}

has_fatal() {
  local f="$1"
  [[ -f "$f" ]] || return 1
  grep -E "Traceback \(most recent call last\)|CUDA out of memory|NCCL error|SignalException|NaN|nan loss" "$f" >/dev/null 2>&1
}

restart_count() {
  if [[ -f "$STATE_FILE" ]]; then
    awk -F= '/^RESTARTS=/{print $2}' "$STATE_FILE" | tail -1
  else
    echo 0
  fi
}

bump_restarts() {
  local n; n=$(restart_count)
  n=$((n + 1))
  {
    echo "RESTARTS=$n"
    echo "LAST=$(date '+%F %T')"
  } > "$STATE_FILE"
  echo "$n"
}

kill_training() {
  log "Killing DDP / train.py (keep watchdog)"
  pkill -f "torch.distributed.run .*train.py" 2>/dev/null || true
  pkill -f "/train.py --config" 2>/dev/null || true
  sleep 5
  pkill -9 -f "torch.distributed.run .*train.py" 2>/dev/null || true
  sleep 2
}

launch_ddp() {
  local cfg="$1" save="$2" logfile="$3"
  shift 3
  log "Launch DDP cfg=$cfg save=$save extra=$*"
  echo "$logfile" > "${LOG_DIR}/latest_log_path.txt"
  nohup "$PY" -m torch.distributed.run --standalone --nproc_per_node="$NPROC" \
    train.py \
    --config "$cfg" \
    --norm_stats_path "$NORM_STATS" \
    --save_dir "$save" \
    "$@" \
    >> "$logfile" 2>&1 &
  echo $! > "${LOG_DIR}/ddp_${RUN_TS}.pid"
  sleep 8
}

start_stage1_resume() {
  local ckpt; ckpt="$(latest_ckpt "$STAGE1_SAVE")"
  if [[ -z "$ckpt" || ! -f "$ckpt" ]]; then
    log "ERROR: stage1 died with no checkpoint; refusing to restart from scratch"
    return 1
  fi
  log "Resume stage1 from $ckpt"
  launch_ddp "$STAGE1_CFG" "$STAGE1_SAVE" "$STAGE1_LOG" --resume "$ckpt"
}

start_stage2() {
  local ckpt; ckpt="$(latest_ckpt "$STAGE1_SAVE" "$STAGE1_EPOCHS")"
  if [[ -z "$ckpt" || ! -f "$ckpt" ]]; then
    log "ERROR: stage1 epoch ${STAGE1_EPOCHS} ckpt missing"
    return 1
  fi
  mkdir -p "$STAGE2_SAVE"
  : > "$STAGE2_LOG"
  log "Start stage2 init_from=$ckpt"
  launch_ddp "$STAGE2_CFG" "$STAGE2_SAVE" "$STAGE2_LOG" --init_from "$ckpt"
}

start_stage2_resume() {
  local ckpt; ckpt="$(latest_ckpt "$STAGE2_SAVE")"
  if [[ -z "$ckpt" || ! -f "$ckpt" ]]; then
    log "Stage2 has no ckpt; restart stage2 from stage1"
    start_stage2
    return
  fi
  log "Resume stage2 from $ckpt"
  launch_ddp "$STAGE2_CFG" "$STAGE2_SAVE" "$STAGE2_LOG" --resume "$ckpt"
}

stage1_done() {
  local ckpt; ckpt="$(latest_ckpt "$STAGE1_SAVE" "$STAGE1_EPOCHS")"
  [[ -n "$ckpt" && -f "$ckpt" ]]
}

stage2_done() {
  local ckpt; ckpt="$(latest_ckpt "$STAGE2_SAVE" "$STAGE2_EPOCHS")"
  [[ -n "$ckpt" && -f "$ckpt" ]]
}

start_eval() {
  if [[ -f "$DONE_FILE" ]]; then
    return 0
  fi
  if [[ -n "$(eval_pids)" ]]; then
    log "Eval already running"
    return 0
  fi
  log "Launch eval_lila_dynamic.sh"
  nohup bash "${REPO_ROOT}/scripts/eval_lila_dynamic.sh" \
    >> "${LOG_DIR}/eval_${RUN_TS}.log" 2>&1 &
  echo $! > "${LOG_DIR}/eval_${RUN_TS}.pid"
  echo "eval_launched $(date '+%F %T')" > "$DONE_FILE"
}

tick() {
  if [[ -f "$DONE_FILE" ]] && grep -q "eval_launched\|eval_skipped" "$DONE_FILE"; then
    if [[ -z "$(eval_pids)" ]] && grep -q "eval_launched" "$DONE_FILE"; then
      if grep -q "EVAL_COMPLETE\|EVAL_SKIPPED" "${LOG_DIR}/eval_${RUN_TS}.log" 2>/dev/null; then
        echo "pipeline_complete $(date '+%F %T')" > "$DONE_FILE"
        log "Pipeline complete"
      fi
    fi
    return 0
  fi

  local n; n=$(restart_count)
  if (( n >= MAX_RESTARTS )); then
    log "ERROR: hit MAX_RESTARTS=$MAX_RESTARTS; stop auto-resume"
    return 1
  fi

  if training_alive; then
    local logf="$STAGE1_LOG"
    if stage1_done && [[ -f "$STAGE2_LOG" ]]; then
      logf="$STAGE2_LOG"
    fi
    if ! log_is_fresh "$logf"; then
      log "WARN: training alive but log stale >${STUCK_SEC}s; restart from last ckpt"
      bump_restarts >/dev/null
      kill_training
      sleep 3
      if stage1_done && ! stage2_done; then
        start_stage2_resume
      else
        start_stage1_resume
      fi
      return 0
    fi
    log "healthy: ddp alive log=$(basename "$logf")"
    return 0
  fi

  # DDP not running
  if stage2_done; then
    log "Stage2 complete; starting eval"
    start_eval
    return 0
  fi
  if stage1_done; then
    # If original wrapper is still about to start stage2, wait briefly
    if [[ -n "$(wrapper_pids)" ]]; then
      log "Stage1 done, wrapper still alive; wait for it to start stage2"
      sleep 20
      training_alive && return 0
      log "Wrapper did not start stage2; watchdog launching stage2"
    fi
    bump_restarts >/dev/null
    start_stage2
    return 0
  fi

  log "Stage1 DDP dead before epoch ${STAGE1_EPOCHS}; resume"
  bump_restarts >/dev/null
  start_stage1_resume
}

# One-shot mode for the agent loop
if [[ "${1:-}" == "--once" ]]; then
  exec 9>"$LOCK_FILE"
  if ! flock -n 9; then
    echo "watchdog already running"
    exit 0
  fi
  tick
  exit 0
fi

exec 9>"$LOCK_FILE"
if ! flock -n 9; then
  echo "another watchdog holds $LOCK_FILE"
  exit 0
fi

log "Watchdog start RUN_TS=$RUN_TS stage1=$STAGE1_SAVE"
while true; do
  tick || true
  if [[ -f "$DONE_FILE" ]] && grep -q "pipeline_complete" "$DONE_FILE"; then
    log "Watchdog exiting: pipeline complete"
    break
  fi
  sleep 60
done
