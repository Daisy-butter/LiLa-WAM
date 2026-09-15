#!/usr/bin/env bash
# Evaluate the finished LiLa-WAM + DynamicWAM stage2 checkpoint on DOMINO tasks.
# Requires a patched RoboTwin/DOMINO env with TASK_ENV._scene_step_clock.
set -euo pipefail

REPO_ROOT=/home/wuruihan/LiLa-WAM
RUN_TS="${RUN_TS:-2026-09-14_11-06-59}"
SAVE_ROOT="${SAVE_ROOT:-/SSD_DISK/users/wuruihan/checkpoints_lila_dynamic}"
LOG_DIR="${LOG_DIR:-/SSD_DISK/users/wuruihan/logs_lila_dynamic}"
NORM_STATS="${NORM_STATS:-/SSD_DISK/users/wuruihan/lila_wam/data/norm_stats/stat-domino.json}"
STAGE2_SAVE="${STAGE2_SAVE:-${SAVE_ROOT}/stage2_${RUN_TS}}"
CHECKPOINT_EP="${CHECKPOINT_EP:-4}"
TEST_NUM="${TEST_NUM:-20}"
TASK_CONFIG="${TASK_CONFIG:-demo_clean}"
INSTRUCTION_TYPE="${INSTRUCTION_TYPE:-unseen}"
SEED="${SEED:-0}"
NUM_WORKERS="${NUM_WORKERS:-2}"

CKPT="$(find "$STAGE2_SAVE" -name "checkpoint_epoch_${CHECKPOINT_EP}.pt" 2>/dev/null | sort | tail -1 || true)"
if [[ -z "${CKPT}" || ! -f "${CKPT}" ]]; then
  echo "EVAL_SKIPPED: stage2 checkpoint_epoch_${CHECKPOINT_EP}.pt not found under $STAGE2_SAVE"
  exit 1
fi

ROBOTWIN_ROOT="${ROBOTWIN_ROOT:-}"
if [[ -z "$ROBOTWIN_ROOT" ]]; then
  for cand in \
      /home/wuruihan/RoboTwin \
      /SSD_DISK/users/wuruihan/RoboTwin \
      /SSD_DISK/users/wuruihan/wam/RoboTwin \
      /home/yf/Desktop/Code/VLA/RoboTwin/RoboTwin; do
    if [[ -d "$cand/envs" ]]; then
      ROBOTWIN_ROOT="$cand"
      break
    fi
  done
fi

ROBOTWIN_PY="${ROBOTWIN_PYTHON:-}"
if [[ -z "$ROBOTWIN_PY" ]]; then
  for cand in \
      /SSD_DISK/users/wuruihan/conda_envs/RoboTwin/bin/python \
      /SSD_DISK/users/wuruihan/conda_envs/robotwin/bin/python \
      /home/wuruihan/miniconda3/envs/RoboTwin/bin/python; do
    if [[ -x "$cand" ]]; then
      ROBOTWIN_PY="$cand"
      break
    fi
  done
fi

if [[ -z "${ROBOTWIN_ROOT:-}" || ! -d "${ROBOTWIN_ROOT}" ]]; then
  echo "EVAL_SKIPPED: RoboTwin root not found."
  echo "Set ROBOTWIN_ROOT to the patched DOMINO/RoboTwin tree (must have _scene_step_clock),"
  echo "copy ${REPO_ROOT}/eval_vla_bridge.py into that root, then rerun:"
  echo "  ROBOTWIN_ROOT=... ROBOTWIN_PYTHON=... bash $0"
  echo "Checkpoint ready: $CKPT"
  exit 0
fi

if [[ -z "${ROBOTWIN_PY:-}" ]]; then
  echo "EVAL_SKIPPED: RoboTwin python not found. Set ROBOTWIN_PYTHON."
  echo "Checkpoint ready: $CKPT"
  exit 0
fi

cp -f "${REPO_ROOT}/eval_vla_bridge.py" "${ROBOTWIN_ROOT}/eval_vla_bridge.py"
echo "Copied eval_vla_bridge.py -> ${ROBOTWIN_ROOT}/eval_vla_bridge.py"

TASKS=(
  adjust_bottle beat_block_hammer click_alarmclock click_bell dump_bin_bigbin
  grab_roller handover_block handover_mic hanging_mug move_can_pot
  move_pillbottle_pad move_playingcard_away move_stapler_pad place_a2b_left
  place_a2b_right place_bread_basket place_bread_skillet place_can_basket
  place_container_plate place_empty_cup place_fan place_mouse_pad
  place_object_basket place_object_scale place_object_stand place_phone_stand
  place_shoe press_stapler put_bottles_dustbin put_object_cabinet rotate_qrcode
  scan_object shake_bottle shake_bottle_horizontally stamp_seal
)

SAVE_EVAL="${LOG_DIR}/eval_result_${RUN_TS}"
LOGW="${SAVE_EVAL}/logs"
RESULT_DIR="${SAVE_EVAL}/bridge_results"
mkdir -p "$LOGW" "$RESULT_DIR"
rm -f "$RESULT_DIR"/worker_*.json

echo "========== EVAL =========="
echo "ckpt: $CKPT"
echo "robotwin: $ROBOTWIN_ROOT"
echo "python: $ROBOTWIN_PY"
echo "tasks: ${#TASKS[@]}  test_num=$TEST_NUM  config=$TASK_CONFIG"

pids=()
worker_ids=()
for ((w = 0; w < NUM_WORKERS; w++)); do
  chunk=()
  for ((i = w; i < ${#TASKS[@]}; i += NUM_WORKERS)); do
    chunk+=("${TASKS[i]}")
  done
  [[ ${#chunk[@]} -eq 0 ]] && continue
  echo "[worker $w] ${chunk[*]}"
  (
    cd "$ROBOTWIN_ROOT" || exit 1
    "$ROBOTWIN_PY" "${ROBOTWIN_ROOT}/eval_vla_bridge.py" \
      --task_names "${chunk[@]}" \
      --task_config "$TASK_CONFIG" \
      --instruction_type "$INSTRUCTION_TYPE" \
      --ckpt_setting "lila_dynamic_stage2_${RUN_TS}" \
      --checkpoint_ep "$CHECKPOINT_EP" \
      --checkpoint_path "$CKPT" \
      --model_base_path "$REPO_ROOT" \
      --norm_stats_path "$NORM_STATS" \
      --config_path "${REPO_ROOT}/configs/lila_dynamic_stage2.yaml" \
      --vla_root "$REPO_ROOT" \
      --save_root "$SAVE_EVAL" \
      --result_json "$RESULT_DIR/worker_$w.json" \
      --seed "$SEED" \
      --test_num "$TEST_NUM"
  ) > "$LOGW/worker_$w.log" 2>&1 &
  pids+=($!)
  worker_ids+=($w)
done

fail=0
for idx in "${!pids[@]}"; do
  if ! wait "${pids[$idx]}"; then
    echo "[worker ${worker_ids[$idx]}] failed, see $LOGW/worker_${worker_ids[$idx]}.log"
    fail=1
  fi
done

echo "EVAL_COMPLETE fail=$fail ckpt=$CKPT"
exit $fail
