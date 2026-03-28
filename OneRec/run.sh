#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}

# Dataset to train (shares TIGER-prepared data)
DATASET=${DATASET:-Beauty}

# Use TIGER data folder by default
DATA_PATH=${DATA_PATH:-${PROJECT_ROOT}/data/tiger_data}

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
SAVE_DIR=${SAVE_DIR:-${SCRIPT_DIR}/checkpoints/${DATASET}/${TIMESTAMP}/}
LOG_DIR=${LOG_DIR:-${SCRIPT_DIR}/logs}
LOG_FILE=${LOG_DIR}/train_${DATASET}_${TIMESTAMP}.log

mkdir -p "${LOG_DIR}" "${SAVE_DIR}"

echo "=================================="
echo "Start OneRecV2 Training - ${DATASET}"
echo "=================================="
echo "DATA_PATH: ${DATA_PATH}"
echo "SAVE_DIR : ${SAVE_DIR}"
echo "LOG_FILE : ${LOG_FILE}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo ""

# Hyperparameters (aligned with TIGER/train_8gpu.sh)
BATCH_SIZE=${BATCH_SIZE:-256}
VAL_BATCH_SIZE=${VAL_BATCH_SIZE:-512}
NUM_WORKERS=${NUM_WORKERS:-4}
MAX_HIST_LEN=${MAX_HIST_LEN:-50}
DROPOUT=${DROPOUT:-0.1}
NUM_EPOCHS=${NUM_EPOCHS:-30}
LEARNING_RATE=${LEARNING_RATE:-5e-4}
EVAL_INTERVAL=${EVAL_INTERVAL:-2}
EVAL_START_EPOCH=${EVAL_START_EPOCH:-12}
BEAM_SIZE=${BEAM_SIZE:-20}
TOPK_LIST=${TOPK_LIST:-"5 10"}
EARLY_STOP=${EARLY_STOP:-2}
SEED=${SEED:-2025}

nohup python3 -u "${SCRIPT_DIR}/main.py" \
    --mode train \
    --data_format json \
    --dataset "${DATASET}" \
    --data_path "${DATA_PATH}" \
    --save_dir "${SAVE_DIR}" \
    --batch_size "${BATCH_SIZE}" \
    --val_batch_size "${VAL_BATCH_SIZE}" \
    --num_workers "${NUM_WORKERS}" \
    --max_hist_len "${MAX_HIST_LEN}" \
    --target_type_num 1 \
    --dropout "${DROPOUT}" \
    --num_epochs "${NUM_EPOCHS}" \
    --learning_rate "${LEARNING_RATE}" \
    --eval_interval "${EVAL_INTERVAL}" \
    --eval_start_epoch "${EVAL_START_EPOCH}" \
    --beam_size "${BEAM_SIZE}" \
    --early_stop "${EARLY_STOP}" \
    --topk_list ${TOPK_LIST} \
    --seed "${SEED}" \
    > "${LOG_FILE}" 2>&1 &

TRAIN_PID=$!
echo "Training PID: ${TRAIN_PID}"
echo "${TRAIN_PID}" > "${SCRIPT_DIR}/train.pid"
echo "Watch log: tail -f ${LOG_FILE}"
