#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

DATASET=${DATASET:-Beauty}
DATA_PATH=${DATA_PATH:-${PROJECT_ROOT}/data/tiger_data}
CHECKPOINT=${CHECKPOINT:-${SCRIPT_DIR}/checkpoints/${DATASET}/best.pt}
OUTPUT_FILE=${OUTPUT_FILE:-${SCRIPT_DIR}/results/${DATASET}/predictions.pkl}
LOG_DIR=${LOG_DIR:-${SCRIPT_DIR}/logs}
LOG_FILE=${LOG_DIR}/test_${DATASET}_$(date +%Y%m%d_%H%M%S).log

mkdir -p "${LOG_DIR}" "${SCRIPT_DIR}/results/${DATASET}"

INFER_BATCH_SIZE=${INFER_BATCH_SIZE:-512}
NUM_WORKERS=${NUM_WORKERS:-4}
MAX_HIST_LEN=${MAX_HIST_LEN:-800}

python3 -u "${SCRIPT_DIR}/main.py" \
    --mode infer \
    --data_format json \
    --dataset "${DATASET}" \
    --data_path "${DATA_PATH}" \
    --checkpoint "${CHECKPOINT}" \
    --output_file "${OUTPUT_FILE}" \
    --infer_batch_size "${INFER_BATCH_SIZE}" \
    --num_workers "${NUM_WORKERS}" \
    --max_hist_len "${MAX_HIST_LEN}" \
    --target_type_num 1 2>&1 | tee "${LOG_FILE}"

echo "Done. output=${OUTPUT_FILE}"
