#!/bin/bash
set -euo pipefail

# 8-GPU DDP training launcher for CEMG/TIGER
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

BASE_MODEL=${BASE_MODEL:-/llm-reco-ssd-share/zhangrongzhou/Tiger/pretrained_models/t5_config}
LOAD_PRETRAINED=${LOAD_PRETRAINED:-false}

DATASET=${DATASET:-Beauty}
DATA_PATH=${DATA_PATH:-${PROJECT_ROOT}/data/tiger_data}
INTER_FILE=${INTER_FILE:-${DATASET}.inter.json}
INDEX_FILE=${INDEX_FILE:-merge/merge.index.json}

NUM_GPUS=${NUM_GPUS:-8}

NUM_LAYERS=${NUM_LAYERS:-4}
NUM_DECODER_LAYERS=${NUM_DECODER_LAYERS:-4}
D_MODEL=${D_MODEL:-128}
D_FF=${D_FF:-1024}
NUM_HEADS=${NUM_HEADS:-6}
D_KV=${D_KV:-64}
DROPOUT_RATE=${DROPOUT_RATE:-0.1}

BATCH_SIZE=${BATCH_SIZE:-256}
INFER_SIZE=${INFER_SIZE:-96}
NUM_EPOCHS=${NUM_EPOCHS:-200}
LEARNING_RATE=${LEARNING_RATE:-1e-4}
EARLY_STOP=${EARLY_STOP:-10}
EVAL_INTERVAL=${EVAL_INTERVAL:-1}
MAX_LEN=${MAX_LEN:-50}
BEAM_SIZE=${BEAM_SIZE:-20}
TOPK_LIST=${TOPK_LIST:-"5 10 20"}
SEED=${SEED:-2025}
NUM_WORKERS=${NUM_WORKERS:-4}

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_DIR=${OUTPUT_DIR:-${SCRIPT_DIR}/ckpt/${DATASET}/${TIMESTAMP}}
LOG_DIR=${LOG_DIR:-${SCRIPT_DIR}/logs}
LOG_FILE=${LOG_DIR}/train_${DATASET}_${TIMESTAMP}.log

mkdir -p "${OUTPUT_DIR}" "${LOG_DIR}"

echo "Starting TIGER SID training"
echo "  dataset=${DATASET}"
echo "  data_path=${DATA_PATH}"
echo "  output_dir=${OUTPUT_DIR}"
echo "  log_file=${LOG_FILE}"
echo "  num_gpus=${NUM_GPUS}"

nohup torchrun --nproc_per_node="${NUM_GPUS}" \
  "${SCRIPT_DIR}/train_sid.py" \
  --base_model "${BASE_MODEL}" \
  --load_pretrained "${LOAD_PRETRAINED}" \
  --dataset "${DATASET}" \
  --data_path "${DATA_PATH}" \
  --inter_file "${INTER_FILE}" \
  --index_file "${INDEX_FILE}" \
  --num_layers "${NUM_LAYERS}" \
  --num_decoder_layers "${NUM_DECODER_LAYERS}" \
  --d_model "${D_MODEL}" \
  --d_ff "${D_FF}" \
  --num_heads "${NUM_HEADS}" \
  --d_kv "${D_KV}" \
  --dropout_rate "${DROPOUT_RATE}" \
  --batch_size "${BATCH_SIZE}" \
  --infer_size "${INFER_SIZE}" \
  --num_epochs "${NUM_EPOCHS}" \
  --lr "${LEARNING_RATE}" \
  --early_stop "${EARLY_STOP}" \
  --eval_interval "${EVAL_INTERVAL}" \
  --max_len "${MAX_LEN}" \
  --beam_size "${BEAM_SIZE}" \
  --topk_list ${TOPK_LIST} \
  --output_dir "${OUTPUT_DIR}" \
  --seed "${SEED}" \
  --num_workers "${NUM_WORKERS}" > "${LOG_FILE}" 2>&1 &

PID=$!
echo "Launched in background. PID=${PID}"
echo "Watch log: tail -f ${LOG_FILE}"
