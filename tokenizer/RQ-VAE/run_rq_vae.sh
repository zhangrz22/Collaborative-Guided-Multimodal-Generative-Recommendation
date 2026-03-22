#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

INPUT_FILE=${INPUT_FILE:-/llm-reco-ssd-share/zhangrongzhou/Collaborative-Guided-Multimodal-Generative-Recommendation/data/item_text_embeddings.parquet}
OUTPUT_FILE=${OUTPUT_FILE:-/llm-reco-ssd-share/zhangrongzhou/Collaborative-Guided-Multimodal-Generative-Recommendation/data/tiger_data/Beauty/item_RQ-VAE_codes.parquet}
MODEL_PATH=${MODEL_PATH:-/llm-reco-ssd-share/zhangrongzhou/Graduation_project/models/beauty_rq_vae.pth}

N_LAYERS=${N_LAYERS:-4}
CODEBOOK_SIZE=${CODEBOOK_SIZE:-256}
HIDDEN_DIM=${HIDDEN_DIM:-1024}
LATENT_DIM=${LATENT_DIM:-256}
EPOCHS=${EPOCHS:-50}
BATCH_SIZE=${BATCH_SIZE:-512}
LR=${LR:-1e-3}
WEIGHT_DECAY=${WEIGHT_DECAY:-1e-5}
COMMITMENT_WEIGHT=${COMMITMENT_WEIGHT:-0.25}
KL_WEIGHT=${KL_WEIGHT:-0.0}
NUM_WORKERS=${NUM_WORKERS:-4}
SEED=${SEED:-2025}
EMA_DECAY=${EMA_DECAY:-0.95}
DEAD_CODE_THRESHOLD=${DEAD_CODE_THRESHOLD:-10.0}
USE_EMA=${USE_EMA:-true}
RESTART_UNUSED_CODES=${RESTART_UNUSED_CODES:-true}
REFINE_COLLISIONS=${REFINE_COLLISIONS:-true}
MAX_REFINE_ROUNDS=${MAX_REFINE_ROUNDS:-20}
TARGET_COLLISION_RATE=${TARGET_COLLISION_RATE:-0.10}
REFINE_SK_EPSILON=${REFINE_SK_EPSILON:-0.003}
REFINE_SK_ITERS=${REFINE_SK_ITERS:-50}

LOG_FILE="${SCRIPT_DIR}/rq_vae_$(date +%Y%m%d_%H%M%S).log"
mkdir -p "$(dirname "$OUTPUT_FILE")" "$(dirname "$MODEL_PATH")"

echo "Start RQ-VAE tokenizer"
echo "  input_file=${INPUT_FILE}"
echo "  output_file=${OUTPUT_FILE}"
echo "  model_path=${MODEL_PATH}"
echo "  n_layers=${N_LAYERS}, codebook_size=${CODEBOOK_SIZE}"
echo "  hidden_dim=${HIDDEN_DIM}, latent_dim=${LATENT_DIM}"
echo "  epochs=${EPOCHS}, batch_size=${BATCH_SIZE}, lr=${LR}"
echo "  kl_weight=${KL_WEIGHT}, ema_decay=${EMA_DECAY}, dead_code_th=${DEAD_CODE_THRESHOLD}"
echo "  refine_collisions=${REFINE_COLLISIONS}, target_collision_rate=${TARGET_COLLISION_RATE}"

CMD=(
  python3 "${SCRIPT_DIR}/process_embedding.py"
  --input_file "${INPUT_FILE}" \
  --output_file "${OUTPUT_FILE}" \
  --model_path "${MODEL_PATH}" \
  --n_layers "${N_LAYERS}" \
  --codebook_size "${CODEBOOK_SIZE}" \
  --hidden_dim "${HIDDEN_DIM}" \
  --latent_dim "${LATENT_DIM}" \
  --epochs "${EPOCHS}" \
  --batch_size "${BATCH_SIZE}" \
  --lr "${LR}" \
  --weight_decay "${WEIGHT_DECAY}" \
  --commitment_weight "${COMMITMENT_WEIGHT}" \
  --kl_weight "${KL_WEIGHT}" \
  --ema_decay "${EMA_DECAY}" \
  --dead_code_threshold "${DEAD_CODE_THRESHOLD}" \
  --max_refine_rounds "${MAX_REFINE_ROUNDS}" \
  --target_collision_rate "${TARGET_COLLISION_RATE}" \
  --refine_sk_epsilon "${REFINE_SK_EPSILON}" \
  --refine_sk_iters "${REFINE_SK_ITERS}" \
  --num_workers "${NUM_WORKERS}" \
  --seed "${SEED}" \
  --amp
)

if [[ "${USE_EMA}" == "true" ]]; then CMD+=(--ema); else CMD+=(--no-ema); fi
if [[ "${RESTART_UNUSED_CODES}" == "true" ]]; then CMD+=(--restart_unused_codes); else CMD+=(--no-restart_unused_codes); fi
if [[ "${REFINE_COLLISIONS}" == "true" ]]; then CMD+=(--refine_collisions); else CMD+=(--no-refine_collisions); fi

nohup "${CMD[@]}" > "${LOG_FILE}" 2>&1 &

PID=$!
echo "Launched in background. PID=${PID}"
echo "Log file: ${LOG_FILE}"
echo "Watch: tail -f ${LOG_FILE}"
