#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

INPUT_FILE=${INPUT_FILE:-/llm-reco-ssd-share/zhangrongzhou/Collaborative-Guided-Multimodal-Generative-Recommendation/data/item_text_embeddings.parquet}
OUTPUT_FILE=${OUTPUT_FILE:-/llm-reco-ssd-share/zhangrongzhou/Collaborative-Guided-Multimodal-Generative-Recommendation/data/tiger_data/Beauty/item_LETTER_codes.parquet}
MODEL_PATH=${MODEL_PATH:-/llm-reco-ssd-share/zhangrongzhou/Graduation_project/models/beauty_letter_rqvae.pth}
CF_CKPT=${CF_CKPT:-/llm-reco-ssd-share/zhangrongzhou/Collaborative-Guided-Multimodal-Generative-Recommendation/SASRec/ckpt/Beauty/SASRec_epoch20_hr0.0817.pth}

# Model
N_E_LIST=${N_E_LIST:-"256 256 256 256"}
E_DIM=${E_DIM:-32}
ENCODER_DIMS=${ENCODER_DIMS:-"1024 512 256 128"}
COMMITMENT_WEIGHT=${COMMITMENT_WEIGHT:-0.25}
EMA_DECAY=${EMA_DECAY:-0.99}
DEAD_THRESHOLD=${DEAD_THRESHOLD:-2.0}
DIVERSITY_WEIGHT=${DIVERSITY_WEIGHT:-0.0001}
QUANT_LOSS_WEIGHT=${QUANT_LOSS_WEIGHT:-1.0}
CF_ALPHA=${CF_ALPHA:-0.05}
CF_WARMUP=${CF_WARMUP:-50}
CF_RAMP=${CF_RAMP:-50}
SK_EPSILONS=${SK_EPSILONS:-"0.0 0.0 0.0 0.003"}
SK_ITERS=${SK_ITERS:-50}
KMEANS_ITERS=${KMEANS_ITERS:-100}
N_CLUSTERS=${N_CLUSTERS:-10}

# Training
EPOCHS=${EPOCHS:-2000}
BATCH_SIZE=${BATCH_SIZE:-1024}
LR=${LR:-1e-3}
WEIGHT_DECAY=${WEIGHT_DECAY:-1e-4}
NUM_WORKERS=${NUM_WORKERS:-4}
SEED=${SEED:-42}
DEVICE=${DEVICE:-cuda:0}

# Refine
MAX_REFINE_ROUNDS=${MAX_REFINE_ROUNDS:-5}
TARGET_COLLISION_RATE=${TARGET_COLLISION_RATE:-0.05}

LOG_FILE="${SCRIPT_DIR}/letter_rqvae_$(date +%Y%m%d_%H%M%S).log"
mkdir -p "$(dirname "$OUTPUT_FILE")" "$(dirname "$MODEL_PATH")"

echo "LETTER RQ-VAE tokenizer (with CF)"
echo "  input=${INPUT_FILE}"
echo "  output=${OUTPUT_FILE}"
echo "  model=${MODEL_PATH}"
echo "  cf_ckpt=${CF_CKPT}"
echo "  n_e_list=${N_E_LIST}, e_dim=${E_DIM}, encoder_dims=${ENCODER_DIMS}"
echo "  cf_alpha=${CF_ALPHA}, cf_warmup=${CF_WARMUP}, cf_ramp=${CF_RAMP}"
echo "  diversity_weight=${DIVERSITY_WEIGHT}, n_clusters=${N_CLUSTERS}"
echo "  epochs=${EPOCHS}, batch=${BATCH_SIZE}, lr=${LR}"

nohup python3 "${SCRIPT_DIR}/process_embedding.py" \
  --input_file "${INPUT_FILE}" \
  --output_file "${OUTPUT_FILE}" \
  --model_path "${MODEL_PATH}" \
  --cf_ckpt "${CF_CKPT}" \
  --n_e_list ${N_E_LIST} \
  --e_dim "${E_DIM}" \
  --encoder_dims ${ENCODER_DIMS} \
  --commitment_weight "${COMMITMENT_WEIGHT}" \
  --ema_decay "${EMA_DECAY}" \
  --dead_threshold "${DEAD_THRESHOLD}" \
  --diversity_weight "${DIVERSITY_WEIGHT}" \
  --quant_loss_weight "${QUANT_LOSS_WEIGHT}" \
  --cf_alpha "${CF_ALPHA}" \
  --cf_warmup "${CF_WARMUP}" \
  --cf_ramp "${CF_RAMP}" \
  --sk_epsilons ${SK_EPSILONS} \
  --sk_iters "${SK_ITERS}" \
  --kmeans_iters "${KMEANS_ITERS}" \
  --n_clusters "${N_CLUSTERS}" \
  --epochs "${EPOCHS}" \
  --batch_size "${BATCH_SIZE}" \
  --lr "${LR}" \
  --weight_decay "${WEIGHT_DECAY}" \
  --num_workers "${NUM_WORKERS}" \
  --seed "${SEED}" \
  --device "${DEVICE}" \
  --max_refine_rounds "${MAX_REFINE_ROUNDS}" \
  --target_collision_rate "${TARGET_COLLISION_RATE}" \
  > "${LOG_FILE}" 2>&1 &

PID=$!
echo "PID=${PID}"
echo "Log: ${LOG_FILE}"
echo "Watch: tail -f ${LOG_FILE}"
