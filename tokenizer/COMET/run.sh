#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

INPUT_FILE=${INPUT_FILE:-/llm-reco-ssd-share/zhangrongzhou/Collaborative-Guided-Multimodal-Generative-Recommendation/data/item_text_embeddings.parquet}
IMAGE_FILE=${IMAGE_FILE:-/llm-reco-ssd-share/zhangrongzhou/Collaborative-Guided-Multimodal-Generative-Recommendation/data/item_image_embeddings.parquet}
OUTPUT_FILE=${OUTPUT_FILE:-/llm-reco-ssd-share/zhangrongzhou/Collaborative-Guided-Multimodal-Generative-Recommendation/data/tiger_data/Beauty/item_COMET_codes.parquet}
MODEL_PATH=${MODEL_PATH:-/llm-reco-ssd-share/zhangrongzhou/Graduation_project/models/beauty_comet_rqvae.pth}
CF_CKPT=${CF_CKPT:-/llm-reco-ssd-share/zhangrongzhou/Collaborative-Guided-Multimodal-Generative-Recommendation/SASRec/ckpt/Beauty_best/SASRec_epoch14_hr0.0982.pth}

# COMET fusion
D_MODEL=${D_MODEL:-256}
N_HEADS=${N_HEADS:-4}
FUSION_DROPOUT=${FUSION_DROPOUT:-0.1}

# Model
N_E_LIST=${N_E_LIST:-"256 256 256 256"}
E_DIM=${E_DIM:-32}
DECODER_DIMS=${DECODER_DIMS:-"128 256 512 1024"}
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
EPOCHS=${EPOCHS:-300}
BATCH_SIZE=${BATCH_SIZE:-1024}
LR=${LR:-1e-3}
WEIGHT_DECAY=${WEIGHT_DECAY:-1e-4}
NUM_WORKERS=${NUM_WORKERS:-4}
SEED=${SEED:-42}
DEVICE=${DEVICE:-cuda:0}

# Refine
MAX_REFINE_ROUNDS=${MAX_REFINE_ROUNDS:-5}
TARGET_COLLISION_RATE=${TARGET_COLLISION_RATE:-0.05}

LOG_FILE="${SCRIPT_DIR}/comet_rqvae_$(date +%Y%m%d_%H%M%S).log"
mkdir -p "$(dirname "$OUTPUT_FILE")" "$(dirname "$MODEL_PATH")"

echo "COMET RQ-VAE tokenizer (CF-as-Query Cross-Attention fusion)"
echo "  input=${INPUT_FILE}"
echo "  image=${IMAGE_FILE}"
echo "  output=${OUTPUT_FILE}"
echo "  model=${MODEL_PATH}"
echo "  cf_ckpt=${CF_CKPT}"
echo "  d_model=${D_MODEL}, n_heads=${N_HEADS}, fusion_dropout=${FUSION_DROPOUT}"
echo "  n_e_list=${N_E_LIST}, e_dim=${E_DIM}, decoder_dims=${DECODER_DIMS}"
echo "  cf_alpha=${CF_ALPHA}, cf_warmup=${CF_WARMUP}, cf_ramp=${CF_RAMP}"
echo "  diversity_weight=${DIVERSITY_WEIGHT}, n_clusters=${N_CLUSTERS}"
echo "  epochs=${EPOCHS}, batch=${BATCH_SIZE}, lr=${LR}"

nohup python3 "${SCRIPT_DIR}/process_embedding.py" \
  --input_file "${INPUT_FILE}" \
  --image_file "${IMAGE_FILE}" \
  --output_file "${OUTPUT_FILE}" \
  --model_path "${MODEL_PATH}" \
  --cf_ckpt "${CF_CKPT}" \
  --d_model "${D_MODEL}" \
  --n_heads "${N_HEADS}" \
  --fusion_dropout "${FUSION_DROPOUT}" \
  --n_e_list ${N_E_LIST} \
  --e_dim "${E_DIM}" \
  --decoder_dims ${DECODER_DIMS} \
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
