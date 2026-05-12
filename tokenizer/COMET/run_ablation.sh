#!/bin/bash
set -euo pipefail

# ============================================================================
# COMET Ablation Experiments
#   1) COMET_no_image  — remove image modality
#   2) COMET_no_cf     — remove CF query (use learnable query)
#   3) COMET_no_image_no_cf — remove both image and CF
# ============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ---------- shared paths (same as run.sh) ----------
INPUT_FILE=${INPUT_FILE:-/llm-reco-ssd-share/zhangrongzhou/Collaborative-Guided-Multimodal-Generative-Recommendation/data/item_text_embeddings.parquet}
IMAGE_FILE=${IMAGE_FILE:-/llm-reco-ssd-share/zhangrongzhou/Collaborative-Guided-Multimodal-Generative-Recommendation/data/item_image_embeddings.parquet}
CF_CKPT=${CF_CKPT:-/llm-reco-ssd-share/zhangrongzhou/Collaborative-Guided-Multimodal-Generative-Recommendation/SASRec/ckpt/Beauty_best/SASRec_epoch14_hr0.0982.pth}

OUTPUT_DIR=${OUTPUT_DIR:-/llm-reco-ssd-share/zhangrongzhou/Collaborative-Guided-Multimodal-Generative-Recommendation/data/tiger_data/Beauty}
MODEL_DIR=${MODEL_DIR:-/llm-reco-ssd-share/zhangrongzhou/Graduation_project/models}

DEVICE=${DEVICE:-cuda:0}
EPOCHS=${EPOCHS:-300}
BATCH_SIZE=${BATCH_SIZE:-1024}

# ---------- shared hyperparams ----------
COMMON_ARGS=(
  --input_file "${INPUT_FILE}"
  --image_file "${IMAGE_FILE}"
  --cf_ckpt "${CF_CKPT}"
  --d_model 256
  --n_heads 4
  --fusion_dropout 0.1
  --n_attn_layers 2
  --text_n_tokens 4
  --n_e_list 256 256 256 256
  --e_dim 64
  --decoder_dims 256 512 1024 2048
  --commitment_weight 0.25
  --ema_decay 0.99
  --dead_threshold 2.0
  --diversity_weight 0.0001
  --quant_loss_weight 1.0
  --w_text 1.0
  --w_image 0.1
  --w_cf 0.1
  --sk_epsilons 0.0 0.0 0.005 0.01
  --sk_iters 50
  --sk_start_ratio 0.2
  --kmeans_iters 100
  --n_clusters 10
  --epochs "${EPOCHS}"
  --batch_size "${BATCH_SIZE}"
  --lr 1e-3
  --weight_decay 1e-4
  --seed 42
  --device "${DEVICE}"
  --max_refine_rounds 5
  --target_collision_rate 0.05
)

# ============================================================================
# Experiment 1: COMET without image
# ============================================================================
echo "========================================"
echo "  Ablation 1: COMET without image"
echo "========================================"

LOG1="${SCRIPT_DIR}/ablation_no_image_$(date +%Y%m%d_%H%M%S).log"

nohup python3 "${SCRIPT_DIR}/process_embedding.py" \
  "${COMMON_ARGS[@]}" \
  --output_file "${OUTPUT_DIR}/item_COMET_no_image_codes.parquet" \
  --model_path "${MODEL_DIR}/beauty_comet_no_image.pth" \
  --ablate_image \
  > "${LOG1}" 2>&1 &

PID1=$!
echo "  PID=${PID1}  Log: ${LOG1}"

# ============================================================================
# Experiment 2: COMET without CF
# ============================================================================
echo "========================================"
echo "  Ablation 2: COMET without CF"
echo "========================================"

LOG2="${SCRIPT_DIR}/ablation_no_cf_$(date +%Y%m%d_%H%M%S).log"

nohup python3 "${SCRIPT_DIR}/process_embedding.py" \
  "${COMMON_ARGS[@]}" \
  --output_file "${OUTPUT_DIR}/item_COMET_no_cf_codes.parquet" \
  --model_path "${MODEL_DIR}/beauty_comet_no_cf.pth" \
  --ablate_cf \
  > "${LOG2}" 2>&1 &

PID2=$!
echo "  PID=${PID2}  Log: ${LOG2}"

# ============================================================================
# Experiment 3: COMET without image AND CF
# ============================================================================
echo "========================================"
echo "  Ablation 3: COMET without image & CF"
echo "========================================"

LOG3="${SCRIPT_DIR}/ablation_no_image_no_cf_$(date +%Y%m%d_%H%M%S).log"

nohup python3 "${SCRIPT_DIR}/process_embedding.py" \
  "${COMMON_ARGS[@]}" \
  --output_file "${OUTPUT_DIR}/item_COMET_no_image_no_cf_codes.parquet" \
  --model_path "${MODEL_DIR}/beauty_comet_no_image_no_cf.pth" \
  --ablate_image --ablate_cf \
  > "${LOG3}" 2>&1 &

PID3=$!
echo "  PID=${PID3}  Log: ${LOG3}"

echo ""
echo "All ablation experiments launched."
echo "Watch logs:"
echo "  tail -f ${LOG1}"
echo "  tail -f ${LOG2}"
echo "  tail -f ${LOG3}"
