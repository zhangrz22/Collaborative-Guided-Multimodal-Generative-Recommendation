#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 模型checkpoint路径
CHECKPOINT=${CHECKPOINT:-/llm-reco-ssd-share/zhangrongzhou/Collaborative-Guided-Multimodal-Generative-Recommendation/OneRec/checkpoints/Beauty/20260329_233337/best.pt}

# 数据配置
DATASET=${DATASET:-Beauty}
DATA_PATH=${DATA_PATH:-../data/tiger_data}
OUTPUT=${OUTPUT:-hit_samples.json}

# 推理配置
DEVICE=${DEVICE:-cuda:0}
BATCH_SIZE=${BATCH_SIZE:-64}
MAX_HIST_LEN=${MAX_HIST_LEN:-50}

LOG_FILE="${SCRIPT_DIR}/inference_hits_$(date +%Y%m%d_%H%M%S).log"

echo "OneRec Inference - Save Top-5 Hit Samples"
echo "  checkpoint=${CHECKPOINT}"
echo "  dataset=${DATASET}"
echo "  data_path=${DATA_PATH}"
echo "  output=${OUTPUT}"
echo "  device=${DEVICE}"
echo "  batch_size=${BATCH_SIZE}"
echo ""

python3 "${SCRIPT_DIR}/inference_hits.py" \
  --checkpoint "${CHECKPOINT}" \
  --dataset "${DATASET}" \
  --data_path "${DATA_PATH}" \
  --output "${OUTPUT}" \
  --device "${DEVICE}" \
  --batch_size "${BATCH_SIZE}" \
  --max_hist_len "${MAX_HIST_LEN}" \
  2>&1 | tee "${LOG_FILE}"

echo ""
echo "Inference completed!"
echo "Log saved to: ${LOG_FILE}"
