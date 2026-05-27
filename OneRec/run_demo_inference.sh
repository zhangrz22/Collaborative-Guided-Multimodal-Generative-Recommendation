#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 模型checkpoint路径
CHECKPOINT=${CHECKPOINT:-/llm-reco-ssd-share/zhangrongzhou/Collaborative-Guided-Multimodal-Generative-Recommendation/OneRec/checkpoints/Beauty/20260526_175217/best.pt}

# 数据配置
DATASET=${DATASET:-Beauty}
DATA_PATH=${DATA_PATH:-../data/tiger_data}

# 历史item PIDs（逗号分隔）
HISTORY_PIDS=${HISTORY_PIDS:-"9580,9621,9766,9856"}

# 推理配置
DEVICE=${DEVICE:-cuda:0}
MAX_HIST_LEN=${MAX_HIST_LEN:-50}
TOP_K=${TOP_K:-10}

echo "OneRec Demo Inference"
echo "  checkpoint=${CHECKPOINT}"
echo "  history_pids=${HISTORY_PIDS}"
echo "  top_k=${TOP_K}"
echo ""

python3 "${SCRIPT_DIR}/demo_inference.py" \
  --checkpoint "${CHECKPOINT}" \
  --dataset "${DATASET}" \
  --data_path "${DATA_PATH}" \
  --history_pids "${HISTORY_PIDS}" \
  --device "${DEVICE}" \
  --max_hist_len "${MAX_HIST_LEN}" \
  --top_k "${TOP_K}"
