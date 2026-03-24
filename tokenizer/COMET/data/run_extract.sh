#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Step 1: Download CLIP model (skip if already exists)
CLIP_DIR=${CLIP_DIR:-/llm-reco-ssd-share/zhangrongzhou/Collaborative-Guided-Multimodal-Generative-Recommendation/tokenizer/COMET/model/clip-vit-large-patch14}

if [ ! -d "${CLIP_DIR}" ] || [ -z "$(ls -A ${CLIP_DIR} 2>/dev/null)" ]; then
    echo "=== Step 1: Downloading CLIP ViT-L/14 ==="
    python3 "${SCRIPT_DIR}/download_clip.py" --save_dir "${CLIP_DIR}"
else
    echo "=== Step 1: CLIP already at ${CLIP_DIR}, skipping ==="
fi

# Step 2: Extract image embeddings
INPUT_JSON=${INPUT_JSON:-${SCRIPT_DIR}/item_info_with_image.json}
OUTPUT_FILE=${OUTPUT_FILE:-/llm-reco-ssd-share/zhangrongzhou/Collaborative-Guided-Multimodal-Generative-Recommendation/data/item_image_embeddings.parquet}
BATCH_SIZE=${BATCH_SIZE:-64}
DEVICE=${DEVICE:-cuda:0}
NUM_WORKERS=${NUM_WORKERS:-4}

echo "=== Step 2: Extracting image embeddings ==="
echo "  input_json=${INPUT_JSON}"
echo "  output_file=${OUTPUT_FILE}"
echo "  clip_dir=${CLIP_DIR}"
echo "  batch_size=${BATCH_SIZE}, device=${DEVICE}"

python3 "${SCRIPT_DIR}/extract_image_embeddings.py" \
    --input_json "${INPUT_JSON}" \
    --output_file "${OUTPUT_FILE}" \
    --clip_dir "${CLIP_DIR}" \
    --batch_size "${BATCH_SIZE}" \
    --device "${DEVICE}" \
    --num_workers "${NUM_WORKERS}"

echo "=== Done ==="
