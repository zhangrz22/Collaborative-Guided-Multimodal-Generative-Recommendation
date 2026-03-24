#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

INPUT_JSON=${INPUT_JSON:-/llm-reco-ssd-share/zhangrongzhou/Collaborative-Guided-Multimodal-Generative-Recommendation/data/item_info.json}
OUTPUT_DIR=${OUTPUT_DIR:-${SCRIPT_DIR}/beauty_image}
OUTPUT_JSON=${OUTPUT_JSON:-${SCRIPT_DIR}/item_info_with_image.json}
WORKERS=${WORKERS:-32}
TIMEOUT=${TIMEOUT:-10}

echo "Download item images"
echo "  input_json=${INPUT_JSON}"
echo "  output_dir=${OUTPUT_DIR}"
echo "  output_json=${OUTPUT_JSON}"
echo "  workers=${WORKERS}, timeout=${TIMEOUT}s"

python3 "${SCRIPT_DIR}/download_images.py" \
  --input_json "${INPUT_JSON}" \
  --output_dir "${OUTPUT_DIR}" \
  --output_json "${OUTPUT_JSON}" \
  --workers "${WORKERS}" \
  --timeout "${TIMEOUT}"
