#!/bin/bash

# 设置参数
INPUT_FILE="/llm-reco-ssd-share/zhangrongzhou/Graduation_project/data/item_embeddings.parquet"
OUTPUT_FILE="/llm-reco-ssd-share/zhangrongzhou/Graduation_project/data/item_codes.parquet"
MODEL_PATH="/llm-reco-ssd-share/zhangrongzhou/Graduation_project/models/beauty_rq_kmeans.pth"
N_LAYERS=4
CODEBOOK_SIZE=256
BATCH_SIZE=1024

# 日志文件
LOG_FILE="process_embedding_$(date +%Y%m%d_%H%M%S).log"

# 创建输出目录
mkdir -p $(dirname $OUTPUT_FILE)
mkdir -p $(dirname $MODEL_PATH)

echo "开始RQ-KMEANS编码..."
echo "输入文件: $INPUT_FILE"
echo "输出文件: $OUTPUT_FILE"
echo "模型路径: $MODEL_PATH"
echo "参数: layers=$N_LAYERS, codebook_size=$CODEBOOK_SIZE"
echo ""

# 使用nohup后台运行，输出重定向到日志文件
nohup python3 process_embedding.py \
    --input_file "$INPUT_FILE" \
    --output_file "$OUTPUT_FILE" \
    --n_layers $N_LAYERS \
    --codebook_size $CODEBOOK_SIZE \
    --batch_size $BATCH_SIZE \
    --model_path "$MODEL_PATH" \
    > $LOG_FILE 2>&1 &

echo "任务已提交后台运行，进程ID: $!"
echo "日志文件: $LOG_FILE"
echo "查看进度: tail -f $LOG_FILE"
