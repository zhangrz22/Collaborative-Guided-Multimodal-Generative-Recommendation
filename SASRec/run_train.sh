#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# ===== 配置参数 =====
DATASET=Beauty
HIDDEN_UNITS=50
NUM_BLOCKS=2
NUM_HEADS=1
MAXLEN=50
DROPOUT_RATE=0.2
BATCH_SIZE=128
LR=0.001
L2_EMB=0.0
NUM_NEG=32
NUM_EPOCHS=20
EVAL_EPOCH=2
NUM_WORKERS=3
# ====================

DATA_PATH=./data
OUTPUT_DIR=./ckpt/${DATASET}/
LOG_FILE=./logs/train_${DATASET}_$(date +%Y%m%d_%H%M%S).log

mkdir -p ./logs
mkdir -p $OUTPUT_DIR

echo "=========================================="
echo "开始训练 SASRec - ${DATASET}"
echo "模型保存到: ${OUTPUT_DIR}"
echo "日志文件: ${LOG_FILE}"
echo "=========================================="

nohup torchrun \
    --nproc_per_node=8 \
    train.py \
    --dataset $DATASET \
    --data_path $DATA_PATH \
    --hidden_units $HIDDEN_UNITS \
    --num_blocks $NUM_BLOCKS \
    --num_heads $NUM_HEADS \
    --maxlen $MAXLEN \
    --dropout_rate $DROPOUT_RATE \
    --batch_size $BATCH_SIZE \
    --lr $LR \
    --l2_emb $L2_EMB \
    --num_neg $NUM_NEG \
    --num_epochs $NUM_EPOCHS \
    --eval_epoch $EVAL_EPOCH \
    --num_workers $NUM_WORKERS \
    --device cuda \
    --output_dir $OUTPUT_DIR \
    > ${LOG_FILE} 2>&1 &

TRAIN_PID=$!
echo "训练进程 PID: ${TRAIN_PID}"
echo ${TRAIN_PID} > train.pid

echo "训练已在后台启动！"
echo "查看实时日志: tail -f ${LOG_FILE}"
echo "停止训练: kill ${TRAIN_PID}"
