#!/bin/bash

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun \
    --nproc_per_node=8 \
    train.py \
    --dataset Beauty \
    --data_path ./data \
    --hidden_units 50 \
    --num_blocks 2 \
    --num_heads 1 \
    --maxlen 50 \
    --dropout_rate 0.2 \
    --batch_size 128 \
    --lr 0.001 \
    --l2_emb 0.0 \
    --num_neg 32 \
    --num_epochs 20 \
    --eval_epoch 2 \
    --num_workers 3 \
    --device cuda \
    --output_dir ./ckpt/Beauty
