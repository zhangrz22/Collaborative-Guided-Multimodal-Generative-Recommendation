#!/usr/bin/env python3
"""
OneRec演示推理脚本
输入：历史item PIDs
输出：Top-5预测的item PIDs
"""
import torch
import json
import argparse
from pathlib import Path
from model import OneRecV2, ModelConfig
from dataset import JSONOneRecDataset

def main():
    parser = argparse.ArgumentParser(description='OneRec演示推理')
    parser.add_argument('--checkpoint', type=str, required=True, help='模型checkpoint路径')
    parser.add_argument('--dataset', type=str, default='Beauty', help='数据集名称')
    parser.add_argument('--data_path', type=str, default='../data/tiger_data', help='数据路径')
    parser.add_argument('--history_pids', type=str, required=True, help='历史item PIDs，逗号分隔，如"9580,9621,9766,9856"')
    parser.add_argument('--device', type=str, default='cuda:0', help='设备')
    parser.add_argument('--max_hist_len', type=int, default=50, help='最大历史长度')
    parser.add_argument('--top_k', type=int, default=5, help='输出Top-K预测结果')

    args = parser.parse_args()

    # 解析历史PIDs
    history_pids = [int(x.strip()) for x in args.history_pids.split(',')]
    print(f"输入历史PIDs: {history_pids}")

    # 加载模型
    print("加载模型...")
    checkpoint = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    config_dict = checkpoint.get('config', {})
    model_config = ModelConfig(**config_dict)
    model = OneRecV2(model_config).to(args.device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"模型加载完成，beam_size={model_config.beam_size}")

    # 加载映射
    print("加载映射...")
    dataset = JSONOneRecDataset(
        dataset=args.dataset,
        data_path=args.data_path,
        mode='test',
        max_hist_len=args.max_hist_len,
        target_type_num=model_config.target_type_num
    )

    # 构建sid_to_item映射
    sid_to_item = {}
    for item_id_str, sid_codes in dataset.item_to_sid.items():
        sid_tuple = tuple(sid_codes)
        item_id_int = int(item_id_str)
        if sid_tuple not in sid_to_item:
            sid_to_item[sid_tuple] = item_id_int

    print(f"映射加载完成: {len(sid_to_item)} unique SIDs")

    # 将历史PIDs转换为SIDs
    history_sids = []
    for pid in history_pids:
        if str(pid) in dataset.item_to_sid:
            sids = dataset.item_to_sid[str(pid)]
            history_sids.extend(sids)
            print(f"  PID {pid} -> SID {sids}")
        else:
            print(f"  警告: PID {pid} 不在映射中")
            history_sids.extend([0, 0, 0, 0])  # padding

    # 填充或截断到max_hist_len * 4
    target_len = args.max_hist_len * 4
    if len(history_sids) > target_len:
        history_sids = history_sids[-target_len:]
    elif len(history_sids) < target_len:
        padding = [-1] * (target_len - len(history_sids))
        history_sids = padding + history_sids

    # 构造模型输入
    his_sids = torch.tensor([history_sids], dtype=torch.long).to(args.device)
    his_pid_types = torch.zeros((1, model_config.target_type_num, args.max_hist_len), dtype=torch.long).to(args.device)
    his_pid_types[0, 0, :] = 1  # 所有历史都标记为type 0
    target_type = torch.tensor([0], dtype=torch.long).to(args.device)

    # 推理
    print("\n开始推理...")
    with torch.no_grad():
        gen_outputs = model.generate(
            his_sids=his_sids,
            his_pid_types=his_pid_types,
            target_type=target_type
        )

    pred_tokens = gen_outputs['tokens']  # [1, beam_size, 4]

    # 转换为item PIDs
    print(f"\nTop-{args.top_k}预测结果:")
    for i in range(min(args.top_k, pred_tokens.size(1))):
        pred_sid = pred_tokens[0, i].cpu().tolist()
        pred_sid_tuple = tuple(pred_sid)
        pred_pid = sid_to_item.get(pred_sid_tuple, -1)

        # 构造token字符串
        token_str = f"<|sid_begin|><s_a_{pred_sid[0]}><s_b_{pred_sid[1]}><s_c_{pred_sid[2]}><s_d_{pred_sid[3]}><|sid_end|>"
        print(f"  [{i+1}] PID: {pred_pid:5d}  SID: {token_str}")

    # 输出简洁格式
    topk_pids = []
    for i in range(min(args.top_k, pred_tokens.size(1))):
        pred_sid_tuple = tuple(pred_tokens[0, i].cpu().tolist())
        pred_pid = sid_to_item.get(pred_sid_tuple, -1)
        topk_pids.append(pred_pid)

    print(f"\nTop-{args.top_k} PIDs: {topk_pids}")

if __name__ == '__main__':
    main()
