#!/usr/bin/env python3
"""
OneRec推理脚本 - 保存Top-5命中样本
用法: python inference_hits.py --checkpoint <path> --output <json_path>
"""
import torch
import json
import argparse
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import DataLoader
from model import OneRecV2, ModelConfig
from dataset import JSONOneRecDataset
import logging
import random

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_model_and_config(checkpoint_path: str, device: str = 'cuda:0'):
    """加载模型和配置"""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # 从checkpoint恢复配置
    config_dict = checkpoint.get('config', {})
    model_config = ModelConfig(**config_dict)

    # 创建模型
    model = OneRecV2(model_config).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    logger.info(f"Model loaded from {checkpoint_path}")
    logger.info(f"Beam size: {model_config.beam_size}")

    return model, model_config


def decode_sids_to_items(sids_batch, sid_to_item):
    """
    将SID序列解码为item ID（与训练代码一致）
    Args:
        sids_batch: [bs, beam_size, semantic_token_num]
        sid_to_item: dict mapping tuple(sid) -> item_id
    Returns:
        item_ids: [bs, beam_size]
    """
    shape = sids_batch.shape[:-1]
    flat = sids_batch.reshape(-1, sids_batch.size(-1)).cpu()
    out = torch.full((flat.size(0),), -1, dtype=torch.long)

    for i in range(flat.size(0)):
        sid_tuple = tuple(int(x) for x in flat[i].tolist())
        out[i] = sid_to_item.get(sid_tuple, -1)

    return out.reshape(shape)


def main():
    parser = argparse.ArgumentParser(description='OneRec推理脚本 - 保存Top-5命中样本')
    parser.add_argument('--checkpoint', type=str, required=True, help='模型checkpoint路径')
    parser.add_argument('--dataset', type=str, default='Beauty', help='数据集名称')
    parser.add_argument('--data_path', type=str,
                        default='../data/tiger_data', help='数据路径')
    parser.add_argument('--output', type=str, default='hit_samples.json', help='输出JSON文件路径')
    parser.add_argument('--device', type=str, default='cuda:0', help='设备')
    parser.add_argument('--batch_size', type=int, default=64, help='批大小')
    parser.add_argument('--max_hist_len', type=int, default=50, help='最大历史长度')

    args = parser.parse_args()

    # 加载模型
    logger.info("Loading model...")
    model, model_config = load_model_and_config(args.checkpoint, args.device)

    # 确保beam_size=5
    if model_config.beam_size != 5:
        logger.warning(f"Model beam_size is {model_config.beam_size}, but we need 5. "
                       f"Predictions may not match expectations.")

    # 加载测试数据集
    logger.info("Loading test dataset...")
    test_dataset = JSONOneRecDataset(
        dataset=args.dataset,
        data_path=args.data_path,
        mode='test',
        max_hist_len=args.max_hist_len,
        target_type_num=model_config.target_type_num
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    # 构建sid_to_item映射（与训练代码一致）
    # 从Dataset的item_to_sid构建反向映射
    logger.info("Building SID to item mapping...")
    sid_to_item = {}
    for item_id_str, sid_codes in test_dataset.item_to_sid.items():
        sid_tuple = tuple(sid_codes)
        item_id_int = int(item_id_str)
        if sid_tuple not in sid_to_item:
            sid_to_item[sid_tuple] = item_id_int
        # 如果碰撞，保留第一个（训练时按频率选择，这里简化处理）

    logger.info(f"SID mapping: {len(sid_to_item)} unique SIDs (from {len(test_dataset.item_to_sid)} items)")
    collision_count = len(test_dataset.item_to_sid) - len(sid_to_item)
    if collision_count > 0:
        logger.info(f"SID collisions: {collision_count} items share SIDs with others")

    # 推理并保存命中样本
    hit_samples = []
    total_samples = 0
    hit_count = 0
    debug_count = 0  # 用于控制调试输出数量

    logger.info("Running inference...")
    with torch.no_grad():
        for batch in tqdm(test_loader, desc='Inference'):
            uid, his_sids, his_pid_types, target_sids, target_type = batch

            batch_size = uid.size(0)
            total_samples += batch_size

            his_sids = his_sids.to(args.device)
            his_pid_types = his_pid_types.to(args.device)
            target_sids = target_sids.to(args.device)
            target_type = target_type.to(args.device)

            # Beam Search生成
            gen_outputs = model.generate(
                his_sids=his_sids,
                his_pid_types=his_pid_types,
                target_type=target_type
            )

            pred_tokens = gen_outputs['tokens']  # [bs, beam_size, semantic_token_num]

            # 在SID层面检查Top-5是否命中（不要先转换成item ID！）
            pred_sids_top5 = pred_tokens[:, :5, :]  # [bs, 5, semantic_token_num]
            target_sids_expanded = target_sids.unsqueeze(1)  # [bs, 1, semantic_token_num]

            # 比较SID是否完全相同
            sid_matches = (pred_sids_top5 == target_sids_expanded).all(dim=2)  # [bs, 5]
            hits = sid_matches.any(dim=1)  # [bs]

            # 解码历史SIDs为item IDs
            max_hist_len = his_sids.size(1) // 4
            his_sids_reshaped = his_sids.view(batch_size, max_hist_len, 4)
            hist_item_ids = decode_sids_to_items(his_sids_reshaped, sid_to_item)

            # 只对命中的样本进行item ID转换和保存
            for i in range(batch_size):
                # 打印前5个样本的详细信息（无论是否命中）
                if debug_count < 5:
                    logger.info(f"\n{'='*80}")
                    logger.info(f"Sample {debug_count + 1}:")
                    logger.info(f"  User ID: {uid[i].item()}")

                    # 历史item IDs（显示所有）
                    hist_pids = hist_item_ids[i].tolist()
                    hist_pids_valid = [pid for pid in hist_pids if pid >= 0]
                    logger.info(f"  History PIDs (total {len(hist_pids_valid)}): {hist_pids_valid}")

                    # 历史SID序列（显示最后3个，token字符串格式）
                    hist_sids_sample = his_sids_reshaped[i]  # [max_hist_len, 4]
                    valid_hist_sids = []
                    for j in range(max_hist_len):
                        if hist_sids_sample[j].sum() > 0:
                            valid_hist_sids.append(hist_sids_sample[j].cpu().tolist())
                    logger.info(f"  History SIDs (last 3, token format):")
                    for sid_list in valid_hist_sids[-3:]:
                        token_str = f"<|sid_begin|><s_a_{sid_list[0]}><s_b_{sid_list[1]}><s_c_{sid_list[2]}><s_d_{sid_list[3]}><|sid_end|>"
                        logger.info(f"    {token_str}")

                    # 目标SID（token字符串格式）
                    target_sids_sample = target_sids[i].cpu().tolist()
                    target_token_str = f"<|sid_begin|><s_a_{target_sids_sample[0]}><s_b_{target_sids_sample[1]}><s_c_{target_sids_sample[2]}><s_d_{target_sids_sample[3]}><|sid_end|>"
                    logger.info(f"  Target SID (token format): {target_token_str}")
                    logger.info(f"  Target SID (numbers): {target_sids_sample}")
                    target_item_id = decode_sids_to_items(
                        target_sids[i:i+1].unsqueeze(1), sid_to_item
                    )[0, 0].item()
                    logger.info(f"  Target PID: {target_item_id}")

                    # Top-5预测SID（token字符串格式）
                    logger.info(f"  Top-5 Predicted SIDs (token format):")
                    for k in range(5):
                        pred_sids_sample = pred_sids_top5[i, k].cpu().tolist()
                        pred_token_str = f"<|sid_begin|><s_a_{pred_sids_sample[0]}><s_b_{pred_sids_sample[1]}><s_c_{pred_sids_sample[2]}><s_d_{pred_sids_sample[3]}><|sid_end|>"
                        logger.info(f"    [{k+1}] {pred_token_str}")

                    # 是否命中
                    logger.info(f"  Hit in Top-5: {hits[i].item()}")
                    if hits[i]:
                        hit_pos = sid_matches[i].nonzero(as_tuple=True)[0][0].item() + 1
                        logger.info(f"  Hit Position: {hit_pos}")

                    # Top-5预测item IDs
                    top5_pred_item_ids = decode_sids_to_items(
                        pred_tokens[i:i+1, :5, :], sid_to_item
                    )[0].tolist()
                    logger.info(f"  Top-5 Predicted PIDs: {top5_pred_item_ids}")

                    # 检查预测的SID是否在映射中
                    logger.info(f"  Checking if predicted SIDs exist in mapping:")
                    for k in range(5):
                        pred_sid_tuple = tuple(pred_sids_top5[i, k].cpu().tolist())
                        exists = pred_sid_tuple in sid_to_item
                        logger.info(f"    [{k+1}] {pred_sid_tuple} -> exists: {exists}")

                    logger.info(f"{'='*80}\n")

                    debug_count += 1

                if hits[i]:
                    hit_count += 1

                    # 提取历史item IDs (去除padding -1)
                    hist_pids = hist_item_ids[i].tolist()
                    hist_pids = [pid for pid in hist_pids if pid >= 0]

                    # 转换Top-5预测SID为item IDs
                    top5_pred_item_ids = decode_sids_to_items(
                        pred_tokens[i:i+1, :5, :], sid_to_item
                    )[0].tolist()  # [5]

                    # 转换目标SID为item ID
                    target_item_id = decode_sids_to_items(
                        target_sids[i:i+1].unsqueeze(1), sid_to_item
                    )[0, 0].item()

                    hit_samples.append({
                        'history_pids': hist_pids,
                        'top5_predictions': top5_pred_item_ids,
                        'target_pid': target_item_id
                    })

    # 保存结果
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(hit_samples, f, indent=2)

    logger.info(f"Inference completed!")
    logger.info(f"Total samples: {total_samples}")
    logger.info(f"Hit samples (Top-5): {hit_count} ({hit_count/total_samples*100:.2f}%)")
    logger.info(f"Saved {len(hit_samples)} hit samples to {output_path}")


if __name__ == '__main__':
    main()
