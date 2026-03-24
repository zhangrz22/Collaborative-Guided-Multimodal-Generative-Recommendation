import socket
import torch.nn as nn
import numpy as np

def get_free_port():
    """获取可用端口"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        s.listen(1)
        port = s.getsockname()[1]
    return port

def count_parameters(model):
    dense_params = 0
    sparse_params = 0

    for name, module in model.named_modules():
        if isinstance(module, nn.Embedding):
            # 统计 Embedding 层参数（sparse）
            for p in module.parameters():
                if p.requires_grad:
                    sparse_params += p.numel()
        elif len(list(module.children())) == 0:  # 叶子节点模块
            # 统计非 Embedding 层参数（dense）
            for p in module.parameters():
                if p.requires_grad:
                    dense_params += p.numel()

    return dense_params, sparse_params


class WarmupCosineScheduler:
    """Warmup + CosineAnnealing 学习率调度器"""

    def __init__(
            self,
            optimizer,
            warmup_steps: int,
            total_steps: int,
            min_lr_ratio: float,
    ):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr_ratio = min_lr_ratio
        self.base_lr = optimizer.param_groups[0]['lr']

    def step(self, global_step):

        if global_step < self.warmup_steps:
            # Warmup phase: linear increase
            lr = self.base_lr * global_step / self.warmup_steps
        else:
            # Cosine annealing phase
            progress = (global_step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            lr = self.min_lr_ratio + (1 - self.min_lr_ratio) * 0.5 * (1 + np.cos(np.pi * progress))
            lr = lr * self.base_lr

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def get_last_lr(self):
        return [param_group['lr'] for param_group in self.optimizer.param_groups]


def evaluate_predictions(predictions: dict, targets: dict, top_k: int):
    """
    评估预测结果

    Args:
        predictions: {uid: [pred_sid1, pred_sid2, ...]}
        targets: {uid: {target_sid1, target_sid2, ...}}
        top_k: 评估的top-k

    Returns:
        dict: 包含各种评估指标
    """
    hit_count = 0
    total_users = len(predictions)
    total_precision = 0.0
    total_recall = 0.0
    total_ndcg = 0.0

    for uid, pred_sids in predictions.items():
        target_sids = targets[uid]
        pred_sids_topk = pred_sids[:top_k]

        # 计算交集
        hits = set(pred_sids_topk) & target_sids

        # Hit Ratio: 只要有交集就算hit
        if len(hits) > 0:
            hit_count += 1

        # Precision: 预测中正确的比例
        precision = len(hits) / len(pred_sids_topk) if len(pred_sids_topk) > 0 else 0
        total_precision += precision

        # Recall: 目标中被预测到的比例
        recall = len(hits) / len(target_sids) if len(target_sids) > 0 else 0
        total_recall += recall

        # NDCG: 考虑位置信息
        dcg = 0.0
        idcg = 0.0
        for i, pred_sid in enumerate(pred_sids_topk):
            if pred_sid in target_sids:
                dcg += 1.0 / np.log2(i + 2)  # i+2 because index starts at 0

        # 理想情况下的DCG
        for i in range(min(len(target_sids), top_k)):
            idcg += 1.0 / np.log2(i + 2)

        ndcg = dcg / idcg
        total_ndcg += ndcg

    metrics = {
        'hit_ratio': hit_count / total_users if total_users > 0 else 0,
        'precision': total_precision / total_users if total_users > 0 else 0,
        'recall': total_recall / total_users if total_users > 0 else 0,
        'ndcg': total_ndcg / total_users if total_users > 0 else 0,
        'total_users': total_users,
        'hit_users': hit_count,
    }

    return metrics
