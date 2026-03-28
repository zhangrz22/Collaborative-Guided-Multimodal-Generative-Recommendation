import torch
import torch.nn as nn
from torch.optim import AdamW
from tqdm import tqdm
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from pathlib import Path
from torchmetrics.aggregation import MeanMetric
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from model import OneRecV2, ModelConfig
import os
import pandas as pd
import time
# DDP相关导入
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import socket
from datetime import timedelta
from dataset import OneRecDataset, JSONOneRecDataset
import pickle
import threading
from utils import get_free_port, count_parameters, WarmupCosineScheduler, evaluate_predictions

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def setup_ddp(rank: int, world_size: int, master_port: int) -> None:
    """设置 DDP 环境"""
    logger.info(f'[Rank {rank}] Setting up DDP...')

    env_vars = {
        'NCCL_SOCKET_IFNAME': 'lo',
        'NCCL_DEBUG': 'WARN',  # ✅ 改为 INFO 以便调试
        'NCCL_IB_DISABLE': '1',
        'NCCL_P2P_DISABLE': '1',
        'NCCL_SHM_DISABLE': '0',
        'NCCL_SOCKET_NTHREADS': '4',
        'NCCL_NSOCKS_PERTHREAD': '2',
        'TORCH_NCCL_ASYNC_ERROR_HANDLING': '1',
    }

    for key, value in env_vars.items():
        os.environ[key] = value

    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = str(master_port)

    # ✅ 在初始化前设置设备
    torch.cuda.set_device(rank)
    logger.info(f'[Rank {rank}] Device set to cuda:{rank}')

    # ✅ 非主进程等待主进程先启动
    if rank != 0:
        time.sleep(2)
        logger.info(f'[Rank {rank}] Waited for master process')

    try:
        # 初始化进程组
        logger.info(f'[Rank {rank}] Initializing process group...')
        dist.init_process_group(
            backend='nccl',
            init_method='env://',
            rank=rank,
            world_size=world_size,
            timeout=timedelta(seconds=300)  # ✅ 增加超时
        )
        logger.info(f'[Rank {rank}] Process group initialized')

        # Barrier 同步
        logger.info(f'[Rank {rank}] Waiting at barrier...')
        dist.barrier()
        logger.info(f'[Rank {rank}] Barrier passed')

        logger.info(f'[Rank {rank}] DDP initialized successfully with world_size={world_size}')

    except Exception as e:
        logger.error(f'[Rank {rank}] Failed to initialize DDP: {e}')
        raise


def cleanup_ddp():
    """清理DDP进程组"""
    dist.destroy_process_group()


def is_main_process():
    """判断是否为主进程"""
    return not dist.is_initialized() or dist.get_rank() == 0

class Trainer:
    def __init__(
            self,
            model: OneRecV2,
            train_loader: DataLoader,
            valid_loader: DataLoader,
            learning_rate: float,
            weight_decay: float,
            num_epochs: int,
            warmup_ratio: float,
            min_lr_ratio: float,
            save_dir: str,
            eval_interval: int,
            eval_start_epoch: int,
            k_list: Tuple[int, ...],
            sid_to_item: Dict[tuple, int],
            early_stop: int = 0,
    ):
        self.k_list = k_list
        self.model = model
        self.sid_to_item = sid_to_item
        self.early_stop = early_stop
        self.no_improve_count = 0
        self.eval_start_epoch = eval_start_epoch
        # 处理DDP包装的模型
        if isinstance(model, DDP):
            self.config = model.module.config
        else:
            self.config = model.config
        self.device = self.config.device
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.num_epochs = num_epochs
        self.eval_interval = eval_interval
        self.save_dir = Path(save_dir)
        self.save_thread = None
        
        # 只有主进程创建目录
        if is_main_process():
            self.save_dir.mkdir(parents=True, exist_ok=True)

        # 优化器
        self.optimizer = AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=(0.9, 0.95)
        )

        # 计算总步数和warmup步数
        total_steps = num_epochs * len(train_loader)
        warmup_steps = int(total_steps * warmup_ratio)

        # Warmup + CosineAnnealing 调度器
        self.scheduler = WarmupCosineScheduler(
            optimizer=self.optimizer,
            warmup_steps=warmup_steps,
            total_steps=total_steps,
            min_lr_ratio=min_lr_ratio
        )

        self.best_recall = 0.0
        self.global_step = 1
        self.start_epoch = 0

        # 只有主进程初始化 TensorBoard
        if is_main_process():
            log_dir = self.save_dir / 'tensorboard'
            self.writer = SummaryWriter(log_dir=str(log_dir))
            logger.info(f'TensorBoard log directory: {log_dir}')
        else:
            self.writer = None
        
        if is_main_process():
            logger.info(f'Total steps: {total_steps}, Warmup steps: {warmup_steps}')

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """训练一个epoch"""
        self.model.train()

        # DDP: 设置sampler的epoch以确保每个epoch的数据shuffle不同
        if hasattr(self.train_loader.sampler, 'set_epoch'):
            self.train_loader.sampler.set_epoch(epoch)

        # 使用统一的字典管理所有 MeanMetric
        metrics_dict = {
            'loss': MeanMetric().to(self.device),
            'acc': MeanMetric().to(self.device),
        }

        # 添加位置级别的指标
        for i in range(self.config.semantic_token_num):
            metrics_dict[f'position_acc_{i}'] = MeanMetric().to(self.device)
            metrics_dict[f'position_loss_{i}'] = MeanMetric().to(self.device)

        # 只在主进程显示进度条，避免多进程tqdm冲突
        if is_main_process():
            pbar = tqdm(self.train_loader, desc=f'Epoch {epoch + 1}/{self.num_epochs} [Train]')
        else:
            pbar = self.train_loader

        for batch in pbar:
            # 从 tuple 加载数据
            _, his_sids, his_pid_types, target_sids, target_type = batch

            batch_size = his_pid_types.size(0)

            his_sids = his_sids.to(self.device)
            his_pid_types = his_pid_types.to(self.device)
            target_sids = target_sids.to(self.device)
            target_type = target_type.to(self.device)

            # 前向传播
            outputs = self.model(
                his_sids=his_sids,
                his_pid_types=his_pid_types,
                target_sids=target_sids,
                target_type=target_type
            )
            loss = outputs['ntp_loss']
            all_correct = outputs['all_correct']

            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()

            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

            self.optimizer.step()
            self.scheduler.step(self.global_step)

            # 统一更新所有指标（使用batch_size加权）
            metrics_dict['loss'].update(loss, weight=batch_size)
            metrics_dict['acc'].update(all_correct, weight=batch_size)

            for i in range(self.config.semantic_token_num):
                metrics_dict[f'position_acc_{i}'].update(
                    outputs[f'position_acc/position_acc_{i}'],
                    weight=batch_size
                )
                metrics_dict[f'position_loss_{i}'].update(
                    outputs[f'position_loss/position_loss_{i}'],
                    weight=batch_size
                )

            # 只在主进程记录到 TensorBoard - 每个batch
            if self.writer is not None:
                self.writer.add_scalar('Train/loss', loss.item(), self.global_step)
                self.writer.add_scalar('Train/accuracy', all_correct.item(), self.global_step)
                self.writer.add_scalar('Train/learning_rate', self.scheduler.get_last_lr()[0], self.global_step)

                # 记录位置级别的准确率和损失
                for i in range(self.config.semantic_token_num):
                    self.writer.add_scalar(
                        f'Train/position_acc_{i}',
                        outputs[f'position_acc/position_acc_{i}'].item(),
                        self.global_step
                    )
                    self.writer.add_scalar(
                        f'Train/position_loss_{i}',
                        outputs[f'position_loss/position_loss_{i}'].item(),
                        self.global_step
                    )

            self.global_step += 1

            # 只在主进程更新进度条
            if is_main_process():
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'acc': f'{all_correct.item():.4f}',
                    'lr': f'{self.scheduler.get_last_lr()[0]:.6f}'
                })

        # 统一计算所有平均指标
        metrics = {
            f'train_{key}': metric.compute().item()
            for key, metric in metrics_dict.items()
        }

        return metrics

    @torch.no_grad()
    def evaluate(self, epoch: int) -> Dict[str, float]:
        """分布式评估函数 - 使用SID反解为item ID后计算Recall和NDCG"""
        self.model.eval()

        # 设置验证集的 epoch（重要！确保不同进程处理不同数据）
        if dist.is_initialized() and hasattr(self.valid_loader.sampler, 'set_epoch'):
            self.valid_loader.sampler.set_epoch(epoch)

        # 使用统一的字典管理所有 MeanMetric
        metrics_dict = {
            'loss': MeanMetric().to(self.device),
            'acc': MeanMetric().to(self.device),
        }

        # 添加 Recall 和 NDCG 指标
        for k in self.k_list:
            metrics_dict[f'Recall@{k}'] = MeanMetric().to(self.device)
            metrics_dict[f'NDCG@{k}'] = MeanMetric().to(self.device)

        max_k = max(self.k_list)

        # 只在主进程显示进度条
        if is_main_process():
            pbar = tqdm(self.valid_loader, desc=f'Epoch {epoch + 1}/{self.num_epochs} [Valid]')
        else:
            pbar = self.valid_loader

        for batch in pbar:
            # 从 tuple 加载数据
            _, his_sids, his_pid_types, target_sids, target_type = batch

            batch_size = his_pid_types.size(0)

            his_sids = his_sids.to(self.device)
            his_pid_types = his_pid_types.to(self.device)
            target_sids = target_sids.to(self.device)
            target_type = target_type.to(self.device)

            # 计算验证损失
            outputs = self.model(
                his_sids=his_sids,
                his_pid_types=his_pid_types,
                target_sids=target_sids,
                target_type=target_type
            )

            loss = outputs['ntp_loss']
            all_correct = outputs['all_correct']

            # 统一更新所有指标（使用batch_size加权）
            metrics_dict['loss'].update(loss, weight=batch_size)
            metrics_dict['acc'].update(all_correct, weight=batch_size)

            # Beam Search 生成
            if isinstance(self.model, DDP):
                model = self.model.module
            else:
                model = self.model

            gen_outputs = model.generate(
                his_sids=his_sids,
                his_pid_types=his_pid_types,
                target_type=target_type
            )

            pred_tokens = gen_outputs['tokens']  # [bs, beam_size, semantic_token_num]

            # SID反解为item ID后计算Recall和NDCG
            pred_item_ids = self._decode_sids_to_items(pred_tokens)  # [bs, beam_size]
            target_item_ids = self._decode_sids_to_items(target_sids.unsqueeze(1))  # [bs, 1]

            # pos_index: [bs, beam_size] bool, 表示每个beam位置是否命中
            pos_index = pred_item_ids[:, :max_k].eq(target_item_ids)

            for k in self.k_list:
                kk = min(k, pred_item_ids.size(1))
                # Recall@k: 前k个中是否有命中
                recall_k = pos_index[:, :kk].any(dim=1).float().mean()
                metrics_dict[f'Recall@{k}'].update(recall_k, weight=batch_size)
                # NDCG@k
                ranks = torch.arange(1, pos_index.size(1) + 1, dtype=torch.float32)
                discounts = (1.0 / torch.log2(ranks + 1.0)).unsqueeze(0).expand_as(pos_index.float())
                dcg = torch.where(pos_index, discounts, torch.zeros_like(discounts))
                ndcg_k = dcg[:, :kk].sum(dim=1).float().mean()
                metrics_dict[f'NDCG@{k}'].update(ndcg_k, weight=batch_size)

            # 只在主进程更新进度条
            if is_main_process():
                postfix = {'loss': f'{loss.item():.4f}'}
                for k in self.k_list:
                    postfix[f'R@{k}'] = f'{metrics_dict[f"Recall@{k}"].compute().item():.4f}'
                pbar.set_postfix(postfix)

        # 同步所有进程的指标
        if dist.is_initialized():
            for metric in metrics_dict.values():
                dist.all_reduce(metric.mean_value, op=dist.ReduceOp.SUM)
                dist.all_reduce(metric.weight, op=dist.ReduceOp.SUM)

        # 统一计算所有平均指标
        metrics = {
            f'valid_{key}': metric.compute().item()
            for key, metric in metrics_dict.items()
        }

        # 只在主进程记录到 TensorBoard
        if is_main_process() and self.writer is not None:
            self.writer.add_scalar('Valid/loss', metrics['valid_loss'], epoch)
            self.writer.add_scalar('Valid/accuracy', metrics['valid_acc'], epoch)
            for k in self.k_list:
                self.writer.add_scalar(f'Valid/Recall@{k}', metrics[f'valid_Recall@{k}'], epoch)
                self.writer.add_scalar(f'Valid/NDCG@{k}', metrics[f'valid_NDCG@{k}'], epoch)

        return metrics

    def _decode_sids_to_items(self, sid_tokens: torch.Tensor) -> torch.Tensor:
        """
        将SID token序列反解为item ID
        sid_tokens: [bs, num, semantic_token_num] 或任意前缀维度 + semantic_token_num
        返回: [bs, num] item IDs, 未知SID返回-1
        """
        shape = sid_tokens.shape[:-1]
        flat = sid_tokens.reshape(-1, sid_tokens.size(-1)).cpu()
        out = torch.full((flat.size(0),), -1, dtype=torch.long)
        for i in range(flat.size(0)):
            sid_tuple = tuple(int(x) for x in flat[i].tolist())
            out[i] = self.sid_to_item.get(sid_tuple, -1)
        return out.reshape(shape)

    def save_checkpoint_async(self, epoch, metrics, is_best, current_hit_ratio):
        """异步保存检查点"""
        # 等待上一次保存完成
        if self.save_thread is not None:
            self.save_thread.join()

        # 启动新的保存线程
        self.save_thread = threading.Thread(
            target=self.save_checkpoint,
            args=(epoch, metrics, is_best, current_hit_ratio)
        )
        self.save_thread.start()

    def save_checkpoint(self, epoch: int, metrics: Dict[str, float], is_best: bool, current_hit_ratio: float):
        # 只有主进程保存检查点
        if not is_main_process():
            return

        if isinstance(self.model, DDP):
            model = self.model.module
        else:
            model = self.model
        checkpoint = {
            'epoch': epoch,
            'global_step': self.global_step,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': metrics,
            'model_config': model.config,
            'best_recall': self.best_recall,
        }

        # 保存最新检查点
        latest_path = self.save_dir / 'latest.pt'
        torch.save(checkpoint, latest_path)
        logger.info(f'Saved latest checkpoint to {latest_path} with recall {current_hit_ratio:.6f}')

        # 保存最佳检查点
        if is_best:
            best_path = self.save_dir / 'best.pt'
            torch.save(checkpoint, best_path)
            logger.info(f'Saved best checkpoint to {best_path} with recall {current_hit_ratio:.6f}')

    def load_checkpoint(self, checkpoint_path: str):
        """加载检查点"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        if isinstance(self.model, DDP):
            model = self.model.module
        else:
            model = self.model
        model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.global_step = checkpoint['global_step'] + 1
        self.start_epoch = checkpoint['epoch'] + 1
        self.best_recall = checkpoint.get('best_recall', checkpoint.get('best_hit_ratio', 0.0))
        logger.info(f'Loaded checkpoint from {checkpoint_path}')
        return checkpoint['metrics']

    def train(self):
        """完整训练流程"""
        logger.info('Starting training...')

        for epoch in range(self.start_epoch, self.num_epochs):
            # 训练
            train_metrics = self.train_epoch(epoch)

            # 格式化输出训练指标
            if is_main_process():
                train_log = f'Epoch {epoch + 1} - Train: loss={train_metrics["train_loss"]:.4f}, acc={train_metrics["train_acc"]:.4f}'
                logger.info(train_log)

            # 评估：需要同时满足 eval_interval 和 eval_start_epoch
            if (epoch + 1) >= self.eval_start_epoch and (epoch + 1) % self.eval_interval == 0:
                valid_metrics = self.evaluate(epoch)

                # all_reduce 后各 rank 的 metrics 完全一致，所有 rank 各自判断即可
                current_recall = valid_metrics[f'valid_Recall@{self.k_list[-1]}']
                is_best = current_recall > self.best_recall
                if is_best:
                    self.best_recall = current_recall
                    self.no_improve_count = 0
                else:
                    self.no_improve_count += 1

                if is_main_process():
                    # 格式化输出验证指标
                    valid_log = (f'Epoch {epoch + 1} - Valid: loss={valid_metrics["valid_loss"]:.4f}, '
                                 f'acc={valid_metrics["valid_acc"]:.4f}')
                    for k in self.k_list:
                        valid_log += f', Recall@{k}={valid_metrics[f"valid_Recall@{k}"]:.4f}'
                        valid_log += f', NDCG@{k}={valid_metrics[f"valid_NDCG@{k}"]:.4f}'
                    logger.info(valid_log)

                    if is_best:
                        logger.info(f'New best Recall@{self.k_list[-1]}: {self.best_recall:.4f}')
                    else:
                        logger.info(f'No improvement for {self.no_improve_count} eval(s)')

                    # 保存检查点
                    self.save_checkpoint(epoch, valid_metrics, is_best, current_recall)

                # 早停：所有 rank 各自判断（metrics 一致所以结果一致），不需要 broadcast
                if self.early_stop > 0 and self.no_improve_count >= self.early_stop:
                    if is_main_process():
                        logger.info(f'Early stopping triggered after {self.no_improve_count} evals without improvement')
                    break

        if is_main_process():
            logger.info('Training completed!')
            logger.info(f'Best Recall@{self.k_list[-1]}: {self.best_recall:.4f}')
            self.writer.close()
            logger.info('TensorBoard writer closed')

    @torch.no_grad()
    def test(self, test_loader: DataLoader) -> Dict[str, float]:
        """训练结束后加载best checkpoint在测试集上评估"""
        # 加载best checkpoint
        best_path = self.save_dir / 'best.pt'
        if not best_path.exists():
            if is_main_process():
                logger.warning(f'Best checkpoint not found at {best_path}, skipping test.')
            return {}

        if is_main_process():
            logger.info('=' * 80)
            logger.info(f'Testing with best checkpoint (Recall@{self.k_list[-1]}={self.best_recall:.4f})')
            logger.info('=' * 80)

        checkpoint = torch.load(best_path, map_location=self.device, weights_only=False)
        if isinstance(self.model, DDP):
            self.model.module.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint['model_state_dict'])

        # 复用evaluate逻辑，临时替换valid_loader
        original_loader = self.valid_loader
        self.valid_loader = test_loader
        test_metrics = self.evaluate(epoch=0)
        self.valid_loader = original_loader

        # 将key从 valid_ 改为 test_
        test_metrics = {k.replace('valid_', 'test_'): v for k, v in test_metrics.items()}

        if is_main_process():
            test_log = 'Test Results:'
            for k in self.k_list:
                test_log += f' Recall@{k}={test_metrics[f"test_Recall@{k}"]:.4f}'
                test_log += f' NDCG@{k}={test_metrics[f"test_NDCG@{k}"]:.4f}'
            logger.info(test_log)

        return test_metrics

def ddp_worker(
        rank: int,
        world_size: int,
        master_port: int,
        data_format: str,
        parquet_path: Optional[str],
        train_indices: Optional[List[int]],
        val_indices: Optional[List[int]],
        dataset_name: Optional[str],
        data_path: Optional[str],
        model_config: ModelConfig,
        max_hist_len: int,
        max_hist_video_len: int,
        max_hist_goods_len: int,
        batch_size: int,
        val_batch_size: int,
        num_workers: int,
        learning_rate: float,
        weight_decay: float,
        num_epochs: int,
        warmup_ratio: float,
        min_lr_ratio: float,
        save_dir: str,
        resume_from: Optional[str],
        eval_interval: int,
        eval_start_epoch: int,
        k_list: Tuple[int, ...],
        sid_to_item: Dict[tuple, int],
        early_stop: int,
):
    """DDP训练工作函数，每个GPU进程执行此函数"""
    # 初始化DDP
    setup_ddp(rank, world_size, master_port)

    # ✅ 根据数据格式加载数据集
    if data_format == 'json':
        logger.info(f'[Rank {rank}] Loading JSON dataset: {dataset_name}')
        train_dataset = JSONOneRecDataset(
            dataset=dataset_name,
            data_path=data_path,
            mode='train',
            max_hist_len=max_hist_len,
            target_type_num=model_config.target_type_num
        )
        val_dataset = JSONOneRecDataset(
            dataset=dataset_name,
            data_path=data_path,
            mode='valid',
            max_hist_len=max_hist_len,
            target_type_num=model_config.target_type_num
        )
    else:
        logger.info(f'[Rank {rank}] Loading Parquet dataset from {parquet_path}')
        full_dataset = OneRecDataset(
            parquet_path=parquet_path,
            max_hist_video_len=max_hist_video_len,
            max_hist_goods_len=max_hist_goods_len
        )
        # ✅ 使用 Subset 创建训练集和验证集
        from torch.utils.data import Subset
        train_dataset = Subset(full_dataset, train_indices)
        val_dataset = Subset(full_dataset, val_indices)

    logger.info(f'[Rank {rank}] Train size: {len(train_dataset)}, Val size: {len(val_dataset)}')

    # 设置模型config的device为当前rank
    model_config.device = f'cuda:{rank}'

    # 创建模型并移动到对应GPU
    model = OneRecV2(model_config).bfloat16()

    if is_main_process():
        dense_params, sparse_params = count_parameters(model)
        logger.info(f'Model dense parameters: {dense_params}, sparse parameters: {sparse_params}')

    # 用DDP包装模型
    model = DDP(model, device_ids=[rank])

    # 创建DistributedSampler
    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        drop_last=False,
        seed=42,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=True,
        # persistent_workers=True,
    )

    val_sampler = DistributedSampler(
        val_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=False,
        drop_last=False,
        seed=42,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=val_batch_size,
        sampler=val_sampler,
        num_workers=num_workers,
        pin_memory=True,
        # persistent_workers=True,
    )

    # 创建训练器
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        valid_loader=val_loader,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        num_epochs=num_epochs,
        warmup_ratio=warmup_ratio,
        min_lr_ratio=min_lr_ratio,
        save_dir=save_dir,
        eval_interval=eval_interval,
        eval_start_epoch=eval_start_epoch,
        k_list=k_list,
        sid_to_item=sid_to_item,
        early_stop=early_stop,
    )

    # 如果指定了恢复路径，则加载检查点
    if resume_from is not None:
        trainer.load_checkpoint(resume_from)

    # 开始训练
    trainer.train()

    # 训练结束后在测试集上评估
    if data_format == 'json':
        test_dataset = JSONOneRecDataset(
            dataset=dataset_name,
            data_path=data_path,
            mode='test',
            max_hist_len=max_hist_len,
            target_type_num=model_config.target_type_num
        )

        test_sampler = DistributedSampler(
            test_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=False,
            drop_last=False,
            seed=42,
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=val_batch_size,
            sampler=test_sampler,
            num_workers=num_workers,
            pin_memory=True,
        )

        trainer.test(test_loader)

    # 清理DDP
    cleanup_ddp()


def train_ddp(
        data_format: str,
        parquet_path: Optional[str],
        train_indices: Optional[List[int]],
        val_indices: Optional[List[int]],
        dataset_name: Optional[str],
        data_path: Optional[str],
        model_config: ModelConfig,
        max_hist_len: int,
        max_hist_video_len: int,
        max_hist_goods_len: int,
        batch_size: int,
        val_batch_size: int,
        num_workers: int,
        learning_rate: float,
        weight_decay: float,
        num_epochs: int,
        warmup_ratio: float,
        min_lr_ratio: float,
        save_dir: str,
        resume_from: Optional[str],
        eval_interval: int,
        eval_start_epoch: int,
        k_list: Tuple[int, ...],
        sid_to_item: Dict[tuple, int],
        world_size: int,
        early_stop: int = 0,
):
    """DDP多卡训练入口函数"""
    import torch.multiprocessing as mp

    # 如果未指定world_size，使用所有可用GPU
    if world_size is None:
        world_size = torch.cuda.device_count()

    if world_size < 1:
        logger.error('No GPU available!')
        return

    logger.info(f'Starting DDP training with {world_size} GPUs')
    if data_format == 'json':
        logger.info(f'Dataset: {dataset_name}')
    else:
        logger.info(f'Train indices: {len(train_indices)}, Val indices: {len(val_indices)}')

    master_port = get_free_port()
    logger.info(f'Using master port: {master_port}')

    # ✅ 传递所有参数
    mp.spawn(
        ddp_worker,
        args=(
            world_size,
            master_port,
            data_format,
            parquet_path,
            train_indices,
            val_indices,
            dataset_name,
            data_path,
            model_config,
            max_hist_len,
            max_hist_video_len,
            max_hist_goods_len,
            batch_size,
            val_batch_size,
            num_workers,
            learning_rate,
            weight_decay,
            num_epochs,
            warmup_ratio,
            min_lr_ratio,
            save_dir,
            resume_from,
            eval_interval,
            eval_start_epoch,
            k_list,
            sid_to_item,
            early_stop,
        ),
        nprocs=world_size,
        join=True
    )

    logger.info('DDP training completed!')

@torch.no_grad()
def infer(
        model: OneRecV2,
        test_loader: DataLoader,
        output_file: str,
        top_k_list: list = (1, 5, 10),
        evaluate: bool = True,
):
    """推理函数"""
    model.eval()
    # 处理DDP包装的模型
    if isinstance(model, DDP):
        config = model.module.config
    else:
        config = model.config

    max_k = max(top_k_list)
    all_predictions = dict()
    all_targets = dict()  # 记录每个用户的所有目标sids

    pbar = tqdm(test_loader, desc='Inference')

    for batch in pbar:
        uids, his_sids, his_pid_types, target_sids, target_type = batch

        his_sids = his_sids.to(config.device)
        his_pid_types = his_pid_types.to(config.device)
        target_type = target_type.to(config.device)

        # 生成
        gen_outputs = model.generate(
            his_sids=his_sids,
            his_pid_types=his_pid_types,
            target_type=target_type
        )

        pred_tokens = gen_outputs['tokens']  # [bs, beam_size, semantic_token_num]

        # 转换为列表
        pred_tokens = pred_tokens.cpu().numpy()
        target_sids = target_sids.cpu().numpy()  # [bs, semantic_token_num]

        # 保存结果
        for i in range(pred_tokens.shape[0]):
            uid = uids[i].item()
            top_sids = [tuple(sids.tolist()) for sids in pred_tokens[i]]
            target_sid = tuple(target_sids[i].tolist())

            # 保存预测结果（每个uid只保存一次预测，保存最大的top_k）
            if uid not in all_predictions:
                all_predictions[uid] = top_sids[:max_k]

            # 收集该用户的所有目标sids
            if uid not in all_targets:
                all_targets[uid] = set()
            all_targets[uid].add(target_sid)

    # 保存预测结果到文件
    with open(output_file, 'wb') as f:
        pickle.dump(all_predictions, f)
    logger.info(f'Predictions saved to {output_file}')

    avg_len = np.mean([len(rec_list) for rec_list in all_predictions.values()])
    logger.info(f'Average predicted length: {avg_len:.4f}')

    # 评估
    if evaluate:
        all_metrics = {}
        for top_k in top_k_list:
            metrics = evaluate_predictions(all_predictions, all_targets, top_k)
            all_metrics[f'top_{top_k}'] = metrics

            logger.info(f"\nEvaluation Results (top-{top_k}):")
            logger.info(f"  Hit Ratio: {metrics['hit_ratio']:.4f}")
            logger.info(f"  Precision: {metrics['precision']:.4f}")
            logger.info(f"  Recall: {metrics['recall']:.4f}")
            logger.info(f"  NDCG: {metrics['ndcg']:.4f}")
            logger.info(f"  Total users: {metrics['total_users']}")
        return all_predictions, all_metrics

    return all_predictions


if __name__ == '__main__':
    from torch.utils.data import random_split
    import torch.multiprocessing as mp
    import argparse

    # ✅ 设置启动方法
    mp.set_start_method('spawn', force=True)

    # 添加命令行参数解析
    parser = argparse.ArgumentParser(description='OneRecV2 Training and Inference')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'infer'],
                        help='运行模式: train（训练）或 infer（推理）')

    # ===== 数据格式选择 =====
    parser.add_argument('--data_format', type=str, default='parquet', choices=['parquet', 'json'],
                        help='数据格式: parquet（原OneRec格式）或 json（LC-Rec风格）')
    parser.add_argument('--dataset', type=str, default='Beauty',
                        help='数据集名称（JSON格式时使用），如Beauty, Baby等')
    parser.add_argument('--data_path', type=str, default='../data/tiger_data',
                        help='数据根目录（JSON格式时使用）')

    # ===== Parquet格式参数 =====
    parser.add_argument('--train_parquet', type=str,
                        default='/llm_reco_ssd/zhangzixing/onerec_pretrain/data/onerec_v2_data/datasets/code_01.userdoc.train.parquet',
                        help='训练数据集路径（Parquet格式）')
    parser.add_argument('--test_parquet', type=str,
                        default='/llm_reco_ssd/zhangzixing/onerec_pretrain/data/onerec_v2_data/datasets/code_01.userdoc.test.parquet',
                        help='测试数据集路径（Parquet格式）')
    parser.add_argument('--max_hist_goods_len', type=int, default=0,
                        help='历史goods最大长度（仅Parquet格式）')

    # ===== 通用参数 =====
    parser.add_argument('--save_dir', type=str, default='./checkpoints/user_doc/',
                        help='训练时使用的checkpoint路径')
    parser.add_argument('--checkpoint', type=str, default='./checkpoints/user_doc/best.pt',
                        help='推理时使用的checkpoint路径')
    parser.add_argument('--output_file', type=str, default='predictions.txt',
                        help='推理结果输出文件路径')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='训练批次大小')
    parser.add_argument('--val_batch_size', type=int, default=32,
                        help='验证批次大小')
    parser.add_argument('--infer_batch_size', type=int, default=512,
                        help='推理批次大小')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='数据加载的工作进程数')
    parser.add_argument('--max_hist_len', type=int, default=50,
                        help='历史最大长度（物品数量）')
    parser.add_argument('--max_hist_video_len', type=int, default=50,
                        help='历史video最大长度（兼容旧参数）')
    parser.add_argument('--target_type_num', type=int, default=1,
                        help='动作类型数目')
    parser.add_argument('--dropout', type=float, default=0.0,
                        help='Dropout rate')
    parser.add_argument('--num_epochs', type=int, default=200,
                        help='训练总轮数')
    parser.add_argument('--learning_rate', type=float, default=5e-4,
                        help='学习率')
    parser.add_argument('--weight_decay', type=float, default=0.1,
                        help='权重衰减')
    parser.add_argument('--warmup_ratio', type=float, default=0.01,
                        help='Warmup步数占总步数的比例')
    parser.add_argument('--min_lr_ratio', type=float, default=0.1,
                        help='最小学习率与初始学习率的比例')
    parser.add_argument('--eval_interval', type=int, default=10,
                        help='每隔多少个epoch评估一次')
    parser.add_argument('--eval_start_epoch', type=int, default=10,
                        help='从第几个epoch开始评估')
    parser.add_argument('--topk_list', type=int, nargs='+', default=[5, 10],
                        help='评估时使用的top-k列表')
    parser.add_argument('--beam_size', type=int, default=20,
                        help='Beam search的beam大小')
    parser.add_argument('--early_stop', type=int, default=0,
                        help='连续多少次eval不提升则早停，0表示不启用')
    parser.add_argument('--seed', type=int, default=2025,
                        help='随机种子')

    args = parser.parse_args()

    # 配置
    config = ModelConfig()
    config.target_type_num = args.target_type_num
    config.dropout = args.dropout
    config.beam_size = args.beam_size

    # 设置随机种子
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.mode == 'train':
        # ==================== 训练模式 ====================
        logger.info("=" * 80)
        logger.info("运行模式: 训练")
        logger.info(f"数据格式: {args.data_format}")
        logger.info("=" * 80)

        if args.data_format == 'json':
            # ===== JSON格式（LC-Rec风格） =====
            logger.info(f"使用数据集: {args.dataset}")
            logger.info("加载训练集...")
            train_dataset = JSONOneRecDataset(
                dataset=args.dataset,
                data_path=args.data_path,
                mode='train',
                max_hist_len=args.max_hist_len,
                target_type_num=args.target_type_num
            )

            logger.info("加载验证集...")
            val_dataset = JSONOneRecDataset(
                dataset=args.dataset,
                data_path=args.data_path,
                mode='valid',
                max_hist_len=args.max_hist_len,
                target_type_num=args.target_type_num
            )

            logger.info(f"训练集大小: {len(train_dataset)}")
            logger.info(f"验证集大小: {len(val_dataset)}")

            # JSON格式下直接传递数据集对象，不需要索引
            train_indices = None
            val_indices = None
            parquet_path = None

            # 构建SID→item反解映射（用于评估时反解SID为item ID）
            # item_to_sid中的value已经是int list，构建反向映射
            # 使用交互频率解决碰撞：频率最高的item作为canonical item
            from collections import Counter
            item_freq = Counter()
            for uid, item_ids in train_dataset.inters.items():
                for item_id in item_ids:
                    item_freq[str(item_id)] += 1

            sid_to_item: Dict[tuple, int] = {}
            for item_id_str, sid_codes in train_dataset.item_to_sid.items():
                sid_tuple = tuple(sid_codes)
                item_id_int = int(item_id_str)
                if sid_tuple not in sid_to_item:
                    sid_to_item[sid_tuple] = item_id_int
                else:
                    # 碰撞：选交互频率更高的
                    existing = str(sid_to_item[sid_tuple])
                    if item_freq[item_id_str] > item_freq[existing]:
                        sid_to_item[sid_tuple] = item_id_int

            logger.info(f"SID反解映射: {len(sid_to_item)} unique SIDs (from {len(train_dataset.item_to_sid)} items)")

        else:
            # ===== Parquet格式（原OneRec） =====
            parquet_path = args.train_parquet

            logger.info("Loading dataset to get indices...")
            full_dataset = OneRecDataset(
                parquet_path=parquet_path,
                max_hist_video_len=args.max_hist_video_len,
                max_hist_goods_len=args.max_hist_goods_len
            )

            # 随机划分索引
            train_size = int(0.99 * len(full_dataset))
            val_size = len(full_dataset) - train_size

            # ✅ 获取划分后的索引
            train_dataset, val_dataset = random_split(
                full_dataset,
                [train_size, val_size],
                generator=torch.Generator().manual_seed(42)
            )

            train_indices = train_dataset.indices
            val_indices = val_dataset.indices

            logger.info(f"数据集总大小: {len(full_dataset)}")
            logger.info(f"训练集大小: {len(train_indices)}")
            logger.info(f"验证集大小: {len(val_indices)}")

            sid_to_item = {}  # Parquet格式暂不支持SID反解

        # ✅ 启动DDP训练
        train_ddp(
            data_format=args.data_format,
            parquet_path=parquet_path,
            train_indices=train_indices,
            val_indices=val_indices,
            dataset_name=args.dataset if args.data_format == 'json' else None,
            data_path=args.data_path if args.data_format == 'json' else None,
            model_config=config,
            max_hist_len=args.max_hist_len,
            max_hist_video_len=args.max_hist_video_len,
            max_hist_goods_len=args.max_hist_goods_len,
            batch_size=args.batch_size,
            val_batch_size=args.val_batch_size,
            num_workers=args.num_workers,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            num_epochs=args.num_epochs,
            warmup_ratio=args.warmup_ratio,
            min_lr_ratio=args.min_lr_ratio,
            save_dir=args.save_dir,
            resume_from=None,
            eval_interval=args.eval_interval,
            eval_start_epoch=args.eval_start_epoch,
            k_list=tuple(args.topk_list),
            sid_to_item=sid_to_item,
            world_size=None,
            early_stop=args.early_stop,
        )
        
    elif args.mode == 'infer':
        # ==================== 推理模式 ====================
        logger.info("=" * 80)
        logger.info("运行模式: 推理")
        logger.info(f"数据格式: {args.data_format}")
        logger.info("=" * 80)

        logger.info(f"加载checkpoint: {args.checkpoint}")
        logger.info(f"输出文件: {args.output_file}")

        # 加载checkpoint
        checkpoint = torch.load(args.checkpoint, map_location=config.device, weights_only=False)
        model_config = checkpoint['model_config']

        # 创建模型
        model = OneRecV2(model_config).bfloat16()
        model.load_state_dict(checkpoint['model_state_dict'])

        logger.info(f"成功加载模型，epoch: {checkpoint['epoch']}, best_recall: {checkpoint.get('best_recall', checkpoint.get('best_hit_ratio', 0.0)):.4f}")

        # 创建测试数据集
        logger.info("加载测试数据集...")
        if args.data_format == 'json':
            logger.info(f"使用数据集: {args.dataset}")
            test_dataset = JSONOneRecDataset(
                dataset=args.dataset,
                data_path=args.data_path,
                mode='test',
                max_hist_len=args.max_hist_len,
                target_type_num=args.target_type_num
            )
        else:
            logger.info(f"测试数据集: {args.test_parquet}")
            test_dataset = OneRecDataset(
                parquet_path=args.test_parquet,
                max_hist_video_len=args.max_hist_video_len,
                max_hist_goods_len=args.max_hist_goods_len
            )

        logger.info(f"测试集大小: {len(test_dataset)}")

        # 创建测试数据加载器
        test_loader = DataLoader(
            test_dataset,
            batch_size=args.infer_batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
        )

        # 执行推理
        logger.info("开始推理...")
        infer(
            model=model,
            test_loader=test_loader,
            output_file=args.output_file,
        )
