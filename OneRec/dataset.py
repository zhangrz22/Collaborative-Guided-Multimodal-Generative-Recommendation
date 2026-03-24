import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
import os
import pandas as pd

class OneRecDataset(Dataset):
    """
    output：
    - his_pids: 历史pid序列（hist_pid和hist_goods_sequence拼接） (torch.LongTensor, shape: [max_hist_video_len + max_hist_goods_len])
    - his_pid_types: 历史pid类型，shape: [2, max_hist_video_len + max_hist_goods_len]
                     [0, :]==True表示为pid，[1, :]==True表示为goods
    - target_sids: 目标sid (torch.LongTensor, shape: [3])
    - target_type: 目标类型（暂时全零，标量） (torch.LongTensor)
    """

    def __init__(
            self,
            parquet_path: str,
            max_hist_video_len: Optional[int] = None,
            pad_hist_video_value: int = -1,
            max_hist_goods_len: Optional[int] = None,
            pad_hist_goods_value: int = -1
    ):
        """
        初始化Dataset

        Args:
            parquet_path: parquet文件路径
            max_hist_video_len: hist_video_sequence的最大长度，如果为None则不截断/填充
            pad_hist_video_value: hist_video_sequence的填充值，默认0
            max_hist_goods_len: hist_goods_sequence的最大长度，如果为None则不截断/填充
            pad_hist_goods_value: hist_goods_sequence的填充值，默认0
        """
        self.parquet_path = str(Path(parquet_path).absolute())  # 使用绝对路径，支持多worker

        # 设置hist_pid的配置
        self.max_hist_video_len = max_hist_video_len
        self.pad_hist_video_value = pad_hist_video_value

        # 设置hist_goods_sequence的配置
        self.max_hist_goods_len = max_hist_goods_len
        self.pad_hist_goods_value = pad_hist_goods_value

        # target_sid固定长度为4 (a, b, c, c)
        self.target_sid_len = 4

        if not Path(self.parquet_path).exists():
            raise FileNotFoundError(f"Parquet文件不存在: {self.parquet_path}")

        # 加载所有数据到内存
        worker_id = os.getpid()  # 获取进程ID，用于区分不同worker
        print(f"[Worker {worker_id}] 加载数据: {self.parquet_path}")
        self.df = pd.read_parquet(self.parquet_path)
        print(f"[Worker {worker_id}] 数据加载完成，共 {len(self.df)} 行")

        # 检测数据格式
        self.data_format = self._detect_format()
        print(f"[Worker {worker_id}] 检测到数据格式: {self.data_format}")

        # 如果max_len为None，先计算数据集中的最大长度
        if self.max_hist_video_len is None or (self.max_hist_goods_len is None and self.data_format != 'format3'):
            print(f"[Worker {worker_id}] 计算序列最大长度...")
            max_hist_video_len_actual = 0
            max_hist_goods_len_actual = 0

            for idx in range(len(self.df)):
                row = self.df.iloc[idx]
                if self.data_format == 'format1':
                    hist_video_sequence = row['hist_item_idx']
                    hist_goods_sequence = []
                elif self.data_format == 'format3':
                    hist_video_sequence = row['hist_pid']
                    hist_goods_sequence = []  # format3没有hist_goods_sequence
                else:
                    hist_video_sequence = row['hist_pid']
                    hist_goods_sequence = row['hist_goods_sequence']

                # 转换为list并计算长度
                def get_len(value):
                    if not isinstance(value, (list, np.ndarray)):
                        return 0 if pd.isna(value) else 1
                    return len(value)

                max_hist_video_len_actual = max(max_hist_video_len_actual, get_len(hist_video_sequence))
                max_hist_goods_len_actual = max(max_hist_goods_len_actual, get_len(hist_goods_sequence))

            max_hist_video_len_actual = max_hist_video_len_actual // self.target_sid_len
            max_hist_goods_len_actual = max_hist_goods_len_actual // self.target_sid_len
            # 如果未指定，使用实际最大长度
            if self.max_hist_video_len is None:
                self.max_hist_video_len = max_hist_video_len_actual
                print(f"[Worker {worker_id}] 自动设置max_hist_video_len={max_hist_video_len_actual}")

            if self.max_hist_goods_len is None and self.data_format != 'format3':
                self.max_hist_goods_len = max_hist_goods_len_actual
                print(f"[Worker {worker_id}] 自动设置max_hist_goods_len={max_hist_goods_len_actual}")

            # format3没有hist_goods_sequence，设置为0
            if self.data_format == 'format3' and self.max_hist_goods_len is None:
                self.max_hist_goods_len = 0

        # 预处理所有数据
        print(f"[Worker {worker_id}] 预处理数据...")
        self.data = []
        for idx in range(len(self.df)):
            self.data.append(self._process_row(self.df.iloc[idx]))
        print(f"[Worker {worker_id}] 数据预处理完成")

    def _detect_format(self) -> str:
        columns = set(self.df.columns)

        # 格式1: clean_onerec_dataset.py
        format1_fields = {'user_id_or_device_id', 'user_idx', 'hist_item_idx', 'target_sid'}
        # 格式2: clean_ad_goods_dataset.py
        format2_fields = {'uid', 'hist_pid', 'hist_goods_sequence', 'target_item_sid'}
        # 格式3: clean_action_dataset.py
        format3_fields = {'user_id_or_device_id', 'hist_pid', 'hist_longview', 'hist_like',
                          'hist_follow', 'hist_forward', 'hist_not_interested',
                          'target_pid', 'target_action', 'target_sid'}

        has_format1 = format1_fields.issubset(columns)
        has_format2 = format2_fields.issubset(columns)
        has_format3 = format3_fields.issubset(columns)

        # 优先检测format3（最具体）
        if has_format3:
            return 'format3'
        elif has_format1 and not has_format2:
            return 'format1'
        elif has_format2 and not has_format1:
            return 'format2'
        elif has_format1 and has_format2:
            # 两种格式都存在，优先使用format2
            print("警告: 检测到两种格式的字段都存在，使用format2")
            return 'format2'
        else:
            raise ValueError(
                f"无法识别数据格式。\n"
                f"可用字段: {columns}\n"
                f"格式1需要: {format1_fields}\n"
                f"格式2需要: {format2_fields}\n"
                f"格式3需要: {format3_fields}"
            )

    def _process_row(self, row: pd.Series) -> Dict:

        def pad_or_truncate(seq, target_len, pad_value):
            if len(seq) > target_len:
                return seq[:target_len]
            elif len(seq) < target_len:
                padding = [pad_value] * (target_len - len(seq))
                return seq + padding
            return seq

        def to_list(value):
            if not isinstance(value, (list, np.ndarray)):
                return [] if pd.isna(value) else [value]
            return list(value)

        # 根据数据格式提取字段
        if self.data_format == 'format1':
            uid = int(row.get('user_id_or_device_id', row.get('user_idx', 0)))
            hist_video_sequence = to_list(row['hist_item_idx'])
            hist_goods_sequence = []
            target_sid = to_list(row['target_sid'])
            target_action = 0
            action_masks = {}
        elif self.data_format == 'format3':
            uid = int(row['user_id_or_device_id'])
            hist_video_sequence = to_list(row['hist_pid'])
            hist_goods_sequence = []
            target_sid = to_list(row['target_sid'])
            target_action = int(row['target_action'])
            action_masks = {
                'hist_longview': to_list(row['hist_longview']),
                'hist_like': to_list(row['hist_like']),
                'hist_follow': to_list(row['hist_follow']),
                'hist_forward': to_list(row['hist_forward']),
                'hist_not_interested': to_list(row['hist_not_interested']),
            }
        else:  # format2
            uid = int(row['uid'])
            hist_video_sequence = to_list(row['hist_pid'])
            hist_goods_sequence = to_list(row['hist_goods_sequence'])
            target_sid = to_list(row['target_item_sid'])
            target_action = 0
            action_masks = {}
        # 统一处理
        hist_video_sequence = pad_or_truncate(hist_video_sequence, self.max_hist_video_len * self.target_sid_len, self.pad_hist_video_value)
        hist_goods_sequence = pad_or_truncate(hist_goods_sequence, self.max_hist_goods_len * self.target_sid_len, self.pad_hist_goods_value)
        target_sid = pad_or_truncate(target_sid, self.target_sid_len, 0)
        result_dict = {
            'uid': uid,
            'hist_video_sequence': np.array(hist_video_sequence, dtype=np.int64),
            'hist_goods_sequence': np.array(hist_goods_sequence, dtype=np.int64),
            'target_sid': np.array(target_sid, dtype=np.int64),
            'target_action': target_action
        }
        # 统一处理 action masks
        for key, mask_list in action_masks.items():
            result_dict[key] = np.array(
                pad_or_truncate(mask_list, self.max_hist_video_len, 0),
                dtype=np.int64
            )
        return result_dict

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        data = self.data[idx]

        # 将hist_pid和hist_goods_sequence拼接
        hist_video_sequence = torch.from_numpy(data['hist_video_sequence']).long()
        hist_goods_sequence = torch.from_numpy(data['hist_goods_sequence']).long()

        # 拼接成his_sids
        his_sids = torch.cat([hist_video_sequence, hist_goods_sequence], dim=0)

        # 创建his_pid_types
        total_len = self.max_hist_video_len + self.max_hist_goods_len

        if self.data_format == 'format3':
            # format3: shape [7, max_hist_video_len + max_hist_goods_len]
            # [0, :]: pid类型标记
            # [1, :]: goods类型标记（format3中全为0）
            # [2, :]: hist_longview
            # [3, :]: hist_like
            # [4, :]: hist_follow
            # [5, :]: hist_forward
            # [6, :]: hist_not_interested
            his_pid_types = torch.zeros((7, total_len), dtype=torch.long)
            his_pid_types[0, :self.max_hist_video_len] = 1  # pid类型标记

            # 映射 action mask 向量
            if 'hist_longview' in data:
                hist_longview = torch.from_numpy(data['hist_longview']).long()
                his_pid_types[2, :self.max_hist_video_len] = hist_longview
            if 'hist_like' in data:
                hist_like = torch.from_numpy(data['hist_like']).long()
                his_pid_types[3, :self.max_hist_video_len] = hist_like
            if 'hist_follow' in data:
                hist_follow = torch.from_numpy(data['hist_follow']).long()
                his_pid_types[4, :self.max_hist_video_len] = hist_follow
            if 'hist_forward' in data:
                hist_forward = torch.from_numpy(data['hist_forward']).long()
                his_pid_types[5, :self.max_hist_video_len] = hist_forward
            if 'hist_not_interested' in data:
                hist_not_interested = torch.from_numpy(data['hist_not_interested']).long()
                his_pid_types[6, :self.max_hist_video_len] = hist_not_interested
        else:
            # format1和format2: shape [2, max_hist_video_len + max_hist_goods_len]
            # [0, :]==True表示为pid，[1, :]==True表示为goods
            his_pid_types = torch.zeros((2, total_len), dtype=torch.long)
            his_pid_types[0, :self.max_hist_video_len] = 1  # 前面是pid
            if self.max_hist_goods_len > 0:  # 只有当有goods时才设置
                his_pid_types[1, self.max_hist_video_len:] = 1  # 后面是goods

        uid = torch.tensor(data['uid'], dtype=torch.long)
        target_sids = torch.from_numpy(data['target_sid']).long()
        target_type = torch.tensor(data['target_action'], dtype=torch.long)
        return uid, his_sids, his_pid_types, target_sids, target_type


class JSONOneRecDataset(Dataset):
    """
    从JSON格式加载数据（LC-Rec风格）

    数据格式：
    - {DATASET}.inter.json: {"user_id": [item_id1, item_id2, ...]}
    - merge.index.json: {"item_id": ["<s_a_3927>", "<s_b_1460>", "<s_c_6934>", "<s_c_1291>"]}

    Output:
    - uid: 用户ID (torch.LongTensor)
    - his_sids: 历史SID序列 (torch.LongTensor, shape: [max_hist_len * 4])
    - his_pid_types: 历史类型标记 (torch.LongTensor, shape: [target_type_num, max_hist_len])
    - target_sids: 目标SID (torch.LongTensor, shape: [4])
    - target_type: 目标类型 (torch.LongTensor)
    """

    def __init__(
        self,
        dataset: str,
        data_path: str = "./data",
        mode: str = "train",
        max_hist_len: int = 800,
        target_type_num: int = 1,
        pad_value: int = -1,
    ):
        """
        Args:
            dataset: 数据集名称，如"Beauty"
            data_path: 数据根目录，默认"./data"
            mode: "train", "valid", "test"
            max_hist_len: 最大历史长度（物品数量，不是token数）
            target_type_num: 目标类型数（默认1，format3为7）
            pad_value: 填充值
        """
        self.dataset = dataset
        self.data_path = Path(data_path)
        self.mode = mode
        self.max_hist_len = max_hist_len
        self.target_type_num = target_type_num
        self.pad_value = pad_value
        self.target_sid_len = 4  # a, b, c, c

        # 加载索引映射
        # Prefer dataset-local path used by CEMG pipeline:
        #   {data_path}/{dataset}/merge.index.json
        # Fallback to legacy path:
        #   {data_path}/merge/merge.index.json
        index_candidates = [
            self.data_path / dataset / "merge.index.json",
            self.data_path / "merge" / "merge.index.json",
        ]
        index_path = None
        for p in index_candidates:
            if p.exists():
                index_path = p
                break
        if index_path is None:
            raise FileNotFoundError(
                "索引文件不存在，已尝试路径:\n"
                + "\n".join([str(p) for p in index_candidates])
            )

        import json
        with open(index_path, 'r') as f:
            self.item_to_sid = json.load(f)

        # 将SID token转为ID（去除<s_a_前缀，提取数字）
        self.token_to_id = {}
        for item_id, tokens in self.item_to_sid.items():
            sids = []
            for token in tokens:
                # "<s_a_3927>" -> 3927
                token_id = int(token.split('_')[-1].rstrip('>'))
                sids.append(token_id)
            self.item_to_sid[item_id] = sids

        # 加载交互数据
        inter_path = self.data_path / dataset / f"{dataset}.inter.json"
        if not inter_path.exists():
            raise FileNotFoundError(f"交互文件不存在: {inter_path}")

        with open(inter_path, 'r') as f:
            self.inters = json.load(f)

        # 处理数据
        self.data = self._process_data()

        worker_id = os.getpid()
        print(f"[Worker {worker_id}] JSONOneRecDataset初始化完成:")
        print(f"  - Dataset: {dataset}")
        print(f"  - Mode: {mode}")
        print(f"  - 样本数: {len(self.data)}")
        print(f"  - Max history length: {max_hist_len}")

    def _process_data(self) -> List[Dict]:
        """处理数据，按照LC-Rec风格划分train/valid/test"""
        data = []

        for uid, item_ids in self.inters.items():
            if len(item_ids) < 2:
                continue  # 至少需要2个物品（1个历史+1个目标）

            # 根据mode划分数据
            if self.mode == 'train':
                # 训练集：使用前n-2个物品，每个位置生成一个样本
                # 与LC-Rec一致：最后2个留给valid和test
                items = item_ids[:-2]
                for i in range(1, len(items)):
                    history = items[:i]
                    target = items[i]
                    data.append({
                        'uid': int(uid),
                        'history': history,
                        'target': target
                    })

            elif self.mode == 'valid':
                # 验证集：倒数第2个作为目标
                if len(item_ids) < 2:
                    continue
                history = item_ids[:-2]
                target = item_ids[-2]
                data.append({
                    'uid': int(uid),
                    'history': history,
                    'target': target
                })

            elif self.mode == 'test':
                # 测试集：最后1个作为目标
                history = item_ids[:-1]
                target = item_ids[-1]
                data.append({
                    'uid': int(uid),
                    'history': history,
                    'target': target
                })

        return data

    def _item_to_sids(self, item_id: int) -> List[int]:
        """将ItemID转换为4个SID"""
        item_key = str(item_id)
        if item_key not in self.item_to_sid:
            # 如果找不到，返回全0
            return [0, 0, 0, 0]
        return self.item_to_sid[item_key]

    def _pad_or_truncate(self, seq: List, target_len: int, pad_value: int) -> List:
        """截断或填充序列"""
        if len(seq) > target_len:
            return seq[-target_len:]  # 保留最近的
        elif len(seq) < target_len:
            padding = [pad_value] * (target_len - len(seq))
            return padding + seq
        return seq

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        sample = self.data[idx]

        # 处理历史序列
        history_items = sample['history']
        if self.max_hist_len > 0:
            history_items = history_items[-self.max_hist_len:]

        # 将ItemID转为SID序列
        history_sids = []
        for item_id in history_items:
            sids = self._item_to_sids(item_id)
            history_sids.extend(sids)

        # 填充历史序列
        target_len = self.max_hist_len * self.target_sid_len
        history_sids = self._pad_or_truncate(history_sids, target_len, self.pad_value)

        # 处理目标
        target_sids = self._item_to_sids(sample['target'])

        # 创建his_pid_types
        # shape: [target_type_num, max_hist_len]
        # 对于标准SeqRec任务，target_type_num=1，[0, :]==1表示都是video类型
        his_pid_types = torch.zeros((self.target_type_num, self.max_hist_len), dtype=torch.long)
        his_pid_types[0, :] = 1  # 所有历史都标记为type 0（video）

        # 转换为tensor
        uid = torch.tensor(sample['uid'], dtype=torch.long)
        his_sids = torch.tensor(history_sids, dtype=torch.long)
        target_sids = torch.tensor(target_sids, dtype=torch.long)
        target_type = torch.tensor(0, dtype=torch.long)  # 默认type 0

        return uid, his_sids, his_pid_types, target_sids, target_type
