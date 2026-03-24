# OneRec with JSON Data Support

本文档说明如何使用JSON格式的数据（LC-Rec风格）训练和测试OneRec模型。

## 数据格式要求

### 1. 交互数据 (`data/{DATASET}/{DATASET}.inter.json`)

```json
{
  "204655": [97660, 97661, 97662, ...],
  "204656": [97668, 97669, 97670, ...]
}
```

- **格式**: `{"user_id": [item_id1, item_id2, ...]}`
- **说明**: 每个用户的交互序列，按时间顺序排列
- **ItemID**: 整数类型

### 2. SID索引映射 (`data/merge/merge.index.json`)

```json
{
  "1": ["<s_a_3927>", "<s_b_1460>", "<s_c_6934>", "<s_c_1291>"],
  "2": ["<s_a_7048>", "<s_b_5521>", "<s_c_5088>", "<s_c_2061>"]
}
```

- **格式**: `{"item_id": ["<s_a_...>", "<s_b_...>", "<s_c_...>", "<s_c_...>"]}`
- **说明**: 将ItemID映射到4层SID (a, b, c, c)
- **共享词表**: 最后两个位置共享c词表

## 数据集划分

与LC-Rec保持一致：

- **训练集 (train)**: 每个用户的前 n-2 个物品，每个位置生成一个样本
- **验证集 (valid)**: 每个用户的倒数第2个物品
- **测试集 (test)**: 每个用户的最后1个物品

例如，用户有 [item1, item2, item3, item4, item5] 5个交互:
- 训练: [item1→item2, item1,item2→item3, item1,item2,item3→item4]
- 验证: [item1,item2,item3→item4]
- 测试: [item1,item2,item3,item4→item5]

## 快速开始

### 1. 训练

使用 `run.sh` 脚本，只需修改数据集名称：

```bash
#!/bin/bash
# 修改这里切换数据集
DATASET=Beauty  # 可选: Baby, Beauty, Cell_Phones_and_Accessories, 等
```

运行训练：

```bash
./run.sh
```

或手动运行：

```bash
python main.py \
    --mode train \
    --data_format json \
    --dataset Beauty \
    --data_path ./data \
    --save_dir ./checkpoints/Beauty/ \
    --batch_size 128 \
    --val_batch_size 32 \
    --max_hist_len 800 \
    --num_workers 4
```

### 2. 测试

使用 `run_test.sh` 脚本：

```bash
#!/bin/bash
DATASET=Beauty  # 修改数据集名称
```

运行测试：

```bash
./run_test.sh
```

或手动运行：

```bash
python main.py \
    --mode infer \
    --data_format json \
    --dataset Beauty \
    --data_path ./data \
    --checkpoint ./checkpoints/Beauty/best.pt \
    --output_file ./results/Beauty/predictions.pkl \
    --infer_batch_size 64 \
    --max_hist_len 800
```

## 可用数据集

当前 `data/` 目录下的数据集：

- Beauty
- Baby
- Cell_Phones_and_Accessories
- Grocery_and_Gourmet_Food
- Health_and_Personal_Care
- Home_and_Kitchen
- Pet_Supplies
- Sports_and_Outdoors
- Tools_and_Home_Improvement
- Toys_and_Games

## 命令行参数说明

### 数据相关

- `--data_format`: 数据格式，`json` 或 `parquet`
- `--dataset`: 数据集名称（JSON格式时使用）
- `--data_path`: 数据根目录，默认 `./data`
- `--max_hist_len`: 历史序列最大长度（物品数量，非token数），默认800

### 训练相关

- `--batch_size`: 训练批次大小，默认128
- `--val_batch_size`: 验证批次大小，默认32
- `--num_workers`: 数据加载进程数，默认4
- `--save_dir`: checkpoint保存路径

### 推理相关

- `--checkpoint`: 模型checkpoint路径
- `--output_file`: 预测结果输出文件
- `--infer_batch_size`: 推理批次大小，默认64

## 模型配置

默认配置（`model.py` 的 `ModelConfig`）：

```python
semantic_token_num: 4        # SID长度 (a, b, c, c)
num_vocab_layers: 3          # 独立词表数 (a, b, c)
position_to_vocab: (0,1,2,2) # 位置到词表映射
vocab_size: 8192             # 每个词表大小
d_model: 2048                # 模型维度
n_layers: 16                 # Transformer层数
n_heads: 16                  # 注意力头数
max_his_len: 800             # 最大历史长度
beam_size: 128               # Beam Search大小
```

## 输出格式

训练过程会保存：

- `checkpoints/{DATASET}/best.pt`: 最佳模型
- `checkpoints/{DATASET}/latest.pt`: 最新模型
- `logs/train_{DATASET}_{timestamp}.log`: 训练日志
- `checkpoints/{DATASET}/tensorboard/`: TensorBoard日志

测试结果：

- `results/{DATASET}/predictions.pkl`: 预测结果（pickle格式）
- 包含 hit@k, precision, recall, NDCG 等指标

## 与原Parquet格式对比

| 特性 | JSON格式 | Parquet格式 |
|------|----------|-------------|
| 数据源 | LC-Rec风格 | 原OneRec |
| 数据划分 | train/valid/test固定 | 需手动random_split |
| 切换数据集 | 修改DATASET变量 | 修改parquet路径 |
| 灵活性 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |
| 读取速度 | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |

## 注意事项

1. **SID格式**: 确保 `merge.index.json` 中的SID格式为 `"<s_a_3927>"` 且数值在 0-8191 范围内
2. **历史长度**: `max_hist_len` 是物品数量，实际token数为 `max_hist_len * 4`
3. **GPU数量**: 默认使用所有可见GPU进行DDP训练
4. **内存占用**: 数据会全部加载到内存，确保足够的RAM

## 故障排查

### 错误：FileNotFoundError

```
FileNotFoundError: 交互文件不存在: ./data/Beauty/Beauty.inter.json
```

**解决**: 检查数据文件路径和文件名是否正确

### 错误：KeyError in item_to_sid

```
KeyError: '12345'
```

**解决**: 某个ItemID在 `merge.index.json` 中不存在，检查数据一致性

### 错误：CUDA out of memory

**解决**: 降低 `batch_size` 或 `max_hist_len`

## 性能优化建议

1. **批次大小**: 根据GPU显存调整 `batch_size`（8×V100建议128-256）
2. **工作进程**: `num_workers` 设为CPU核心数的1/4到1/2
3. **历史长度**: 较短的历史可以加快训练（如400-800）
4. **混合精度**: 默认使用 `bfloat16`，降低显存占用
