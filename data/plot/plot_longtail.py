#!/usr/bin/env python3
"""
绘制长尾物品分布图
"""
import json
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from pathlib import Path

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 加载交互数据
data_path = Path(__file__).parent.parent / 'tiger_data' / 'Beauty' / 'Beauty.inter.json'
with open(data_path, 'r') as f:
    interactions = json.load(f)

# 统计训练集中每个物品的交互次数
item_counts = Counter()
for user_data in interactions.values():
    # 使用leave-last-two-out: 训练集是除了最后两个item的所有item
    train_items = user_data[:-2]
    item_counts.update(train_items)

# 按频次排序
sorted_items = sorted(item_counts.items(), key=lambda x: x[1], reverse=True)
frequencies = [count for _, count in sorted_items]

# 计算20%分界点
n_items = len(frequencies)
threshold_idx = int(n_items * 0.2)

# 创建图形
fig, ax = plt.subplots(figsize=(10, 6))

# 绘制频次曲线（使用更平滑的线条）
x = np.arange(1, n_items + 1)
ax.plot(x, frequencies, linewidth=1.2, color='#2E86AB', antialiased=True)

# 标记20%分界线
ax.axvline(x=threshold_idx, color='#A23B72', linestyle='--', linewidth=2, label=f'20%分界线 (前{threshold_idx}个物品)')

# 填充区域
ax.fill_between(x[:threshold_idx], 0, frequencies[:threshold_idx], alpha=0.3, color='#F18F01', label='热门物品 (20%)')
ax.fill_between(x[threshold_idx:], 0, frequencies[threshold_idx:], alpha=0.3, color='#C73E1D', label='长尾物品 (80%)')

# 设置坐标轴
ax.set_xlabel('物品排序（按交互频次降序）', fontsize=14)
ax.set_ylabel('训练集交互次数', fontsize=14)
ax.grid(True, alpha=0.3, linestyle='--')

# 添加统计信息到图例
total_interactions = sum(frequencies)
head_interactions = sum(frequencies[:threshold_idx])
tail_interactions = sum(frequencies[threshold_idx:])
head_pct = head_interactions / total_interactions * 100
tail_pct = tail_interactions / total_interactions * 100

# 创建自定义图例
from matplotlib.patches import Patch
legend_elements = [
    plt.Line2D([0], [0], color='#A23B72', linestyle='--', linewidth=2, label=f'20%分界线 (前{threshold_idx}个物品)'),
    Patch(facecolor='#F18F01', alpha=0.3, label=f'热门物品 (20%, 交互占比{head_pct:.1f}%)'),
    Patch(facecolor='#C73E1D', alpha=0.3, label=f'长尾物品 (80%, 交互占比{tail_pct:.1f}%)')
]
ax.legend(handles=legend_elements, fontsize=12, loc='upper right')

plt.tight_layout()

# 保存图片
output_path = Path(__file__).parent / 'longtail_distribution.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"图片已保存至: {output_path}")

# 打印统计信息
print(f"\n统计信息:")
print(f"  总物品数: {n_items}")
print(f"  热门物品数 (前20%): {threshold_idx}")
print(f"  长尾物品数 (后80%): {n_items - threshold_idx}")
print(f"  热门物品交互占比: {head_pct:.2f}%")
print(f"  长尾物品交互占比: {tail_pct:.2f}%")
