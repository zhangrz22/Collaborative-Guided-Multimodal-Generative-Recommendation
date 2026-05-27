#!/usr/bin/env python3
"""
统计测试集中热门物品和长尾物品的占比
"""
import json
from pathlib import Path
from collections import Counter

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

# 按频次排序，确定热门物品和长尾物品
sorted_items = sorted(item_counts.items(), key=lambda x: x[1], reverse=True)
n_items = len(sorted_items)
threshold_idx = int(n_items * 0.2)

# 创建热门物品集合
popular_items = set([item_id for item_id, _ in sorted_items[:threshold_idx]])
longtail_items = set([item_id for item_id, _ in sorted_items[threshold_idx:]])

print(f"总物品数: {n_items}")
print(f"热门物品数 (前20%): {len(popular_items)}")
print(f"长尾物品数 (后80%): {len(longtail_items)}")
print()

# 统计测试集中的热门和长尾物品
test_popular_count = 0
test_longtail_count = 0

for user_data in interactions.values():
    # 测试集是最后一个item
    test_item = user_data[-1]

    if test_item in popular_items:
        test_popular_count += 1
    elif test_item in longtail_items:
        test_longtail_count += 1

total_test = test_popular_count + test_longtail_count

print(f"测试集统计:")
print(f"  总测试样本数: {total_test}")
print(f"  热门物品样本数: {test_popular_count}")
print(f"  长尾物品样本数: {test_longtail_count}")
print(f"  热门物品占比: {test_popular_count / total_test * 100:.2f}%")
print(f"  长尾物品占比: {test_longtail_count / total_test * 100:.2f}%")
print()

# 输出用于计算的比例
p_popular = test_popular_count / total_test
p_longtail = test_longtail_count / total_test
print(f"用于计算的比例:")
print(f"  p_popular = {p_popular:.4f}")
print(f"  p_longtail = {p_longtail:.4f}")
