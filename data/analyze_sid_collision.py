#!/usr/bin/env python3
"""
统计Beauty.pretrain.json中的SID冲突率
"""

import json
from collections import Counter
import re

# 文件路径
input_file = '/Users/zrz/Desktop/组会/CEMG/data/Beauty.pretrain.json'

print("="*60)
print("加载Beauty.pretrain.json文件...")
print("="*60)

# 加载数据
with open(input_file, 'r', encoding='utf-8') as f:
    data = json.load(f)

print(f"加载完成，共 {len(data)} 个items")

# 提取所有SID
print("\n提取SID...")
item_ids = []
sids = []

for item_id, item_info in data.items():
    sid = item_info.get('sid', '')
    item_ids.append(item_id)
    sids.append(sid)

print(f"提取了 {len(sids)} 个SID")

# 统计SID出现次数
print("\n统计SID冲突...")
sid_counter = Counter(sids)

# 基本统计
total_items = len(sids)
unique_sids = len(sid_counter)

print("\n" + "="*60)
print("SID冲突率统计结果")
print("="*60)
print(f"\n基本统计:")
print(f"  总item数: {total_items}")
print(f"  唯一SID数: {unique_sids}")
print(f"  冲突率: {(1 - unique_sids/total_items)*100:.2f}%")

# 分析冲突分布
collision_counts = list(sid_counter.values())
collision_distribution = Counter(collision_counts)

print(f"\n冲突分布:")
print(f"  无冲突的SID数（1个item）: {collision_distribution.get(1, 0)}")

# 统计有冲突的SID
sids_with_collision = sum(1 for count in collision_counts if count > 1)
items_in_collision = sum(count for count in collision_counts if count > 1)

print(f"  有冲突的SID数（>1个item）: {sids_with_collision}")
print(f"  涉及冲突的item数: {items_in_collision}")
print(f"  涉及冲突的item占比: {items_in_collision/total_items*100:.2f}%")

# 显示冲突程度分布
print(f"\n冲突程度分布（每个SID对应的item数）:")
for count in sorted(collision_distribution.keys())[:10]:
    num_sids = collision_distribution[count]
    print(f"  {count}个item共享同一SID: {num_sids}个SID")

# 找出冲突最严重的SID
max_collision = max(collision_counts)
print(f"\n最严重冲突:")
print(f"  最多有 {max_collision} 个items共享同一个SID")

# 找出冲突最严重的前5个SID
most_collided_sids = sid_counter.most_common(5)
print(f"\n冲突最严重的前5个SID:")
for i, (sid, count) in enumerate(most_collided_sids, 1):
    print(f"  {i}. SID {sid}: {count}个items")

# 创建SID到items的映射
sid_to_items = {}
for item_id, sid in zip(item_ids, sids):
    if sid not in sid_to_items:
        sid_to_items[sid] = []
    sid_to_items[sid].append(item_id)

# 输出前5个冲突最严重的SID的item列表
print(f"\n冲突最严重的5个SID对应的item编号:")
print("-"*60)
for i, (sid, count) in enumerate(most_collided_sids, 1):
    items = sid_to_items[sid]
    print(f"\n{i}. SID {sid} ({count}个items):")
    print(f"   Item IDs: {items[:20]}")  # 只显示前20个
    if len(items) > 20:
        print(f"   ... 还有 {len(items)-20} 个items")

print("\n" + "="*60)




