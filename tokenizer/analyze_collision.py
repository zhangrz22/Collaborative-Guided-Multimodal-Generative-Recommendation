#!/usr/bin/env python3
"""
分析RQ-KMEANS编码的冲突率
计算有多少个items共享相同的code
"""

import pandas as pd
import argparse
from collections import Counter
import numpy as np


def load_codes(file_path):
    """
    加载code文件
    """
    print(f"加载code文件: {file_path}")
    df = pd.read_parquet(file_path)

    print(f"数据形状: {df.shape}")
    print(f"列名: {df.columns.tolist()}")

    return df


def analyze_collision(df):
    """
    分析code冲突情况
    """
    print("\n" + "="*60)
    print("开始分析code冲突率...")
    print("="*60)

    # 将code转换为tuple以便统计
    codes = [tuple(code) for code in df['code']]

    # 统计每个code出现的次数
    code_counter = Counter(codes)

    # 基本统计
    total_items = len(df)
    unique_codes = len(code_counter)

    print(f"\n基本统计:")
    print(f"  总item数: {total_items}")
    print(f"  唯一code数: {unique_codes}")
    print(f"  冲突率: {(1 - unique_codes/total_items)*100:.2f}%")

    # 分析冲突分布
    collision_counts = list(code_counter.values())
    collision_distribution = Counter(collision_counts)

    print(f"\n冲突分布:")
    print(f"  无冲突的code数（1个item）: {collision_distribution.get(1, 0)}")

    # 统计有冲突的code
    codes_with_collision = sum(1 for count in collision_counts if count > 1)
    items_in_collision = sum(count for count in collision_counts if count > 1)

    print(f"  有冲突的code数（>1个item）: {codes_with_collision}")
    print(f"  涉及冲突的item数: {items_in_collision}")
    print(f"  涉及冲突的item占比: {items_in_collision/total_items*100:.2f}%")

    # 显示冲突程度分布
    print(f"\n冲突程度分布（每个code对应的item数）:")
    for count in sorted(collision_distribution.keys())[:10]:  # 显示前10种情况
        num_codes = collision_distribution[count]
        print(f"  {count}个item共享同一code: {num_codes}个code")

    # 找出冲突最严重的code
    max_collision = max(collision_counts)
    print(f"\n最严重冲突:")
    print(f"  最多有 {max_collision} 个items共享同一个code")

    # 找出冲突最严重的前5个code
    most_collided_codes = code_counter.most_common(5)
    print(f"\n冲突最严重的前5个code:")
    for i, (code, count) in enumerate(most_collided_codes, 1):
        print(f"  {i}. Code {code}: {count}个items")

    # 输出冲突最严重的5个code对应的item编号
    print(f"\n冲突最严重的5个code对应的item编号:")
    print("-"*60)

    # 创建code到items的映射
    code_to_items = {}
    for idx, row in df.iterrows():
        code_tuple = tuple(row['code'])
        item_id = row['item_id']
        if code_tuple not in code_to_items:
            code_to_items[code_tuple] = []
        code_to_items[code_tuple].append(item_id)

    # 输出前5个冲突最严重的code的item列表
    for i, (code, count) in enumerate(most_collided_codes, 1):
        items = code_to_items[code]
        print(f"\n{i}. Code {code} ({count}个items):")
        print(f"   Item IDs: {items}")


    # 计算平均冲突度
    avg_items_per_code = total_items / unique_codes
    print(f"\n平均统计:")
    print(f"  平均每个code对应 {avg_items_per_code:.2f} 个items")

    # 理论最大code数
    theoretical_max = 256 ** 4
    print(f"\n理论容量:")
    print(f"  4层256码本理论最大code数: {theoretical_max:,}")
    print(f"  实际使用率: {unique_codes/theoretical_max*100:.6f}%")

    print("="*60)


def main():
    parser = argparse.ArgumentParser(description='分析RQ-KMEANS编码的冲突率')
    parser.add_argument('--input_file',
                        default='/llm-reco-ssd-share/zhangrongzhou/Graduation_project/data/item_codes.parquet',
                        help='输入的code parquet文件路径')

    args = parser.parse_args()

    # 加载数据
    df = load_codes(args.input_file)

    # 分析冲突
    analyze_collision(df)


if __name__ == '__main__':
    main()


