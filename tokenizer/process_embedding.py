#!/usr/bin/env python3
"""
处理parquet格式embedding文件的RQ-KMEANS编码脚本
输入：包含item_id和embedding的parquet文件
输出：item_id和sid的对应关系文件
"""

import numpy as np
import torch
import pickle
import argparse
import os
import pandas as pd
from tqdm import tqdm
from res_kmeans import ResKmeans


def load_parquet_embeddings(file_path):
    """
    从parquet文件加载embeddings
    返回: item_ids (numpy array), embeddings (numpy array)
    """
    print(f"加载parquet文件: {file_path}")
    df = pd.read_parquet(file_path)

    print(f"数据形状: {df.shape}")
    print(f"列名: {df.columns.tolist()}")

    # 提取item_id和embedding
    item_ids = df['item_id'].values
    embeddings = np.array(df['embedding'].tolist())

    print(f"加载完成:")
    print(f"  Items数量: {len(item_ids)}")
    print(f"  Embedding维度: {embeddings.shape}")

    return item_ids, embeddings


def train_rq_kmeans(embeddings, n_layers=4, codebook_size=256, model_save_path=None):
    """
    训练RQ-KMEANS模型
    """
    print(f"\n开始训练RQ-KMEANS模型...")
    print(f"参数: layers={n_layers}, codebook_size={codebook_size}")

    # 转换为torch tensor
    embeddings_tensor = torch.FloatTensor(embeddings)

    # 初始化模型
    dim = embeddings.shape[-1]
    extra_kmeans_config = {
        'niter': 100,
        'verbose': True
    }
    model = ResKmeans(
        n_layers=n_layers,
        codebook_size=codebook_size,
        dim=dim,
        extra_kmeans_config=extra_kmeans_config
    )

    # 训练模型
    model.train(embeddings_tensor, verbose=True)

    # 保存模型
    if model_save_path:
        os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
        torch.save(model.state_dict(), model_save_path)
        print(f"模型已保存到: {model_save_path}")

    return model


def encode_embeddings(model, embeddings, batch_size=1024):
    """
    使用训练好的RQ-KMEANS模型编码embeddings
    """
    print(f"\n开始编码embeddings...")
    codes = []
    embeddings_tensor = torch.FloatTensor(embeddings)

    with torch.no_grad():
        for i in tqdm(range(0, len(embeddings_tensor), batch_size), desc="编码进度"):
            batch_end = min(len(embeddings_tensor), i + batch_size)
            batch_emb = embeddings_tensor[i:batch_end]

            # 编码
            batch_codes = model.encode(batch_emb)
            codes.append(batch_codes)

    # 合并所有codes
    all_codes = torch.cat(codes, dim=0)
    print(f"编码完成: {all_codes.shape}")

    return all_codes.detach().cpu().numpy()


def save_results(item_ids, codes, output_file):
    """
    保存推理结果为parquet文件
    """
    print("\n保存推理结果...")
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # 创建DataFrame
    df = pd.DataFrame({
        'item_id': item_ids,
        'code': [code.tolist() for code in codes]
    })

    # 保存为parquet
    df.to_parquet(output_file, engine='pyarrow', compression='snappy')
    print(f"推理结果已保存到: {output_file}")
    print(f"  总共 {len(df)} 个items")
    print(f"  Code维度: {len(df['code'][0])}")


def main():
    parser = argparse.ArgumentParser(description='处理parquet格式embedding文件并进行RQ-KMEANS编码')
    parser.add_argument('--input_file', required=True, help='输入的parquet embedding文件路径')
    parser.add_argument('--output_file', required=True, help='输出的parquet文件路径')
    parser.add_argument('--n_layers', type=int, default=4, help='RQ-KMEANS层数')
    parser.add_argument('--codebook_size', type=int, default=256, help='码本大小')
    parser.add_argument('--batch_size', type=int, default=1024, help='编码时的批大小')
    parser.add_argument('--model_path', required=True, help='保存/加载模型的路径')
    parser.add_argument('--load_model', action='store_true', help='是否加载预训练模型')

    args = parser.parse_args()

    # 检查输入文件
    if not os.path.exists(args.input_file):
        raise FileNotFoundError(f"输入文件不存在: {args.input_file}")

    # 加载embedding数据
    item_ids, embeddings = load_parquet_embeddings(args.input_file)

    # 训练或加载RQ-KMEANS模型
    if args.load_model and args.model_path and os.path.exists(args.model_path):
        print(f"\n加载预训练模型: {args.model_path}")
        dim = embeddings.shape[-1]
        model = ResKmeans(
            n_layers=args.n_layers,
            codebook_size=args.codebook_size,
            dim=dim
        )
        model.load_state_dict(torch.load(args.model_path))
    else:
        model = train_rq_kmeans(
            embeddings,
            n_layers=args.n_layers,
            codebook_size=args.codebook_size,
            model_save_path=args.model_path
        )

    # 编码embeddings
    codes = encode_embeddings(model, embeddings, batch_size=args.batch_size)

    # 保存推理结果
    save_results(item_ids, codes, args.output_file)

    print("\n" + "="*60)
    print("处理完成!")
    print("="*60)


if __name__ == '__main__':
    main()





