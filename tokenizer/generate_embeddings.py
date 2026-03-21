import json
import torch
import torch.nn.functional as F
from torch import Tensor
from transformers import AutoTokenizer, AutoModel
import pandas as pd
from tqdm import tqdm
import numpy as np

# --- 配置路径 ---
model_path = '/llm-reco-ssd-share/zhangrongzhou/Graduation_project/base_model/Qwen3-Embedding-8B'
input_file = '/llm-reco-ssd-share/zhangrongzhou/Graduation_project/data/item_text_descriptions.json'
output_file = '/llm-reco-ssd-share/zhangrongzhou/Graduation_project/data/item_embeddings.parquet'

# --- 任务参数 ---
INSTRUCTION = "Represent the item description for recommendation: "
BATCH_SIZE = 16  # 8B模型显存占用较大，如果16显存够可以调大到32
MAX_LENGTH = 2048

print("="*60)
print(f"Loading Qwen3-Embedding-8B from: {model_path}")
print("="*60)

# 1. 加载 Tokenizer 和 Model
tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side='left')
model = AutoModel.from_pretrained(
    model_path,
    attn_implementation="flash_attention_2", # 确保已安装 flash-attn
    torch_dtype=torch.float16,
    device_map="auto"
)
model.eval()

def last_token_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
    """使用 Last Token Pooling 提取特征向量"""
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]

# 2. 加载数据
print("\nLoading text descriptions...")
with open(input_file, 'r', encoding='utf-8') as f:
    item_descriptions = json.load(f)

item_ids = []
texts = []
for item_id, description in item_descriptions.items():
    item_ids.append(int(item_id))
    # 关键：在这里加入任务指令前缀
    texts.append(INSTRUCTION + str(description))

print(f"Total items to process: {len(texts)}")

# 3. 批量处理
print("\n" + "="*60)
print(f"Generating embeddings (Instruction: '{INSTRUCTION}')")
print(f"Max Length: {MAX_LENGTH}, Normalize: False")
print("="*60)

all_embeddings = []

with torch.no_grad():
    for i in tqdm(range(0, len(texts), BATCH_SIZE), desc="Processing Batches"):
        batch_texts = texts[i:i + BATCH_SIZE]

        # Tokenize
        batch_dict = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt",
        )
        # 将 input_ids 等移动到模型所在的 device
        batch_dict = {k: v.to(model.device) for k, v in batch_dict.items()}

        # 正向传播
        outputs = model(**batch_dict)
        
        # 提取池化后的向量
        embeddings = last_token_pool(outputs.last_hidden_state, batch_dict['attention_mask'])

        # 转为 float32 并移至 CPU 存储 (不进行归一化)
        embeddings_cpu = embeddings.cpu().float().numpy()
        all_embeddings.extend(embeddings_cpu)
        
        # 显存清理辅助
        del outputs, batch_dict
        if i % 100 == 0:
            torch.cuda.empty_cache()

# 4. 保存结果
print("\n" + "="*60)
print("Saving to Parquet...")
print("="*60)

# 创建 DataFrame 并保存
# 注意：直接存储 list 格式的 embedding 
df = pd.DataFrame({
    'item_id': item_ids,
    'embedding': [emb.tolist() for emb in all_embeddings]
})

df.to_parquet(output_file, engine='pyarrow', compression='snappy')

print(f"Success! Saved {len(df)} items.")
print(f"Output: {output_file}")
print(f"Final Dimension: {len(df['embedding'][0])}")
print("="*60)