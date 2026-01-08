import json
import torch
import torch.nn.functional as F
from torch import Tensor
from transformers import AutoTokenizer, AutoModel
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

# Remote paths
model_path = '/llm-reco-ssd-share/zhangrongzhou/Graduation_project/base_model/Qwen3-Embedding-8B'
input_file = '/llm-reco-ssd-share/zhangrongzhou/Graduation_project/data/item_text_descriptions.json'
output_file = '/llm-reco-ssd-share/zhangrongzhou/Graduation_project/data/item_embeddings.parquet'

print("="*60)
print("Loading Qwen3-Embedding-8B model...")
print("="*60)

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side='left')
model = AutoModel.from_pretrained(
    model_path,
    attn_implementation="flash_attention_2",
    torch_dtype=torch.float16,
    device_map="auto"
)
model.eval()

print(f"Model loaded successfully on device: {model.device}")
print(f"Model dtype: {model.dtype}")

def last_token_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
    """Extract embeddings using last token pooling"""
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]

print("\n" + "="*60)
print("Loading text descriptions...")
print("="*60)

with open(input_file, 'r', encoding='utf-8') as f:
    item_descriptions = json.load(f)

print(f"Loaded {len(item_descriptions)} item descriptions")

# Prepare data for batch processing
print("\n" + "="*60)
print("Generating embeddings...")
print("="*60)

item_ids = []
texts = []
for item_id, description in item_descriptions.items():
    item_ids.append(int(item_id))
    texts.append(description)

# Batch processing parameters
batch_size = 32
max_length = 8192
all_embeddings = []

# Process in batches with progress bar
with torch.no_grad():
    for i in tqdm(range(0, len(texts), batch_size), desc="Processing batches"):
        batch_texts = texts[i:i + batch_size]

        # Tokenize
        batch_dict = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        batch_dict = {k: v.to(model.device) for k, v in batch_dict.items()}

        # Get embeddings
        outputs = model(**batch_dict)
        embeddings = last_token_pool(outputs.last_hidden_state, batch_dict['attention_mask'])

        # Normalize embeddings
        embeddings = F.normalize(embeddings, p=2, dim=1)

        # Move to CPU and convert to list
        embeddings_cpu = embeddings.cpu().float().numpy()
        all_embeddings.extend(embeddings_cpu)

print(f"\nGenerated {len(all_embeddings)} embeddings")
print(f"Embedding dimension: {all_embeddings[0].shape[0]}")

# Save to parquet file
print("\n" + "="*60)
print("Saving embeddings to parquet file...")
print("="*60)

# Create DataFrame
df = pd.DataFrame({
    'item_id': item_ids,
    'embedding': [emb.tolist() for emb in all_embeddings]
})

# Save as parquet
df.to_parquet(output_file, engine='pyarrow', compression='snappy')

print(f"Successfully saved embeddings to: {output_file}")
print(f"Total items: {len(df)}")
print(f"Embedding dimension: {len(df['embedding'][0])}")
print("="*60)
