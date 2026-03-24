#!/usr/bin/env python3
"""
Extract image embeddings using CLIP ViT-L/14.

Input:  item_info_with_image.json  (has "image_path" field per item)
Output: item_image_embeddings.parquet  (columns: item_id, embedding)

CLIP's image encoder outputs a single 768-dim vector per image (CLS + projection),
no manual pooling needed.

Usage:
    python extract_image_embeddings.py \
        --input_json /path/to/item_info_with_image.json \
        --output_file /path/to/item_image_embeddings.parquet \
        --clip_dir /path/to/clip-vit-large-patch14 \
        --batch_size 64 --device cuda:0
"""

import argparse
import json
import os

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor


class ImageDataset(Dataset):
    """Dataset that loads images by item_id and local path."""

    def __init__(self, item_ids, image_paths, processor):
        self.item_ids = item_ids
        self.image_paths = image_paths
        self.processor = processor

    def __len__(self):
        return len(self.item_ids)

    def __getitem__(self, idx):
        item_id = self.item_ids[idx]
        path = self.image_paths[idx]
        try:
            image = Image.open(path).convert("RGB")
            inputs = self.processor(images=image, return_tensors="pt")
            pixel_values = inputs["pixel_values"].squeeze(0)  # [3, 224, 224]
        except Exception:
            # Fallback: black image
            pixel_values = torch.zeros(3, 224, 224)
        return item_id, pixel_values


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_json", required=True, help="item_info_with_image.json")
    parser.add_argument("--output_file", required=True, help="Output parquet path")
    parser.add_argument("--clip_dir", required=True, help="Local CLIP model directory")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()

    # Load item info
    with open(args.input_json, "r") as f:
        item_info = json.load(f)

    # Filter items with valid image paths
    item_ids = []
    image_paths = []
    missing = []
    for iid, info in item_info.items():
        path = info.get("image_path", "")
        if path and os.path.exists(path):
            item_ids.append(int(iid))
            image_paths.append(path)
        else:
            missing.append(int(iid))

    print(f"Items with images: {len(item_ids)}, missing: {len(missing)}")

    # Load CLIP
    print(f"Loading CLIP from {args.clip_dir} ...")
    processor = CLIPProcessor.from_pretrained(args.clip_dir)
    model = CLIPModel.from_pretrained(args.clip_dir)
    device = torch.device(args.device)
    model = model.vision_model.to(device)  # Only need vision encoder
    visual_projection = CLIPModel.from_pretrained(args.clip_dir).visual_projection.to(device)
    model.eval()
    visual_projection.eval()
    print(f"CLIP loaded on {device}")

    # Build dataloader
    dataset = ImageDataset(item_ids, image_paths, processor)
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=(device.type == "cuda"),
    )

    # Extract embeddings
    all_ids = []
    all_embs = []

    with torch.no_grad():
        for batch_ids, pixel_values in tqdm(loader, desc="Extracting"):
            pixel_values = pixel_values.to(device)
            outputs = model(pixel_values=pixel_values)
            # CLS token pooled output -> visual projection -> 768-dim
            pooled = outputs.pooler_output  # [B, 1024] (ViT-L hidden dim)
            emb = visual_projection(pooled)  # [B, 768] (CLIP projection dim)
            emb = emb.cpu().numpy()

            all_ids.extend(batch_ids.tolist())
            all_embs.append(emb)

    all_embs = np.concatenate(all_embs, axis=0)
    print(f"Embeddings shape: {all_embs.shape}")  # (N, 768)

    # Add zero vectors for missing items
    emb_dim = all_embs.shape[1]
    if missing:
        print(f"Adding zero vectors for {len(missing)} missing items")
        all_ids.extend(missing)
        all_embs = np.concatenate([
            all_embs,
            np.zeros((len(missing), emb_dim), dtype=np.float32),
        ], axis=0)

    # Sort by item_id
    order = np.argsort(all_ids)
    all_ids = [all_ids[i] for i in order]
    all_embs = all_embs[order]

    # Save
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    df = pd.DataFrame({
        "item_id": all_ids,
        "embedding": [emb.tolist() for emb in all_embs],
    })
    df.to_parquet(args.output_file, engine="pyarrow", compression="snappy")
    print(f"Saved: {args.output_file}")
    print(df.head())


if __name__ == "__main__":
    main()
