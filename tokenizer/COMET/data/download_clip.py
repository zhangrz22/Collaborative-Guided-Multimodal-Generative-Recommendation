#!/usr/bin/env python3
"""
Download CLIP ViT-L/14 model from HuggingFace to local directory.

Usage:
    python download_clip.py --save_dir /path/to/save
"""

import argparse
from transformers import CLIPModel, CLIPProcessor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="openai/clip-vit-large-patch14")
    parser.add_argument("--save_dir", type=str, required=True, help="Local directory to save model")
    args = parser.parse_args()

    print(f"Downloading {args.model_name} ...")
    model = CLIPModel.from_pretrained(args.model_name)
    processor = CLIPProcessor.from_pretrained(args.model_name)

    model.save_pretrained(args.save_dir)
    processor.save_pretrained(args.save_dir)
    print(f"Saved to {args.save_dir}")


if __name__ == "__main__":
    main()
