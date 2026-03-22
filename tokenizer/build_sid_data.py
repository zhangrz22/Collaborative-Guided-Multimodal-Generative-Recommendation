#!/usr/bin/env python3
"""
Build TIGER SID data from RQ-KMeans item codes.

Inputs:
1) item_codes.parquet with columns: item_id, code (e.g. [3, 189, 128, 88])
2) interaction sequence txt (user_id \t item_id item_id ...)
3) optional item_info.json

Outputs under data dir:
- {dataset}/merge.index.json        {item_id: ["<s_a_x>", "<s_b_y>", "<s_c_z>", "<s_d_w>"]}
- {dataset}/{dataset}.inter.json    {user_id: [item_id, item_id, ...]}
- {dataset}/item_sid_map.json       {item_id: {"code": [...], "sid_tokens": [...], "sid": "..."}}
- {dataset}/sid_item_resolution.json
                                    {sid: {"sid_tokens": [...], "candidate_item_ids": [...],
                                           "candidate_item_freq": {...}, "canonical_item_id": ...}}
- optional {dataset}/{dataset}.pretrain.json
                                    {item_id: {..., "sid": "<|sid_begin|>...<|sid_end|>"}}
"""

import json
import os
from collections import Counter, defaultdict
from typing import Dict, List

import pandas as pd


# ---------------------------------------------------------------------------
# Default in-file config (remote-friendly, relative paths)
# Run directly: python3 build_sid_data.py
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DEFAULT_DATA_ROOT = os.path.join(PROJECT_ROOT, "data")
DEFAULT_TIGER_DATA_DIR = os.path.join(DEFAULT_DATA_ROOT, "tiger_data")
DEFAULT_DATASET = "Beauty"
DEFAULT_DATASET_DIR = os.path.join(DEFAULT_TIGER_DATA_DIR, DEFAULT_DATASET)

DEFAULT_CONFIG = {
    # Put/generated item_codes.parquet here by default
    "codes_parquet": os.path.join(DEFAULT_DATASET_DIR, "item_codes.parquet"),
    # Existing user sequence file
    "interaction_txt": os.path.join(DEFAULT_DATA_ROOT, "interaction_sequences_truncated.txt"),
    # TIGER-only data output root
    "data_dir": DEFAULT_TIGER_DATA_DIR,
    "dataset": DEFAULT_DATASET,
    "layer_names": "a,b,c,d",
    # Optional
    "item_info_json": os.path.join(DEFAULT_DATA_ROOT, "item_info.json"),
    "output_pretrain_json": None,
}


def _ensure_code_list(raw_code) -> List[int]:
    if isinstance(raw_code, list):
        return [int(x) for x in raw_code]
    if hasattr(raw_code, "tolist"):
        out = raw_code.tolist()
        if isinstance(out, list):
            return [int(x) for x in out]
    raise TypeError(f"Unsupported code type: {type(raw_code)}")


def build_sid_tokens(code: List[int], layer_names: List[str]) -> List[str]:
    if len(code) != len(layer_names):
        raise ValueError(
            f"Code length ({len(code)}) != number of layer names ({len(layer_names)})"
        )
    return [f"<s_{layer}_{code[i]}>" for i, layer in enumerate(layer_names)]


def load_item_codes(codes_parquet: str, layer_names: List[str]) -> Dict[str, Dict]:
    df = pd.read_parquet(codes_parquet)
    required = {"item_id", "code"}
    if not required.issubset(set(df.columns)):
        raise ValueError(f"{codes_parquet} must contain columns {sorted(required)}")

    item_sid_map = {}
    for _, row in df.iterrows():
        item_id = str(int(row["item_id"]))
        code = _ensure_code_list(row["code"])
        sid_tokens = build_sid_tokens(code, layer_names)
        sid = "<|sid_begin|>" + "".join(sid_tokens) + "<|sid_end|>"
        item_sid_map[item_id] = {
            "code": code,
            "sid_tokens": sid_tokens,
            "sid": sid,
        }
    return item_sid_map


def build_interactions(interaction_txt: str) -> Dict[str, List[int]]:
    interactions = {}
    with open(interaction_txt, "r", encoding="utf-8") as f:
        header = f.readline().strip().split("\t")
        if len(header) < 2 or header[0] != "user_id":
            raise ValueError(
                f"Unexpected header in {interaction_txt}: {header}. Expected: user_id\\titem_sequence"
            )
        for line in f:
            line = line.strip()
            if not line:
                continue
            user_id, seq = line.split("\t")
            items = [int(x) for x in seq.split()]
            interactions[user_id] = items
    return interactions


def build_pretrain_json(item_info_json: str, item_sid_map: Dict[str, Dict], output_path: str):
    with open(item_info_json, "r", encoding="utf-8") as f:
        item_info = json.load(f)

    pretrain = {}
    missed = 0
    for item_id, info in item_info.items():
        sid_pack = item_sid_map.get(str(item_id))
        if sid_pack is None:
            missed += 1
            continue
        pretrain[str(item_id)] = {
            "title": info.get("title", ""),
            "description": info.get("description", ""),
            "categories": info.get("categories", []),
            "sid": sid_pack["sid"],
        }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(pretrain, f, ensure_ascii=False, indent=2)

    print(f"Saved pretrain file: {output_path}")
    print(f"Items written: {len(pretrain)} (missed from item_info: {missed})")


def build_sid_item_resolution(item_sid_map: Dict[str, Dict], interactions: Dict[str, List[int]]) -> Dict[str, Dict]:
    item_freq = Counter()
    for seq in interactions.values():
        item_freq.update(int(x) for x in seq)

    sid_to_items = defaultdict(list)
    for item_id, pack in item_sid_map.items():
        sid_to_items[pack["sid"]].append(int(item_id))

    sid_resolution = {}
    for sid, item_ids in sid_to_items.items():
        uniq_items = sorted(set(item_ids))
        canonical_item = sorted(uniq_items, key=lambda x: (-item_freq.get(x, 0), x))[0]
        sid_resolution[sid] = {
            "sid_tokens": item_sid_map[str(canonical_item)]["sid_tokens"],
            "candidate_item_ids": uniq_items,
            "candidate_item_freq": {str(i): int(item_freq.get(i, 0)) for i in uniq_items},
            "canonical_item_id": int(canonical_item),
        }
    return sid_resolution


def main():
    cfg = DEFAULT_CONFIG.copy()
    print("Using default config:")
    for k in sorted(cfg.keys()):
        print(f"  {k}: {cfg[k]}")

    layer_names = [x.strip() for x in cfg["layer_names"].split(",") if x.strip()]
    if not layer_names:
        raise ValueError("--layer_names is empty")

    os.makedirs(cfg["data_dir"], exist_ok=True)
    dataset_dir = os.path.join(cfg["data_dir"], cfg["dataset"])
    os.makedirs(dataset_dir, exist_ok=True)

    if not os.path.exists(cfg["codes_parquet"]):
        raise FileNotFoundError(
            f"codes_parquet not found: {cfg['codes_parquet']}\n"
            f"Please place item_codes.parquet there, or edit DEFAULT_CONFIG in this script."
        )
    if not os.path.exists(cfg["interaction_txt"]):
        raise FileNotFoundError(
            f"interaction_txt not found: {cfg['interaction_txt']}\n"
            f"Please edit DEFAULT_CONFIG in this script."
        )

    item_sid_map = load_item_codes(cfg["codes_parquet"], layer_names)
    print(f"Loaded item codes: {len(item_sid_map)}")

    # Save index in TIGER-compatible format (inside dataset dir)
    index_map = {item_id: pack["sid_tokens"] for item_id, pack in item_sid_map.items()}
    index_path = os.path.join(dataset_dir, "merge.index.json")
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(index_map, f, ensure_ascii=False, indent=2)
    print(f"Saved merge index: {index_path}")

    # Save rich sid map for debugging/analysis
    sid_map_path = os.path.join(dataset_dir, "item_sid_map.json")
    with open(sid_map_path, "w", encoding="utf-8") as f:
        json.dump(item_sid_map, f, ensure_ascii=False, indent=2)
    print(f"Saved item sid map: {sid_map_path}")

    # Build and save interaction file
    interactions = build_interactions(cfg["interaction_txt"])
    inter_path = os.path.join(dataset_dir, f"{cfg['dataset']}.inter.json")
    with open(inter_path, "w", encoding="utf-8") as f:
        json.dump(interactions, f, ensure_ascii=False)
    print(f"Saved interactions: {inter_path} (users: {len(interactions)})")

    sid_resolution = build_sid_item_resolution(item_sid_map, interactions)
    sid_resolution_path = os.path.join(dataset_dir, "sid_item_resolution.json")
    with open(sid_resolution_path, "w", encoding="utf-8") as f:
        json.dump(sid_resolution, f, ensure_ascii=False, indent=2)
    multi_sid_count = sum(
        1 for v in sid_resolution.values() if len(v["candidate_item_ids"]) > 1
    )
    print(f"Saved SID resolution: {sid_resolution_path}")
    print(f"  unique SID count: {len(sid_resolution)}")
    print(f"  SID mapped to multiple items: {multi_sid_count}")

    # Optional pretrain file
    if cfg["item_info_json"] and os.path.exists(cfg["item_info_json"]):
        pretrain_path = cfg["output_pretrain_json"] or os.path.join(
            dataset_dir, f"{cfg['dataset']}.pretrain.json"
        )
        build_pretrain_json(cfg["item_info_json"], item_sid_map, pretrain_path)
    else:
        print("Skip pretrain build: item_info_json is not set or not found.")

    sample_item = next(iter(item_sid_map))
    print("Sample:")
    print(f"  item_id={sample_item}")
    print(f"  code={item_sid_map[sample_item]['code']}")
    print(f"  sid={item_sid_map[sample_item]['sid']}")


if __name__ == "__main__":
    main()
