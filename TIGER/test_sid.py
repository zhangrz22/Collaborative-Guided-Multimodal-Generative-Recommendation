import argparse
import json
import os
import random
from typing import Optional, Dict, List

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import T5Config, T5ForConditionalGeneration

from dataloader_sid import build_sid_dataloader
from dataset_sid import SIDDataset, create_sid_tokenizer


class TIGER(nn.Module):
    def __init__(self, config: T5Config):
        super().__init__()
        self.model = T5ForConditionalGeneration(config)

    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        num_beams: int = 20,
        prefix_allowed_tokens_fn=None,
    ):
        return self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=5,
            min_length=5,
            num_beams=num_beams,
            num_return_sequences=num_beams,
            prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
            early_stopping=True,
        )


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def calculate_pos_index_by_item(pred_item_ids, label_item_ids, maxk=20):
    pred_item_ids = pred_item_ids.detach().cpu()[:, :maxk]
    label_item_ids = label_item_ids.detach().cpu().unsqueeze(1)
    return pred_item_ids.eq(label_item_ids)


def recall_at_k(pos_index, k):
    return pos_index[:, :k].any(dim=1).float()


def ndcg_at_k(pos_index, k):
    ranks = torch.arange(1, pos_index.shape[-1] + 1, dtype=torch.float32)
    discounts = (1.0 / torch.log2(ranks + 1.0)).unsqueeze(0).expand_as(pos_index.float())
    dcg = torch.where(pos_index, discounts, torch.zeros_like(discounts))
    return dcg[:, :k].sum(dim=1).float()


def load_sid_to_item_map(sid_item_path: str, tokenizer) -> Dict:
    with open(sid_item_path, "r", encoding="utf-8") as f:
        sid_resolution = json.load(f)

    sid_tuple_to_item = {}
    for _, pack in sid_resolution.items():
        sid_tokens = pack.get("sid_tokens")
        canonical_item_id = pack.get("canonical_item_id")
        if not sid_tokens or canonical_item_id is None:
            continue
        sid_ids = tuple(int(tokenizer.convert_tokens_to_ids(tok)) for tok in sid_tokens)
        sid_tuple_to_item[sid_ids] = int(canonical_item_id)
    return sid_tuple_to_item


def decode_sid_preds_to_item_ids(preds, sid_tuple_to_item: Dict, unknown_item_id: int = -1):
    preds = preds.detach().cpu()
    batch_size, beam_size, _ = preds.shape
    out = torch.full((batch_size, beam_size), unknown_item_id, dtype=torch.long)
    for i in range(batch_size):
        for j in range(beam_size):
            sid_tuple = tuple(int(x) for x in preds[i, j].tolist())
            out[i, j] = sid_tuple_to_item.get(sid_tuple, unknown_item_id)
    return out


def build_sid_trie(index_path: str, tokenizer) -> Dict:
    with open(index_path, "r", encoding="utf-8") as f:
        indices = json.load(f)

    trie: Dict = {}
    layer_token_sets: List[set] = [set(), set(), set(), set()]
    for _, sid_tokens in indices.items():
        sid_ids = [tokenizer.convert_tokens_to_ids(tok) for tok in sid_tokens]
        if len(sid_ids) != 4:
            continue
        node = trie
        for i, tid in enumerate(sid_ids):
            tid = int(tid)
            layer_token_sets[i].add(tid)
            node = node.setdefault(tid, {})
    return {
        "trie": trie,
        "layer_token_lists": [sorted(list(s)) for s in layer_token_sets],
    }


def make_prefix_allowed_tokens_fn(trie_pack, decoder_start_token_id: int):
    trie = trie_pack["trie"]
    layer_token_lists = trie_pack["layer_token_lists"]

    def _fn(batch_id, input_ids):
        seq = input_ids.tolist()
        if not seq:
            return layer_token_lists[0]
        sid_prefix = seq[1:] if seq[0] == decoder_start_token_id else seq
        plen = len(sid_prefix)
        if plen <= 0:
            return layer_token_lists[0]
        if plen >= 4:
            return [decoder_start_token_id]

        node = trie
        for tid in sid_prefix:
            if tid not in node:
                return layer_token_lists[plen]
            node = node[tid]
        allowed = list(node.keys())
        return allowed if allowed else layer_token_lists[plen]

    return _fn


@torch.no_grad()
def evaluate(model, loader, topk_list, beam_size, device, prefix_allowed_tokens_fn, sid_tuple_to_item):
    model.eval()
    recalls_sum = {f"Recall@{k}": 0.0 for k in topk_list}
    ndcgs_sum = {f"NDCG@{k}": 0.0 for k in topk_list}
    total_samples = 0

    for batch in tqdm(loader, desc="Eval"):
        input_ids = batch["history"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        labels_item = batch["target_item"].to(device, non_blocking=True)

        preds = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            num_beams=beam_size,
            prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
        )
        preds = preds[:, 1:]
        preds = preds.reshape(input_ids.shape[0], beam_size, -1)
        pred_item_ids = decode_sid_preds_to_item_ids(preds, sid_tuple_to_item)
        pos_index = calculate_pos_index_by_item(pred_item_ids, labels_item, maxk=beam_size)
        bs = input_ids.shape[0]
        total_samples += bs

        for k in topk_list:
            kk = min(k, beam_size)
            recalls_sum[f"Recall@{k}"] += recall_at_k(pos_index, kk).sum().item()
            ndcgs_sum[f"NDCG@{k}"] += ndcg_at_k(pos_index, kk).sum().item()

    if total_samples == 0:
        avg_recalls = {k: 0.0 for k in recalls_sum.keys()}
        avg_ndcgs = {k: 0.0 for k in ndcgs_sum.keys()}
        return avg_recalls, avg_ndcgs
    avg_recalls = {k: v / total_samples for k, v in recalls_sum.items()}
    avg_ndcgs = {k: v / total_samples for k, v in ndcgs_sum.items()}
    return avg_recalls, avg_ndcgs


def main():
    parser = argparse.ArgumentParser(description="Evaluate TIGER SID model")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--base_model", type=str, required=True)
    parser.add_argument("--dataset", type=str, default="Beauty")
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--inter_file", type=str, default=None)
    parser.add_argument("--index_file", type=str, default=None)
    parser.add_argument("--sid_item_file", type=str, default=None)

    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--num_decoder_layers", type=int, default=4)
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--d_ff", type=int, default=1024)
    parser.add_argument("--num_heads", type=int, default=6)
    parser.add_argument("--d_kv", type=int, default=64)
    parser.add_argument("--dropout_rate", type=float, default=0.1)

    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--max_len", type=int, default=50)
    parser.add_argument("--beam_size", type=int, default=10)
    parser.add_argument("--topk_list", type=int, nargs="+", default=[5, 10])
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    if args.inter_file is None:
        args.inter_file = f"{args.dataset}.inter.json"
    if args.index_file is None:
        args.index_file = f"{args.dataset}/merge.index.json"
    if args.sid_item_file is None:
        args.sid_item_file = f"{args.dataset}/sid_item_resolution.json"
    inter_path = os.path.join(args.data_path, args.dataset, args.inter_file)
    index_path = os.path.join(args.data_path, args.index_file)
    sid_item_path = os.path.join(args.data_path, args.sid_item_file)

    tokenizer = create_sid_tokenizer(args.base_model, index_path)
    if not os.path.exists(sid_item_path):
        raise FileNotFoundError(
            f"SID resolution file not found: {sid_item_path}. "
            "Please run tokenizer/build_sid_data.py to generate it."
        )
    sid_tuple_to_item = load_sid_to_item_map(sid_item_path, tokenizer)
    print(f"Loaded SID decode map: {len(sid_tuple_to_item)} SID -> canonical item")

    cfg = T5Config.from_pretrained(args.base_model)
    cfg.num_layers = args.num_layers
    cfg.num_decoder_layers = args.num_decoder_layers
    cfg.d_model = args.d_model
    cfg.d_ff = args.d_ff
    cfg.num_heads = args.num_heads
    cfg.d_kv = args.d_kv
    cfg.dropout_rate = args.dropout_rate
    cfg.vocab_size = len(tokenizer)
    cfg.pad_token_id = tokenizer.pad_token_id
    cfg.eos_token_id = tokenizer.eos_token_id
    cfg.decoder_start_token_id = tokenizer.pad_token_id

    trie_pack = build_sid_trie(index_path, tokenizer)
    prefix_allowed_tokens_fn = make_prefix_allowed_tokens_fn(
        trie_pack, decoder_start_token_id=cfg.decoder_start_token_id
    )

    model = TIGER(cfg)
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu"))
    model.to(device)
    model.eval()

    test_dataset = SIDDataset(inter_path, index_path, tokenizer, mode="test", max_len=args.max_len, pad_token_id=tokenizer.pad_token_id)
    test_loader = build_sid_dataloader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pad_token_id=tokenizer.pad_token_id,
    )

    recalls, ndcgs = evaluate(
        model,
        test_loader,
        args.topk_list,
        args.beam_size,
        device,
        prefix_allowed_tokens_fn,
        sid_tuple_to_item,
    )
    print("Test Recall:", recalls)
    print("Test NDCG:", ndcgs)


if __name__ == "__main__":
    main()
