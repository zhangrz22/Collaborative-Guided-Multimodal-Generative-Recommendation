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


def calculate_pos_index(preds, labels, maxk=20):
    preds = preds.detach().cpu()
    labels = labels.detach().cpu()
    pos_index = torch.zeros((preds.shape[0], maxk), dtype=torch.bool)
    for i in range(preds.shape[0]):
        cur_label = labels[i].tolist()
        for j in range(maxk):
            if preds[i, j].tolist() == cur_label:
                pos_index[i, j] = True
                break
    return pos_index


def recall_at_k(pos_index, k):
    return pos_index[:, :k].any(dim=1).float()


def ndcg_at_k(pos_index, k):
    ranks = torch.arange(1, pos_index.shape[-1] + 1, dtype=torch.float32)
    discounts = (1.0 / torch.log2(ranks + 1.0)).unsqueeze(0).expand_as(pos_index.float())
    dcg = torch.where(pos_index, discounts, torch.zeros_like(discounts))
    return dcg[:, :k].sum(dim=1).float()


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
def evaluate(model, loader, topk_list, beam_size, device, prefix_allowed_tokens_fn):
    model.eval()
    recalls_sum = {f"Recall@{k}": 0.0 for k in topk_list}
    ndcgs_sum = {f"NDCG@{k}": 0.0 for k in topk_list}
    total_samples = 0

    for batch in tqdm(loader, desc="Eval"):
        input_ids = batch["history"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        labels = batch["target"].to(device, non_blocking=True)

        preds = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            num_beams=beam_size,
            prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
        )
        preds = preds[:, 1:]
        preds = preds.reshape(input_ids.shape[0], beam_size, -1)
        pos_index = calculate_pos_index(preds, labels, maxk=beam_size)
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

    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--num_decoder_layers", type=int, default=4)
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--d_ff", type=int, default=1024)
    parser.add_argument("--num_heads", type=int, default=6)
    parser.add_argument("--d_kv", type=int, default=64)
    parser.add_argument("--dropout_rate", type=float, default=0.1)

    parser.add_argument("--batch_size", type=int, default=96)
    parser.add_argument("--max_len", type=int, default=50)
    parser.add_argument("--beam_size", type=int, default=20)
    parser.add_argument("--topk_list", type=int, nargs="+", default=[5, 10, 20])
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
    inter_path = os.path.join(args.data_path, args.dataset, args.inter_file)
    index_path = os.path.join(args.data_path, args.index_file)

    tokenizer = create_sid_tokenizer(args.base_model, index_path)

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
        model, test_loader, args.topk_list, args.beam_size, device, prefix_allowed_tokens_fn
    )
    print("Test Recall:", recalls)
    print("Test NDCG:", ndcgs)


if __name__ == "__main__":
    main()
