import argparse
import os
import random
from typing import Optional

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

    def generate(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None, num_beams: int = 20):
        return self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=5,
            num_beams=num_beams,
            num_return_sequences=num_beams,
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
    return pos_index[:, :k].sum(dim=1).float().mean().item()


def ndcg_at_k(pos_index, k):
    ranks = torch.arange(1, pos_index.shape[-1] + 1)
    dcg = 1.0 / torch.log2(ranks + 1)
    dcg = dcg.unsqueeze(0).expand_as(pos_index.float())
    dcg = torch.where(pos_index, dcg, torch.tensor(0.0))
    return dcg[:, :k].sum(dim=1).float().mean().item()


@torch.no_grad()
def evaluate(model, loader, topk_list, beam_size, device):
    model.eval()
    recalls = {f"Recall@{k}": [] for k in topk_list}
    ndcgs = {f"NDCG@{k}": [] for k in topk_list}

    for batch in tqdm(loader, desc="Eval"):
        input_ids = batch["history"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        labels = batch["target"].to(device, non_blocking=True)

        preds = model.generate(input_ids=input_ids, attention_mask=attention_mask, num_beams=beam_size)
        preds = preds[:, 1:]
        preds = preds.reshape(input_ids.shape[0], beam_size, -1)
        pos_index = calculate_pos_index(preds, labels, maxk=beam_size)

        for k in topk_list:
            recalls[f"Recall@{k}"].append(recall_at_k(pos_index, k))
            ndcgs[f"NDCG@{k}"].append(ndcg_at_k(pos_index, k))

    avg_recalls = {k: float(np.mean(v)) for k, v in recalls.items()}
    avg_ndcgs = {k: float(np.mean(v)) for k, v in ndcgs.items()}
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

    recalls, ndcgs = evaluate(model, test_loader, args.topk_list, args.beam_size, device)
    print("Test Recall:", recalls)
    print("Test NDCG:", ndcgs)


if __name__ == "__main__":
    main()
