import argparse
import os
import random
from typing import Optional

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from transformers import T5Config, T5ForConditionalGeneration

from dataloader_sid import build_sid_dataloader
from dataset_sid import SIDDataset, create_sid_tokenizer


class TIGER(nn.Module):
    def __init__(self, config: T5Config):
        super().__init__()
        self.model = T5ForConditionalGeneration(config)

    @property
    def n_parameters(self):
        num_params = lambda ps: sum(p.numel() for p in ps if p.requires_grad)
        total_params = num_params(self.parameters())
        emb_params = num_params(self.model.get_input_embeddings().parameters())
        return (
            f"#Embedding parameters: {emb_params}\n"
            f"#Non-embedding parameters: {total_params - emb_params}\n"
            f"#Total trainable parameters: {total_params}\n"
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        return outputs.loss, outputs.logits

    def generate(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None, num_beams: int = 20):
        return self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=5,  # decoder_start_token + 4 SID tokens
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


def train_one_epoch(model, loader, optimizer, device, local_rank):
    model.train()
    total_loss = 0.0
    iterator = tqdm(loader, desc="Train", disable=(local_rank != 0))
    for batch in iterator:
        input_ids = batch["history"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        labels = batch["target"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        loss, _ = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / max(len(loader), 1)


@torch.no_grad()
def evaluate(model, loader, topk_list, beam_size, device, local_rank):
    model.eval()
    recalls = {f"Recall@{k}": [] for k in topk_list}
    ndcgs = {f"NDCG@{k}": [] for k in topk_list}

    iterator = tqdm(loader, desc="Eval", disable=(local_rank != 0))
    for batch in iterator:
        input_ids = batch["history"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        labels = batch["target"].to(device, non_blocking=True)

        preds = model.generate(input_ids=input_ids, attention_mask=attention_mask, num_beams=beam_size)
        preds = preds[:, 1:]  # strip decoder start token
        preds = preds.reshape(input_ids.shape[0], beam_size, -1)
        pos_index = calculate_pos_index(preds, labels, maxk=beam_size)

        for k in topk_list:
            recalls[f"Recall@{k}"].append(recall_at_k(pos_index, k))
            ndcgs[f"NDCG@{k}"].append(ndcg_at_k(pos_index, k))

    avg_recalls = {k: float(np.mean(v)) for k, v in recalls.items()}
    avg_ndcgs = {k: float(np.mean(v)) for k, v in ndcgs.items()}
    return avg_recalls, avg_ndcgs


def main():
    parser = argparse.ArgumentParser(description="Train TIGER with SID data (8-GPU DDP supported)")
    parser.add_argument("--base_model", type=str, required=True, help="Path to T5 config/model")
    parser.add_argument("--load_pretrained", type=str, default="false", choices=["true", "false"])
    parser.add_argument("--dataset", type=str, default="Beauty")
    parser.add_argument("--data_path", type=str, required=True, help="Path containing dataset and merge/index")
    parser.add_argument("--inter_file", type=str, default=None)
    parser.add_argument("--index_file", type=str, default="merge/merge.index.json")

    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--num_decoder_layers", type=int, default=4)
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--d_ff", type=int, default=1024)
    parser.add_argument("--num_heads", type=int, default=6)
    parser.add_argument("--d_kv", type=int, default=64)
    parser.add_argument("--dropout_rate", type=float, default=0.1)

    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--infer_size", type=int, default=96)
    parser.add_argument("--num_epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--early_stop", type=int, default=10)
    parser.add_argument("--eval_interval", type=int, default=1)
    parser.add_argument("--max_len", type=int, default=50)

    parser.add_argument("--beam_size", type=int, default=20)
    parser.add_argument("--topk_list", type=int, nargs="+", default=[5, 10, 20])

    parser.add_argument("--output_dir", type=str, default="./ckpt")
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size > 1
    if ddp:
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    set_seed(args.seed + local_rank)
    os.makedirs(args.output_dir, exist_ok=True)

    if args.inter_file is None:
        args.inter_file = f"{args.dataset}.inter.json"
    inter_path = os.path.join(args.data_path, args.dataset, args.inter_file)
    index_path = os.path.join(args.data_path, args.index_file)

    if local_rank == 0:
        print("=" * 80)
        print("TIGER SID Training")
        print("=" * 80)
        print(f"World size: {world_size}")
        print(f"inter_path: {inter_path}")
        print(f"index_path: {index_path}")

    tokenizer = create_sid_tokenizer(args.base_model, index_path)

    base_cfg = T5Config.from_pretrained(args.base_model)
    if args.load_pretrained.lower() != "true":
        base_cfg.num_layers = args.num_layers
        base_cfg.num_decoder_layers = args.num_decoder_layers
        base_cfg.d_model = args.d_model
        base_cfg.d_ff = args.d_ff
        base_cfg.num_heads = args.num_heads
        base_cfg.d_kv = args.d_kv
        base_cfg.dropout_rate = args.dropout_rate
    base_cfg.vocab_size = len(tokenizer)
    base_cfg.pad_token_id = tokenizer.pad_token_id
    base_cfg.eos_token_id = tokenizer.eos_token_id

    if args.load_pretrained.lower() == "true":
        pretrained = T5ForConditionalGeneration.from_pretrained(args.base_model)
        pretrained.resize_token_embeddings(len(tokenizer))
        model = TIGER(base_cfg)
        model.model.load_state_dict(pretrained.state_dict())
    else:
        model = TIGER(base_cfg)

    model.to(device)
    if local_rank == 0:
        print(model.n_parameters)

    if ddp:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank], output_device=local_rank
        )

    train_dataset = SIDDataset(inter_path, index_path, tokenizer, mode="train", max_len=args.max_len, pad_token_id=tokenizer.pad_token_id)
    valid_dataset = SIDDataset(inter_path, index_path, tokenizer, mode="valid", max_len=args.max_len, pad_token_id=tokenizer.pad_token_id)
    test_dataset = SIDDataset(inter_path, index_path, tokenizer, mode="test", max_len=args.max_len, pad_token_id=tokenizer.pad_token_id)

    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=local_rank, shuffle=True) if ddp else None

    train_loader = build_sid_dataloader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        num_workers=args.num_workers,
        pad_token_id=tokenizer.pad_token_id,
        sampler=train_sampler,
    )
    # Valid/test only rank0 evaluates, so no distributed sampler is needed.
    valid_loader = build_sid_dataloader(
        valid_dataset,
        batch_size=args.infer_size,
        shuffle=False,
        num_workers=args.num_workers,
        pad_token_id=tokenizer.pad_token_id,
    )
    test_loader = build_sid_dataloader(
        test_dataset,
        batch_size=args.infer_size,
        shuffle=False,
        num_workers=args.num_workers,
        pad_token_id=tokenizer.pad_token_id,
    )

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    best_recall10 = -1.0
    best_epoch = -1
    early_stop_counter = 0
    save_path = os.path.join(args.output_dir, f"tiger_{args.dataset}_best.pth")

    for epoch in range(args.num_epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        train_loss = train_one_epoch(model, train_loader, optimizer, device, local_rank)
        if local_rank == 0:
            print(f"Epoch {epoch + 1}/{args.num_epochs} loss={train_loss:.6f}")

        should_stop = False
        if (epoch + 1) % args.eval_interval == 0 and local_rank == 0:
            eval_model = model.module if ddp else model
            val_recalls, val_ndcgs = evaluate(
                eval_model, valid_loader, args.topk_list, args.beam_size, device, local_rank
            )
            print(f"Valid Recall: {val_recalls}")
            print(f"Valid NDCG: {val_ndcgs}")

            cur_recall10 = val_recalls.get("Recall@10", -1.0)
            if cur_recall10 > best_recall10:
                best_recall10 = cur_recall10
                best_epoch = epoch + 1
                early_stop_counter = 0
                torch.save(eval_model.state_dict(), save_path)
                print(f"New best model saved: {save_path}")
            else:
                early_stop_counter += 1
                print(f"No improvement: early_stop_counter={early_stop_counter}/{args.early_stop}")
                if early_stop_counter >= args.early_stop:
                    should_stop = True

        if ddp:
            stop_tensor = torch.tensor([1 if should_stop else 0], device=device)
            dist.broadcast(stop_tensor, src=0)
            should_stop = bool(stop_tensor.item())
        if should_stop:
            break

    if local_rank == 0 and best_epoch > 0:
        print("=" * 80)
        print(f"Evaluating best checkpoint from epoch {best_epoch}, Recall@10={best_recall10:.6f}")
        print("=" * 80)
        eval_model = model.module if ddp else model
        ckpt = torch.load(save_path, map_location=device)
        eval_model.load_state_dict(ckpt)
        test_recalls, test_ndcgs = evaluate(
            eval_model, test_loader, args.topk_list, args.beam_size, device, local_rank
        )
        print(f"Test Recall: {test_recalls}")
        print(f"Test NDCG: {test_ndcgs}")

    if ddp:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()

