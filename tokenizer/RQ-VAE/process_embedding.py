#!/usr/bin/env python3
"""
RQ-VAE tokenizer for item embeddings in parquet.

Input:  parquet with columns [item_id, embedding]
Output: parquet with columns [item_id, code]
"""

import argparse
import os
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from rq_vae import RQVAE


# ---------------------------------------------------------------------------
# Data I/O
# ---------------------------------------------------------------------------

def load_parquet(path: str):
    print(f"Loading: {path}")
    df = pd.read_parquet(path)
    ids = df["item_id"].to_numpy()
    emb = np.asarray(df["embedding"].tolist(), dtype=np.float32)
    print(f"  items={len(ids)}, dim={emb.shape[1]}")
    return ids, emb


def save_codes(item_ids, codes, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df = pd.DataFrame({
        "item_id": item_ids,
        "code": [c.tolist() for c in codes],
    })
    df.to_parquet(path, engine="pyarrow", compression="snappy")
    print(f"Saved: {path}")
    print(df.head())


# ---------------------------------------------------------------------------
# Collision helpers
# ---------------------------------------------------------------------------

def collision_stats(codes):
    groups = defaultdict(list)
    for i, c in enumerate(codes):
        groups[tuple(c.tolist())].append(i)
    n = len(codes)
    uniq = len(groups)
    collision_groups = [g for g in groups.values() if len(g) > 1]
    rate = (n - uniq) / max(n, 1)
    max_dup = max((len(g) for g in groups.values()), default=1)
    return rate, collision_groups, max_dup


@torch.no_grad()
def refine_collisions(model, emb, codes, device, max_rounds=20, target_rate=0.0):
    best_codes = codes.copy()
    best_rate, _, best_max = collision_stats(best_codes)
    print(f"[Refine] initial collision_rate={best_rate:.4f}, max_dup={best_max}")

    # Enable Sinkhorn on last layer only (like LETTER)
    for vq in model.rq.vq_layers[:-1]:
        vq.sk_epsilon = 0.0
    if model.rq.vq_layers[-1].sk_epsilon == 0.0:
        model.rq.vq_layers[-1].sk_epsilon = 0.003

    for r in range(1, max_rounds + 1):
        rate, collision_groups, max_dup = collision_stats(codes)
        if rate <= target_rate or not collision_groups:
            break
        print(f"[Refine] round={r} collision_rate={rate:.4f}, groups={len(collision_groups)}, max_dup={max_dup}")

        for group in collision_groups:
            t = torch.from_numpy(emb[np.array(group)]).to(device)
            new_codes = model.encode(t, use_sk=True).cpu().numpy()
            for j, idx in enumerate(group):
                codes[idx] = new_codes[j]

        new_rate, _, new_max = collision_stats(codes)
        if new_rate < best_rate:
            best_rate = new_rate
            best_max = new_max
            best_codes = codes.copy()
        if best_rate <= target_rate:
            break

    print(f"[Refine] final collision_rate={best_rate:.4f}, max_dup={best_max}")
    return best_codes


# ---------------------------------------------------------------------------
# Train / Encode
# ---------------------------------------------------------------------------

def train(model, emb, args, device):
    loader = DataLoader(
        TensorDataset(torch.from_numpy(emb)),
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=(device.type == "cuda"),
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # KMeans init
    print("Running KMeans init ...")
    model.eval()
    with torch.no_grad():
        model.init_codebooks(torch.from_numpy(emb).to(device), n_iters=args.kmeans_iters)
    print("KMeans init done.")

    best_loss = float("inf")
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = total_recon = total_vq = 0.0
        n = 0
        code_counts = torch.zeros(len(args.n_e_list), max(args.n_e_list), dtype=torch.long)

        for (batch_x,) in tqdm(loader, desc=f"Epoch {epoch}/{args.epochs}"):
            batch_x = batch_x.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            out = model(batch_x)
            out["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            bs = batch_x.shape[0]
            n += bs
            total_loss += out["loss"].item() * bs
            total_recon += out["recon_loss"].item() * bs
            total_vq += out["quant_loss"].item() * bs

            codes = out["codes"].detach().cpu()
            for layer in range(codes.shape[1]):
                code_counts[layer] += torch.bincount(codes[:, layer], minlength=args.n_e_list[layer]).cpu()

        # Per-layer usage
        usage_parts = []
        for layer in range(len(args.n_e_list)):
            c = code_counts[layer, :args.n_e_list[layer]]
            used = int((c > 0).sum().item())
            probs = c.float() / c.sum().clamp(min=1)
            nz = probs[probs > 0]
            ppl = torch.exp(-(nz * nz.log()).sum()).item() if nz.numel() > 0 else 0
            usage_parts.append(f"L{layer}:{used}/{args.n_e_list[layer]}(ppl={ppl:.0f})")

        epoch_loss = total_loss / n
        print(f"[Epoch {epoch}] loss={epoch_loss:.6f} recon={total_recon/n:.6f} "
              f"vq={total_vq/n:.6f} | {' '.join(usage_parts)}")

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            save_checkpoint(model, args.model_path, args)
            print(f"  -> saved (loss={epoch_loss:.6f})")

    # Reload best
    load_checkpoint(model, args.model_path, device)


@torch.no_grad()
def encode_all(model, emb, batch_size, device):
    loader = DataLoader(
        TensorDataset(torch.from_numpy(emb)),
        batch_size=batch_size, shuffle=False, num_workers=0,
        pin_memory=(device.type == "cuda"),
    )
    model.eval()
    all_codes = []
    for (batch_x,) in tqdm(loader, desc="Encoding"):
        batch_x = batch_x.to(device, non_blocking=True)
        all_codes.append(model.encode(batch_x).cpu())
    return torch.cat(all_codes, dim=0).numpy()


# ---------------------------------------------------------------------------
# Checkpoint
# ---------------------------------------------------------------------------

def save_checkpoint(model, path, args):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "args": vars(args)}, path)


def load_checkpoint(model, path, device):
    ckpt = torch.load(path, map_location=device)
    sd = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
    model.load_state_dict(sd, strict=False)
    print(f"Loaded: {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input_file", required=True)
    p.add_argument("--output_file", required=True)
    p.add_argument("--model_path", required=True)
    p.add_argument("--load_model", action="store_true")

    # Model
    p.add_argument("--n_e_list", type=int, nargs="+", default=[256, 256, 256, 256])
    p.add_argument("--e_dim", type=int, default=32)
    p.add_argument("--encoder_dims", type=int, nargs="+", default=[1024, 512, 256, 128])
    p.add_argument("--commitment_weight", type=float, default=0.25)
    p.add_argument("--quant_loss_weight", type=float, default=1.0)
    p.add_argument("--sk_epsilons", type=float, nargs="+", default=[0.0, 0.0, 0.0, 0.003])
    p.add_argument("--sk_iters", type=int, default=50)
    p.add_argument("--kmeans_iters", type=int, default=50)

    # Training
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch_size", type=int, default=1024)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)

    # Refine
    p.add_argument("--max_refine_rounds", type=int, default=20)
    p.add_argument("--target_collision_rate", type=float, default=0.0)
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    item_ids, emb = load_parquet(args.input_file)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RQVAE(
        in_dim=emb.shape[1],
        e_dim=args.e_dim,
        n_e_list=args.n_e_list,
        encoder_dims=args.encoder_dims,
        commitment_weight=args.commitment_weight,
        sk_epsilons=args.sk_epsilons,
        sk_iters=args.sk_iters,
        quant_loss_weight=args.quant_loss_weight,
    ).to(device)

    if args.load_model and os.path.exists(args.model_path):
        load_checkpoint(model, args.model_path, device)
    else:
        train(model, emb, args, device)

    codes = encode_all(model, emb, args.batch_size, device)
    rate, _, max_dup = collision_stats(codes)
    print(f"[Codes] collision_rate={rate:.4f}, max_dup={max_dup}")

    codes = refine_collisions(
        model, emb, codes, device,
        max_rounds=args.max_refine_rounds,
        target_rate=args.target_collision_rate,
    )
    save_codes(item_ids, codes, args.output_file)


if __name__ == "__main__":
    main()
