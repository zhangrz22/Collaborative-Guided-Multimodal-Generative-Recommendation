#!/usr/bin/env python3
"""
RQ-VAE tokenizer for item embeddings in parquet.

Input parquet columns:
- item_id
- embedding (list/array)

Output parquet columns:
- item_id
- code (list of RQ layer indices)
"""

import argparse
import os
from typing import Tuple
from collections import defaultdict
from contextlib import nullcontext

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from rq_vae import RQVAE


def load_parquet_embeddings(file_path: str) -> Tuple[np.ndarray, np.ndarray]:
    print(f"Loading parquet: {file_path}")
    df = pd.read_parquet(file_path)
    required_cols = {"item_id", "embedding"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"Missing columns: {required_cols - set(df.columns)}")

    item_ids = df["item_id"].to_numpy()
    embeddings = np.asarray(df["embedding"].tolist(), dtype=np.float32)
    print(f"Rows: {len(item_ids)}, Embedding shape: {embeddings.shape}")
    return item_ids, embeddings


def build_model(args, input_dim: int, device: torch.device):
    model = RQVAE(
        input_dim=input_dim,
        hidden_dim=args.hidden_dim,
        latent_dim=args.latent_dim,
        num_layers=args.n_layers,
        codebook_size=args.codebook_size,
        commitment_weight=args.commitment_weight,
        kl_weight=args.kl_weight,
        ema=args.ema,
        ema_decay=args.ema_decay,
        restart_unused_codes=args.restart_unused_codes,
        dead_code_threshold=args.dead_code_threshold,
    ).to(device)
    return model


def summarize_code_usage(counts: torch.Tensor):
    """
    counts: [n_layers, codebook_size]
    """
    out = []
    for layer in range(counts.shape[0]):
        c = counts[layer]
        used = int((c > 0).sum().item())
        usage = used / float(c.shape[0])
        probs = c.float() / max(float(c.sum().item()), 1.0)
        nz = probs[probs > 0]
        perplexity = torch.exp(-(nz * torch.log(nz)).sum()).item() if nz.numel() > 0 else 0.0
        out.append((usage, perplexity, used))
    return out


def train_model(model: RQVAE, emb: np.ndarray, args, device: torch.device):
    x = torch.from_numpy(emb)
    loader = DataLoader(
        TensorDataset(x),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.01)

    if args.kmeans_init:
        print(f"Running residual kmeans init, iters={args.kmeans_iters} ...")
        init_x = x.to(device, non_blocking=True)
        with torch.no_grad():
            z = model.encode_to_latent(init_x)
            model.rq.init_codebooks_kmeans(z, n_iters=args.kmeans_iters)
        print("Kmeans init done.")

    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda" and args.amp))
    best_loss = float("inf")
    best_epoch = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        stats = {"loss": 0.0, "recon": 0.0, "vq": 0.0, "commit": 0.0, "kl": 0.0}
        count = 0
        code_counts = torch.zeros(
            (args.n_layers, args.codebook_size), dtype=torch.long, device="cpu"
        )

        for (batch_x,) in tqdm(loader, desc=f"Epoch {epoch}/{args.epochs}"):
            batch_x = batch_x.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            amp_ctx = (
                torch.amp.autocast("cuda", enabled=True)
                if (device.type == "cuda" and args.amp)
                else nullcontext()
            )
            with amp_ctx:
                out = model(batch_x)
                loss = out["loss"]

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            bs = batch_x.shape[0]
            count += bs
            stats["loss"] += out["loss"].item() * bs
            stats["recon"] += out["recon_loss"].item() * bs
            stats["vq"] += out["codebook_loss"].item() * bs
            stats["commit"] += out["commit_loss"].item() * bs
            stats["kl"] += out["kl_loss"].item() * bs

            codes = out["codes"].detach().cpu()
            for layer in range(codes.shape[1]):
                binc = torch.bincount(codes[:, layer], minlength=args.codebook_size)
                code_counts[layer] += binc

        scheduler.step()

        epoch_loss = stats["loss"] / count
        usage = summarize_code_usage(code_counts)
        usage_str = " ".join(
            [
                f"L{i}:used={used}/{args.codebook_size}({u*100:.1f}%),ppl={ppl:.1f}"
                for i, (u, ppl, used) in enumerate(usage)
            ]
        )
        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"[Epoch {epoch}] "
            f"loss={epoch_loss:.6f} "
            f"recon={stats['recon']/count:.6f} "
            f"vq={stats['vq']/count:.6f} "
            f"commit={stats['commit']/count:.6f} "
            f"kl={stats['kl']/count:.6f} "
            f"lr={current_lr:.2e} "
            f"| {usage_str}"
        )

        # Save best checkpoint
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            best_epoch = epoch
            save_checkpoint(model, args.model_path, args)
            print(f"  -> Best model saved (epoch {epoch}, loss={epoch_loss:.6f})")

    print(f"Training done. Best epoch={best_epoch}, best_loss={best_loss:.6f}")


@torch.no_grad()
def encode_embeddings(model: RQVAE, emb: np.ndarray, batch_size: int, device: torch.device):
    x = torch.from_numpy(emb)
    loader = DataLoader(
        TensorDataset(x),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )
    all_codes = []
    model.eval()
    for (batch_x,) in tqdm(loader, desc="Encoding"):
        batch_x = batch_x.to(device, non_blocking=True)
        codes = model.encode(batch_x)
        all_codes.append(codes.cpu())
    return torch.cat(all_codes, dim=0).numpy()


def save_codes(item_ids: np.ndarray, codes: np.ndarray, output_file: str):
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    df = pd.DataFrame(
        {
            "item_id": item_ids,
            "code": [code.tolist() for code in codes],
        }
    )
    df.to_parquet(output_file, engine="pyarrow", compression="snappy")
    print(f"Saved codes: {output_file}")
    print(df.head())


def collision_stats(codes: np.ndarray):
    tuples = [tuple(c.tolist()) for c in codes]
    groups = defaultdict(list)
    for i, key in enumerate(tuples):
        groups[key].append(i)
    collision_groups = [idxs for idxs in groups.values() if len(idxs) > 1]
    n = len(codes)
    uniq = len(groups)
    rate = (n - uniq) / max(n, 1)
    max_dup = max((len(v) for v in groups.values()), default=1)
    return rate, collision_groups, max_dup


@torch.no_grad()
def refine_collisions(
    model: RQVAE,
    emb: np.ndarray,
    codes: np.ndarray,
    args,
    device: torch.device,
):
    best_codes = codes.copy()
    best_rate, _, best_max_dup = collision_stats(best_codes)
    print(
        f"[Refine] initial collision_rate={best_rate:.4f}, max_dup={best_max_dup}, "
        f"target={args.target_collision_rate:.4f}"
    )

    if best_rate <= args.target_collision_rate:
        return best_codes

    for round_idx in range(1, args.max_refine_rounds + 1):
        rate, collision_groups, max_dup = collision_stats(codes)
        print(
            f"[Refine] round={round_idx} before: collision_rate={rate:.4f}, "
            f"groups={len(collision_groups)}, max_dup={max_dup}"
        )
        if rate <= args.target_collision_rate or not collision_groups:
            break

        collision_indices = sorted({i for group in collision_groups for i in group})
        if not collision_indices:
            break
        group_tensor = torch.from_numpy(emb[np.asarray(collision_indices)]).to(device, non_blocking=True)
        new_codes = model.encode(
            group_tensor,
            use_sk=True,
            sk_epsilon=args.refine_sk_epsilon,
            sk_iters=args.refine_sk_iters,
        ).cpu().numpy()
        for j, idx in enumerate(collision_indices):
            codes[idx] = new_codes[j]

        new_rate, _, new_max_dup = collision_stats(codes)
        print(
            f"[Refine] round={round_idx} after : collision_rate={new_rate:.4f}, "
            f"max_dup={new_max_dup}"
        )
        if new_rate < best_rate:
            best_rate = new_rate
            best_max_dup = new_max_dup
            best_codes = codes.copy()
            print(f"[Refine] new best collision_rate={best_rate:.4f}")
        if best_rate <= args.target_collision_rate:
            break

    print(
        f"[Refine] best collision_rate={best_rate:.4f}, best_max_dup={best_max_dup}"
    )
    return best_codes


def save_checkpoint(model: RQVAE, path: str, args):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    ckpt = {
        "state_dict": model.state_dict(),
        "config": {
            "hidden_dim": args.hidden_dim,
            "latent_dim": args.latent_dim,
            "n_layers": args.n_layers,
            "codebook_size": args.codebook_size,
            "commitment_weight": args.commitment_weight,
            "kl_weight": args.kl_weight,
            "ema": args.ema,
            "ema_decay": args.ema_decay,
            "restart_unused_codes": args.restart_unused_codes,
            "dead_code_threshold": args.dead_code_threshold,
            "kmeans_init": args.kmeans_init,
            "kmeans_iters": args.kmeans_iters,
            "refine_collisions": args.refine_collisions,
            "max_refine_rounds": args.max_refine_rounds,
            "target_collision_rate": args.target_collision_rate,
            "refine_sk_epsilon": args.refine_sk_epsilon,
            "refine_sk_iters": args.refine_sk_iters,
        },
    }
    torch.save(ckpt, path)
    print(f"Saved model: {path}")


def load_checkpoint(model: RQVAE, path: str, device: torch.device):
    ckpt = torch.load(path, map_location=device)
    state_dict = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"Missing keys when loading checkpoint: {len(missing)}")
    if unexpected:
        print(f"Unexpected keys when loading checkpoint: {len(unexpected)}")
    print(f"Loaded model: {path}")


def parse_args():
    parser = argparse.ArgumentParser(description="RQ-VAE tokenizer for item embeddings")
    parser.add_argument("--input_file", required=True, help="Input parquet file")
    parser.add_argument("--output_file", required=True, help="Output parquet with item_id/code")
    parser.add_argument("--model_path", required=True, help="Model checkpoint path")
    parser.add_argument("--load_model", action="store_true", help="Load model instead of training")

    parser.add_argument("--n_layers", type=int, default=4)
    parser.add_argument("--codebook_size", type=int, default=256)
    parser.add_argument("--hidden_dim", type=int, default=1024)
    parser.add_argument("--latent_dim", type=int, default=256)
    parser.add_argument("--commitment_weight", type=float, default=0.25)
    parser.add_argument("--kl_weight", type=float, default=0.0)
    parser.add_argument("--ema", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--ema_decay", type=float, default=0.95)
    parser.add_argument("--restart_unused_codes", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dead_code_threshold", type=float, default=10.0)
    parser.add_argument("--kmeans_init", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--kmeans_iters", type=int, default=25)

    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--refine_collisions", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max_refine_rounds", type=int, default=2)
    parser.add_argument("--target_collision_rate", type=float, default=0.10)
    parser.add_argument("--refine_sk_epsilon", type=float, default=0.003)
    parser.add_argument("--refine_sk_iters", type=int, default=50)
    return parser.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if not os.path.exists(args.input_file):
        raise FileNotFoundError(f"Input not found: {args.input_file}")

    item_ids, embeddings = load_parquet_embeddings(args.input_file)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(args, input_dim=embeddings.shape[1], device=device)

    if args.load_model and os.path.exists(args.model_path):
        load_checkpoint(model, args.model_path, device)
    else:
        train_model(model, embeddings, args, device)
        # Reload best checkpoint for encoding
        if os.path.exists(args.model_path):
            load_checkpoint(model, args.model_path, device)

    codes = encode_embeddings(model, embeddings, batch_size=args.batch_size, device=device)
    init_rate, _, init_max_dup = collision_stats(codes)
    print(f"[Codes] initial collision_rate={init_rate:.4f}, max_dup={init_max_dup}")
    if args.refine_collisions:
        codes = refine_collisions(model, embeddings, codes, args, device)
        final_rate, _, final_max_dup = collision_stats(codes)
        print(f"[Codes] refined collision_rate={final_rate:.4f}, max_dup={final_max_dup}")
    save_codes(item_ids, codes, args.output_file)


if __name__ == "__main__":
    main()
