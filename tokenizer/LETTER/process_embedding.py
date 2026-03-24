#!/usr/bin/env python3
"""
LETTER-style RQ-VAE tokenizer with CF contrastive loss.

Input:  parquet with columns [item_id, embedding]
        + SASRec checkpoint for CF embeddings
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

from rq_vae import RQVAE, constrained_km


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


def load_cf_embeddings(ckpt_path: str, item_ids):
    """Load SASRec item embeddings and align with parquet item_ids.

    SASRec checkpoint stores item_emb.weight of shape [item_num+1, 128].
    Row 0 is padding, so item i's embedding is at row i.
    """
    print(f"Loading CF embeddings: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    sd = ckpt if isinstance(ckpt, dict) and "item_emb.weight" in ckpt else ckpt.get("state_dict", ckpt)
    emb_weight = sd["item_emb.weight"]  # [item_num+1, 128]
    print(f"  SASRec item_emb shape: {emb_weight.shape}")

    # Build lookup: item_id -> row index in SASRec (item_id == row index)
    cf_list = []
    missing = 0
    for iid in item_ids:
        idx = int(iid)
        if 0 < idx < emb_weight.shape[0]:
            cf_list.append(emb_weight[idx].numpy())
        else:
            # Fallback: zero vector (should be rare)
            cf_list.append(np.zeros(emb_weight.shape[1], dtype=np.float32))
            missing += 1

    cf_emb = np.stack(cf_list, axis=0).astype(np.float32)
    if missing > 0:
        print(f"  WARNING: {missing}/{len(item_ids)} items missing CF embedding")
    print(f"  CF embedding shape: {cf_emb.shape}")
    return cf_emb


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
def refine_collisions(model, emb, codes, device, max_rounds=5, target_rate=0.05):
    best_codes = codes.copy()
    best_rate, _, best_max = collision_stats(best_codes)
    print(f"[Refine] initial collision_rate={best_rate:.4f}, max_dup={best_max}")

    # Enable Sinkhorn on last layer only
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
# Cluster labels (LETTER diversity loss)
# ---------------------------------------------------------------------------

def compute_cluster_labels(model, n_clusters=10):
    """Run constrained KMeans on each VQ layer's codebook. Returns list of label lists."""
    cluster_labels_list = []
    for vq in model.rq.vq_layers:
        w = vq.codebook.detach().cpu().numpy()
        _, labels = constrained_km(w, n_clusters=n_clusters)
        cluster_labels_list.append(labels)
    return cluster_labels_list


# ---------------------------------------------------------------------------
# Train / Encode
# ---------------------------------------------------------------------------

def train(model, emb, cf_emb, args, device):
    n_items = len(emb)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # KMeans init
    print("Running KMeans init ...")
    model.eval()
    with torch.no_grad():
        model.init_codebooks(torch.from_numpy(emb).to(device), n_iters=args.kmeans_iters)
    print("KMeans init done.")

    cf_tensor = torch.from_numpy(cf_emb).to(device)
    target_cf_alpha = model.cf_alpha

    print(f"CF warmup: epochs 1~{args.cf_warmup} pure AE+VQ, "
          f"then ramp CF alpha 0→{target_cf_alpha} over epochs {args.cf_warmup+1}~{args.cf_warmup + args.cf_ramp}")

    best_loss = float("inf")
    for epoch in range(1, args.epochs + 1):
        model.train()

        # CF alpha schedule: warmup -> linear ramp -> full
        if epoch <= args.cf_warmup:
            model.cf_alpha = 0.0
        elif epoch <= args.cf_warmup + args.cf_ramp:
            progress = (epoch - args.cf_warmup) / args.cf_ramp
            model.cf_alpha = target_cf_alpha * progress
        else:
            model.cf_alpha = target_cf_alpha

        # Re-cluster codebook each epoch for diversity loss
        cluster_labels_list = compute_cluster_labels(model, n_clusters=args.n_clusters)

        # Shuffle and batch manually (to keep cf_emb aligned)
        perm = np.random.permutation(n_items)
        total_loss = total_recon = total_vq = total_cf = 0.0
        n = 0
        code_counts = torch.zeros(len(args.n_e_list), max(args.n_e_list), dtype=torch.long)

        # Only pass CF embeddings after warmup
        use_cf = model.cf_alpha > 0

        n_batches = (n_items + args.batch_size - 1) // args.batch_size
        for bi in tqdm(range(n_batches), desc=f"Epoch {epoch}/{args.epochs}"):
            batch_idx = perm[bi * args.batch_size: (bi + 1) * args.batch_size]
            batch_x = torch.from_numpy(emb[batch_idx]).to(device, non_blocking=True)
            batch_cf = cf_tensor[batch_idx] if use_cf else None

            optimizer.zero_grad(set_to_none=True)
            out = model(batch_x, cf_emb=batch_cf, cluster_labels_list=cluster_labels_list)
            out["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            bs = batch_x.shape[0]
            n += bs
            total_loss += out["loss"].item() * bs
            total_recon += out["recon_loss"].item() * bs
            total_vq += out["quant_loss"].item() * bs
            total_cf += out["cf_loss"].item() * bs

            codes = out["codes"].detach().cpu()
            for layer in range(codes.shape[1]):
                code_counts[layer] += torch.bincount(codes[:, layer], minlength=args.n_e_list[layer]).cpu()

        scheduler.step()

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
        cf_alpha_str = f" cf_alpha={model.cf_alpha:.4f}" if use_cf else " cf=OFF"
        print(f"[Epoch {epoch}] loss={epoch_loss:.6f} recon={total_recon/n:.6f} "
              f"vq={total_vq/n:.6f} cf={total_cf/n:.6f}{cf_alpha_str} | {' '.join(usage_parts)}")

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            save_checkpoint(model, args.model_path, args)
            print(f"  -> saved (loss={epoch_loss:.6f})")

    # Restore and reload best
    model.cf_alpha = target_cf_alpha
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
    p.add_argument("--cf_ckpt", required=True, help="SASRec checkpoint path")
    p.add_argument("--load_model", action="store_true")

    # Model
    p.add_argument("--n_e_list", type=int, nargs="+", default=[256, 256, 256, 256])
    p.add_argument("--e_dim", type=int, default=32)
    p.add_argument("--encoder_dims", type=int, nargs="+", default=[1024, 512, 256, 128])
    p.add_argument("--commitment_weight", type=float, default=0.25)
    p.add_argument("--ema_decay", type=float, default=0.99)
    p.add_argument("--dead_threshold", type=float, default=2.0)
    p.add_argument("--diversity_weight", type=float, default=0.01)
    p.add_argument("--quant_loss_weight", type=float, default=1.0)
    p.add_argument("--cf_alpha", type=float, default=0.02)
    p.add_argument("--cf_dim", type=int, default=128)
    p.add_argument("--cf_warmup", type=int, default=50, help="Epochs of pure AE+VQ before CF loss")
    p.add_argument("--cf_ramp", type=int, default=50, help="Epochs to linearly ramp CF alpha from 0 to target")
    p.add_argument("--sk_epsilons", type=float, nargs="+", default=[0.0, 0.0, 0.0, 0.003])
    p.add_argument("--sk_iters", type=int, default=50)
    p.add_argument("--kmeans_iters", type=int, default=50)
    p.add_argument("--n_clusters", type=int, default=10, help="KMeans clusters for diversity loss")

    # Training
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch_size", type=int, default=1024)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)

    # Refine
    p.add_argument("--max_refine_rounds", type=int, default=5)
    p.add_argument("--target_collision_rate", type=float, default=0.05)
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    item_ids, emb = load_parquet(args.input_file)
    cf_emb = load_cf_embeddings(args.cf_ckpt, item_ids)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RQVAE(
        in_dim=emb.shape[1],
        e_dim=args.e_dim,
        n_e_list=args.n_e_list,
        encoder_dims=args.encoder_dims,
        commitment_weight=args.commitment_weight,
        ema_decay=args.ema_decay,
        dead_threshold=args.dead_threshold,
        diversity_weight=args.diversity_weight,
        sk_epsilons=args.sk_epsilons,
        sk_iters=args.sk_iters,
        quant_loss_weight=args.quant_loss_weight,
        cf_dim=args.cf_dim,
        cf_alpha=args.cf_alpha,
    ).to(device)

    if args.load_model and os.path.exists(args.model_path):
        load_checkpoint(model, args.model_path, device)
    else:
        train(model, emb, cf_emb, args, device)

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
