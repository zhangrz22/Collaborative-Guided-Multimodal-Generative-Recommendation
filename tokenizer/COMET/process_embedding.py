#!/usr/bin/env python3
"""
COMET RQ-VAE tokenizer with CF-as-Query Cross-Attention fusion.

Input:  text embeddings parquet  (item_id, embedding) — 4096-dim
        image embeddings parquet (item_id, embedding) — 768-dim
        SASRec checkpoint for CF embeddings            — 128-dim
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


def load_image_parquet(path: str, item_ids):
    """Load image embeddings and align with item_ids.

    Returns:
        image_emb: np.ndarray [N, image_dim]
        image_mask: np.ndarray [N] bool, True = missing (zero-vector)
    """
    print(f"Loading image embeddings: {path}")
    df = pd.read_parquet(path)
    # Build lookup: item_id -> embedding
    img_dict = {}
    for _, row in df.iterrows():
        img_dict[row["item_id"]] = np.asarray(row["embedding"], dtype=np.float32)

    # Detect dim from first entry
    image_dim = next(iter(img_dict.values())).shape[0]
    print(f"  image dim={image_dim}, items in parquet={len(img_dict)}")

    image_emb = np.zeros((len(item_ids), image_dim), dtype=np.float32)
    image_mask = np.zeros(len(item_ids), dtype=bool)
    missing = 0

    for i, iid in enumerate(item_ids):
        if iid in img_dict:
            vec = img_dict[iid]
            # Detect zero-vector (missing image downloaded as zeros)
            if np.allclose(vec, 0.0):
                image_mask[i] = True
                missing += 1
            else:
                image_emb[i] = vec
        else:
            image_mask[i] = True
            missing += 1

    print(f"  matched={len(item_ids) - missing}, missing={missing}")
    return image_emb, image_mask


def load_cf_embeddings(ckpt_path: str, item_ids):
    """Load SASRec item embeddings and align with parquet item_ids."""
    print(f"Loading CF embeddings: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    sd = ckpt if isinstance(ckpt, dict) and "item_emb.weight" in ckpt else ckpt.get("state_dict", ckpt)
    emb_weight = sd["item_emb.weight"]  # [item_num+1, e_dim]
    print(f"  SASRec item_emb shape: {emb_weight.shape}")

    cf_list = []
    missing = 0
    for iid in item_ids:
        idx = int(iid)
        if 0 < idx < emb_weight.shape[0]:
            cf_list.append(emb_weight[idx].numpy())
        else:
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
def refine_collisions(model, text_emb, image_emb, cf_emb, image_mask,
                      codes, device, max_rounds=5, target_rate=0.05):
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
            idx_arr = np.array(group)
            t_text = torch.from_numpy(text_emb[idx_arr]).to(device)
            t_image = torch.from_numpy(image_emb[idx_arr]).to(device)
            t_cf = torch.from_numpy(cf_emb[idx_arr]).to(device)
            t_mask = torch.from_numpy(image_mask[idx_arr]).to(device)
            new_codes = model.encode(t_text, t_image, t_cf,
                                     image_mask=t_mask, use_sk=True).cpu().numpy()
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
    """Run constrained KMeans on each VQ layer's codebook."""
    cluster_labels_list = []
    for vq in model.rq.vq_layers:
        w = vq.codebook.detach().cpu().numpy()
        _, labels = constrained_km(w, n_clusters=n_clusters)
        cluster_labels_list.append(labels)
    return cluster_labels_list


# ---------------------------------------------------------------------------
# Train / Encode
# ---------------------------------------------------------------------------

def train(model, text_emb, image_emb, cf_emb, image_mask, args, device):
    n_items = len(text_emb)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # KMeans init
    print("Running KMeans init ...")
    model.eval()
    with torch.no_grad():
        t_text = torch.from_numpy(text_emb).to(device)
        t_image = torch.from_numpy(image_emb).to(device)
        t_cf = torch.from_numpy(cf_emb).to(device)
        t_mask = torch.from_numpy(image_mask).to(device)
        model.init_codebooks(t_text, t_image, t_cf, image_mask=t_mask,
                             n_iters=args.kmeans_iters)
        del t_text, t_image, t_cf, t_mask
    print("KMeans init done.")

    cf_tensor = torch.from_numpy(cf_emb).to(device)
    image_tensor = torch.from_numpy(image_emb).to(device)
    mask_tensor = torch.from_numpy(image_mask).to(device)

    best_loss = float("inf")
    for epoch in range(1, args.epochs + 1):
        model.train()

        # Re-cluster codebook each epoch for diversity loss
        cluster_labels_list = compute_cluster_labels(model, n_clusters=args.n_clusters)

        # Shuffle and batch manually (to keep all modalities aligned)
        perm = np.random.permutation(n_items)
        total_loss = total_recon = total_vq = 0.0
        total_text_recon = total_image_recon = total_cf_recon = 0.0
        n = 0
        code_counts = torch.zeros(len(args.n_e_list), max(args.n_e_list), dtype=torch.long)

        n_batches = (n_items + args.batch_size - 1) // args.batch_size
        for bi in tqdm(range(n_batches), desc=f"Epoch {epoch}/{args.epochs}"):
            batch_idx = perm[bi * args.batch_size: (bi + 1) * args.batch_size]
            batch_text = torch.from_numpy(text_emb[batch_idx]).to(device, non_blocking=True)
            batch_image = image_tensor[batch_idx]
            batch_cf = cf_tensor[batch_idx]
            batch_mask = mask_tensor[batch_idx]

            optimizer.zero_grad(set_to_none=True)
            out = model(batch_text, batch_image, batch_cf, image_mask=batch_mask,
                        cluster_labels_list=cluster_labels_list)
            out["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            bs = batch_text.shape[0]
            n += bs
            total_loss += out["loss"].item() * bs
            total_recon += out["recon_loss"].item() * bs
            total_vq += out["quant_loss"].item() * bs
            total_text_recon += out["text_recon_loss"].item() * bs
            total_image_recon += out["image_recon_loss"].item() * bs
            total_cf_recon += out["cf_recon_loss"].item() * bs

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
              f"(text={total_text_recon/n:.6f} image={total_image_recon/n:.6f} cf={total_cf_recon/n:.6f}) "
              f"vq={total_vq/n:.6f} | {' '.join(usage_parts)}")

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            save_checkpoint(model, args.model_path, args)
            print(f"  -> saved (loss={epoch_loss:.6f})")

    # Reload best
    load_checkpoint(model, args.model_path, device)


@torch.no_grad()
def encode_all(model, text_emb, image_emb, cf_emb, image_mask, batch_size, device):
    n = len(text_emb)
    model.eval()
    all_codes = []

    image_tensor = torch.from_numpy(image_emb).to(device)
    cf_tensor = torch.from_numpy(cf_emb).to(device)
    mask_tensor = torch.from_numpy(image_mask).to(device)

    n_batches = (n + batch_size - 1) // batch_size
    for bi in tqdm(range(n_batches), desc="Encoding"):
        s = bi * batch_size
        e = min(s + batch_size, n)
        batch_text = torch.from_numpy(text_emb[s:e]).to(device, non_blocking=True)
        batch_image = image_tensor[s:e]
        batch_cf = cf_tensor[s:e]
        batch_mask = mask_tensor[s:e]
        codes = model.encode(batch_text, batch_image, batch_cf,
                             image_mask=batch_mask)
        all_codes.append(codes.cpu())

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
    p.add_argument("--input_file", required=True, help="Text embeddings parquet")
    p.add_argument("--image_file", required=True, help="Image embeddings parquet")
    p.add_argument("--output_file", required=True)
    p.add_argument("--model_path", required=True)
    p.add_argument("--cf_ckpt", required=True, help="SASRec checkpoint path")
    p.add_argument("--load_model", action="store_true")

    # COMET fusion
    p.add_argument("--d_model", type=int, default=256, help="Cross-attention dimension")
    p.add_argument("--n_heads", type=int, default=4, help="Number of attention heads")
    p.add_argument("--fusion_dropout", type=float, default=0.1, help="Dropout in cross-attention")

    # Model
    p.add_argument("--n_e_list", type=int, nargs="+", default=[256, 256, 256, 256])
    p.add_argument("--e_dim", type=int, default=32)
    p.add_argument("--decoder_dims", type=int, nargs="+", default=[128, 256, 512, 1024])
    p.add_argument("--commitment_weight", type=float, default=0.25)
    p.add_argument("--ema_decay", type=float, default=0.99)
    p.add_argument("--dead_threshold", type=float, default=2.0)
    p.add_argument("--diversity_weight", type=float, default=0.0001)
    p.add_argument("--quant_loss_weight", type=float, default=1.0)
    p.add_argument("--w_text", type=float, default=1.0, help="Weight for text reconstruction loss")
    p.add_argument("--w_image", type=float, default=0.1, help="Weight for image reconstruction loss")
    p.add_argument("--w_cf", type=float, default=0.1, help="Weight for CF reconstruction loss")
    p.add_argument("--sk_epsilons", type=float, nargs="+", default=[0.0, 0.0, 0.0, 0.003])
    p.add_argument("--sk_iters", type=int, default=50)
    p.add_argument("--kmeans_iters", type=int, default=100)
    p.add_argument("--n_clusters", type=int, default=10)

    # Training
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch_size", type=int, default=1024)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cuda:0")

    # Refine
    p.add_argument("--max_refine_rounds", type=int, default=5)
    p.add_argument("--target_collision_rate", type=float, default=0.05)
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Load all three modalities
    item_ids, text_emb = load_parquet(args.input_file)
    image_emb, image_mask = load_image_parquet(args.image_file, item_ids)
    cf_emb = load_cf_embeddings(args.cf_ckpt, item_ids)

    text_dim = text_emb.shape[1]
    image_dim = image_emb.shape[1]
    cf_dim = cf_emb.shape[1]
    print(f"Dimensions: text={text_dim}, image={image_dim}, cf={cf_dim}")
    print(f"Missing images: {image_mask.sum()}/{len(image_mask)}")

    device = torch.device(args.device)
    print(f"Using device: {device}")

    model = RQVAE(
        text_dim=text_dim,
        image_dim=image_dim,
        cf_dim=cf_dim,
        d_model=args.d_model,
        n_heads=args.n_heads,
        e_dim=args.e_dim,
        fusion_dropout=args.fusion_dropout,
        decoder_dims=args.decoder_dims,
        n_e_list=args.n_e_list,
        commitment_weight=args.commitment_weight,
        ema_decay=args.ema_decay,
        dead_threshold=args.dead_threshold,
        diversity_weight=args.diversity_weight,
        sk_epsilons=args.sk_epsilons,
        sk_iters=args.sk_iters,
        quant_loss_weight=args.quant_loss_weight,
        w_text=args.w_text,
        w_image=args.w_image,
        w_cf=args.w_cf,
    ).to(device)

    if args.load_model and os.path.exists(args.model_path):
        load_checkpoint(model, args.model_path, device)
    else:
        train(model, text_emb, image_emb, cf_emb, image_mask, args, device)

    codes = encode_all(model, text_emb, image_emb, cf_emb, image_mask,
                       args.batch_size, device)
    rate, _, max_dup = collision_stats(codes)
    print(f"[Codes] collision_rate={rate:.4f}, max_dup={max_dup}")

    codes = refine_collisions(
        model, text_emb, image_emb, cf_emb, image_mask,
        codes, device,
        max_rounds=args.max_refine_rounds,
        target_rate=args.target_collision_rate,
    )
    save_codes(item_ids, codes, args.output_file)


if __name__ == "__main__":
    main()
