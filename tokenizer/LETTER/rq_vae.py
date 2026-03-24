import random

import torch
from torch import nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def kmeans_torch(x: torch.Tensor, k: int, n_iters: int = 50):
    n = x.shape[0]
    if n < k:
        x = x.repeat((k + n - 1) // n, 1)
        n = x.shape[0]
    perm = torch.randperm(n, device=x.device)
    centers = x[perm[:k]].clone()
    for _ in range(n_iters):
        d = torch.cdist(x, centers, p=2)
        assign = d.argmin(dim=1)
        for i in range(k):
            mask = assign == i
            if mask.any():
                centers[i] = x[mask].mean(dim=0)
            else:
                centers[i] = x[torch.randint(0, n, (1,), device=x.device)]
    return centers


@torch.no_grad()
def sinkhorn(distances: torch.Tensor, epsilon: float, n_iters: int):
    Q = torch.exp(-distances / epsilon)
    B, K = Q.shape
    Q /= Q.sum()
    for _ in range(n_iters):
        Q /= Q.sum(dim=1, keepdim=True).clamp(min=1e-12)
        Q /= B
        Q /= Q.sum(dim=0, keepdim=True).clamp(min=1e-12)
        Q /= K
    return Q * B


def constrained_km(data, n_clusters=10):
    """KMeans on numpy array. Returns (centers, labels)."""
    from sklearn.cluster import KMeans as SKKMeans
    import numpy as np
    clf = SKKMeans(n_clusters=n_clusters, max_iter=30, n_init=5)
    clf.fit(data)
    return torch.from_numpy(clf.cluster_centers_), clf.labels_.tolist()


# ---------------------------------------------------------------------------
# EMA VQ layer + diversity loss
# ---------------------------------------------------------------------------

class VectorQuantizerEMA(nn.Module):
    """EMA codebook + dead-code restart + LETTER diversity loss."""

    def __init__(self, n_e: int, e_dim: int, commitment_weight: float = 0.25,
                 ema_decay: float = 0.99, dead_threshold: float = 2.0,
                 diversity_weight: float = 0.01,
                 sk_epsilon: float = 0.0, sk_iters: int = 50):
        super().__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.commitment_weight = commitment_weight
        self.ema_decay = ema_decay
        self.dead_threshold = dead_threshold
        self.diversity_weight = diversity_weight
        self.sk_epsilon = sk_epsilon
        self.sk_iters = sk_iters

        self.register_buffer("codebook", torch.randn(n_e, e_dim))
        self.register_buffer("ema_count", torch.zeros(n_e))
        self.register_buffer("ema_weight", self.codebook.clone())

    def _distances(self, x: torch.Tensor):
        return (
            (x ** 2).sum(1, keepdim=True)
            + (self.codebook ** 2).sum(1).unsqueeze(0)
            - 2.0 * x @ self.codebook.t()
        )

    @torch.no_grad()
    def init_codebook(self, x: torch.Tensor, n_iters: int = 50):
        centers = kmeans_torch(x.float(), self.n_e, n_iters=n_iters)
        self.codebook.copy_(centers)
        self.ema_weight.copy_(centers)
        self.ema_count.fill_(1.0)

    @torch.no_grad()
    def _ema_update(self, x: torch.Tensor, indices: torch.Tensor):
        x = x.float()
        one_hot = F.one_hot(indices, self.n_e).float()
        count = one_hot.sum(0)
        weight_sum = one_hot.t() @ x

        self.ema_count.mul_(self.ema_decay).add_(count, alpha=1 - self.ema_decay)
        self.ema_weight.mul_(self.ema_decay).add_(weight_sum, alpha=1 - self.ema_decay)

        # Dead code restart
        dead = self.ema_count < self.dead_threshold
        if dead.any():
            n_dead = int(dead.sum().item())
            rand_idx = torch.randint(0, x.shape[0], (n_dead,), device=x.device)
            self.ema_weight[dead] = x[rand_idx]
            self.ema_count[dead] = self.dead_threshold

        n = self.ema_count.sum()
        smoothed = (self.ema_count + 1e-5) / (n + self.n_e * 1e-5) * n
        self.codebook.copy_(self.ema_weight / smoothed.unsqueeze(1))

    def _diversity_loss(self, x_q, indices, cluster_labels):
        """LETTER diversity loss: contrastive within KMeans clusters."""
        if cluster_labels is None or len(cluster_labels) != self.n_e:
            return torch.zeros((), device=x_q.device)

        cluster_map = {}
        for idx, cl in enumerate(cluster_labels):
            cluster_map.setdefault(cl, []).append(idx)

        pos_targets = []
        valid_mask = []
        for i in range(x_q.shape[0]):
            code_idx = indices[i].item()
            cl = cluster_labels[code_idx]
            pool = cluster_map[cl]
            if len(pool) < 2:
                pos_targets.append(0)
                valid_mask.append(False)
                continue
            target = random.choice(pool)
            while target == code_idx:
                target = random.choice(pool)
            pos_targets.append(target)
            valid_mask.append(True)

        if not any(valid_mask):
            return torch.zeros((), device=x_q.device)

        y_true = torch.tensor(pos_targets, device=x_q.device, dtype=torch.long)
        # Normalize for stable similarity
        x_q_norm = F.normalize(x_q, dim=-1)
        cb_norm = F.normalize(self.codebook, dim=-1)
        sim = x_q_norm @ cb_norm.t() / 0.1  # temperature=0.1

        # Mask out self
        for i in range(sim.shape[0]):
            sim[i, indices[i]] = -1e9

        valid_t = torch.tensor(valid_mask, device=x_q.device)
        loss = F.cross_entropy(sim[valid_t], y_true[valid_t])
        return loss

    def forward(self, x: torch.Tensor, cluster_labels=None, use_sk: bool = False):
        d = self._distances(x)

        if use_sk and self.sk_epsilon > 0 and x.shape[0] > 1:
            max_d, min_d = d.max(), d.min()
            centered = (d - (max_d + min_d) / 2) / ((max_d - min_d) / 2).clamp(min=1e-5)
            Q = sinkhorn(centered.double(), self.sk_epsilon, self.sk_iters)
            indices = Q.argmax(dim=1).long()
        else:
            indices = d.argmin(dim=1)

        x_q = self.codebook[indices]

        if self.training:
            self._ema_update(x.detach(), indices.detach())

        # Commitment loss
        commit_loss = F.mse_loss(x_q.detach(), x)

        # Diversity loss
        div_loss = torch.zeros((), device=x.device)
        if self.training and self.diversity_weight > 0:
            div_loss = self._diversity_loss(x_q.detach(), indices, cluster_labels)

        loss = self.commitment_weight * commit_loss + self.diversity_weight * div_loss

        # Straight-through
        x_q = x + (x_q - x).detach()
        return x_q, loss, indices


# ---------------------------------------------------------------------------
# Residual VQ
# ---------------------------------------------------------------------------

class ResidualVQ(nn.Module):

    def __init__(self, n_e_list, e_dim, commitment_weight=0.25,
                 ema_decay=0.99, dead_threshold=2.0, diversity_weight=0.01,
                 sk_epsilons=None, sk_iters=50):
        super().__init__()
        n_layers = len(n_e_list)
        if sk_epsilons is None:
            sk_epsilons = [0.0] * n_layers
        self.vq_layers = nn.ModuleList([
            VectorQuantizerEMA(n_e, e_dim, commitment_weight=commitment_weight,
                               ema_decay=ema_decay, dead_threshold=dead_threshold,
                               diversity_weight=diversity_weight,
                               sk_epsilon=eps, sk_iters=sk_iters)
            for n_e, eps in zip(n_e_list, sk_epsilons)
        ])

    @torch.no_grad()
    def init_codebooks(self, z, n_iters=50):
        residual = z.clone()
        for vq in self.vq_layers:
            vq.init_codebook(residual, n_iters=n_iters)
            indices = vq._distances(residual).argmin(dim=1)
            residual = residual - vq.codebook[indices]

    def forward(self, z, cluster_labels_list=None, use_sk=False):
        residual = z
        x_q_sum = torch.zeros_like(z)
        all_losses = []
        all_indices = []
        for i, vq in enumerate(self.vq_layers):
            cl = cluster_labels_list[i] if cluster_labels_list else None
            x_q, loss, indices = vq(residual, cluster_labels=cl, use_sk=use_sk)
            residual = residual - x_q
            x_q_sum = x_q_sum + x_q
            all_losses.append(loss)
            all_indices.append(indices)
        return x_q_sum, torch.stack(all_losses).mean(), torch.stack(all_indices, dim=1)

    @torch.no_grad()
    def encode(self, z, use_sk=False):
        residual = z
        all_indices = []
        for vq in self.vq_layers:
            d = vq._distances(residual)
            indices = d.argmin(dim=1)
            residual = residual - vq.codebook[indices]
            all_indices.append(indices)
        return torch.stack(all_indices, dim=1)


# ---------------------------------------------------------------------------
# LETTER RQ-VAE (EMA codebook + CF contrastive + diversity)
# ---------------------------------------------------------------------------

class RQVAE(nn.Module):

    def __init__(self, in_dim, e_dim=32, n_e_list=None, encoder_dims=None,
                 commitment_weight=0.25, ema_decay=0.99, dead_threshold=2.0,
                 diversity_weight=0.01,
                 sk_epsilons=None, sk_iters=50,
                 quant_loss_weight=1.0,
                 cf_alpha=0.1):
        super().__init__()
        if n_e_list is None:
            n_e_list = [256, 256, 256, 256]
        if encoder_dims is None:
            encoder_dims = [1024, 512, 256, 128]

        self.quant_loss_weight = quant_loss_weight
        self.cf_alpha = cf_alpha

        # Encoder
        dims = [in_dim] + encoder_dims + [e_dim]
        enc = []
        for i in range(len(dims) - 1):
            enc.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                enc.append(nn.ReLU())
        self.encoder = nn.Sequential(*enc)

        # Decoder
        dims_rev = dims[::-1]
        dec = []
        for i in range(len(dims_rev) - 1):
            dec.append(nn.Linear(dims_rev[i], dims_rev[i + 1]))
            if i < len(dims_rev) - 2:
                dec.append(nn.ReLU())
        self.decoder = nn.Sequential(*dec)

        self.rq = ResidualVQ(
            n_e_list=n_e_list, e_dim=e_dim,
            commitment_weight=commitment_weight,
            ema_decay=ema_decay, dead_threshold=dead_threshold,
            diversity_weight=diversity_weight,
            sk_epsilons=sk_epsilons, sk_iters=sk_iters,
        )

    def forward(self, x, cf_emb=None, cluster_labels_list=None):
        z = self.encoder(x)
        z_q, quant_loss, codes = self.rq(z, cluster_labels_list=cluster_labels_list)
        z_q_st = z + (z_q - z).detach()
        x_rec = self.decoder(z_q_st)

        recon_loss = F.mse_loss(x_rec, x)
        loss = recon_loss + self.quant_loss_weight * quant_loss

        # CF contrastive loss (InfoNCE, raw dot product like original LETTER)
        cf_loss = torch.zeros((), device=x.device)
        if cf_emb is not None and self.cf_alpha > 0:
            sim = z_q_st @ cf_emb.t()
            labels = torch.arange(sim.shape[0], device=sim.device)
            cf_loss = F.cross_entropy(sim, labels)
            loss = loss + self.cf_alpha * cf_loss

        return {
            "loss": loss,
            "recon_loss": recon_loss.detach(),
            "quant_loss": quant_loss.detach(),
            "cf_loss": cf_loss.detach(),
            "codes": codes,
        }

    @torch.no_grad()
    def encode(self, x, use_sk=False):
        z = self.encoder(x)
        return self.rq.encode(z, use_sk=use_sk)

    @torch.no_grad()
    def init_codebooks(self, x, n_iters=50):
        z = self.encoder(x)
        self.rq.init_codebooks(z, n_iters=n_iters)
