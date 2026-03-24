import torch
from torch import nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def sample_gumbel(shape, device, eps=1e-20):
    U = torch.rand(shape, device=device)
    return -torch.log(-torch.log(U + eps) + eps)


def gumbel_softmax_sample(logits, temperature, device):
    y = logits + sample_gumbel(logits.shape, device)
    return F.softmax(y / temperature, dim=-1)


@torch.no_grad()
def sinkhorn_knopp(cost, eps=0.003, max_iter=100):
    B, K = cost.shape
    K_mat = torch.exp(-cost / eps)
    u = torch.ones(B, device=cost.device, dtype=cost.dtype)
    v = torch.ones(K, device=cost.device, dtype=cost.dtype)
    row_m = torch.full((B,), 1.0 / B, device=cost.device, dtype=cost.dtype)
    col_m = torch.full((K,), 1.0 / K, device=cost.device, dtype=cost.dtype)
    for _ in range(max_iter):
        u = row_m / (K_mat @ v + 1e-8)
        v = col_m / (K_mat.T @ u + 1e-8)
    P = u.unsqueeze(1) * K_mat * v.unsqueeze(0)
    return P


@torch.no_grad()
def kmeans_init(x, k, n_iters=50):
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
                centers[i] = x[mask].mean(0)
            else:
                centers[i] = x[torch.randint(0, n, (1,), device=x.device)]
    return centers


# ---------------------------------------------------------------------------
# Quantize layer (Gumbel-Softmax training, Sinkhorn for last layer refine)
# ---------------------------------------------------------------------------

class Quantize(nn.Module):

    def __init__(self, n_embed, embed_dim, commitment_weight=0.25,
                 use_sinkhorn=False, sk_epsilon=0.003, sk_iters=100):
        super().__init__()
        self.n_embed = n_embed
        self.embed_dim = embed_dim
        self.commitment_weight = commitment_weight
        self.use_sinkhorn = use_sinkhorn
        self.sk_epsilon = sk_epsilon
        self.sk_iters = sk_iters
        self.embedding = nn.Embedding(n_embed, embed_dim)
        nn.init.uniform_(self.embedding.weight)
        self.kmeans_initted = False

    def _distances(self, x):
        return (
            (x ** 2).sum(1, keepdim=True)
            + (self.embedding.weight ** 2).sum(1).unsqueeze(0)
            - 2.0 * x @ self.embedding.weight.t()
        )

    @torch.no_grad()
    def _kmeans_init(self, x):
        centers = kmeans_init(x, self.n_embed)
        self.embedding.weight.copy_(centers)
        self.kmeans_initted = True

    def forward(self, x, gumbel_t=0.2):
        if not self.kmeans_initted and self.training:
            self._kmeans_init(x)

        dist = self._distances(x)
        ids = dist.detach().argmin(dim=1)

        if self.training:
            if self.use_sinkhorn:
                # Sinkhorn assignment
                max_d, min_d = dist.max(), dist.min()
                mid = (max_d + min_d) / 2
                amp = (max_d - mid).clamp(min=1e-5)
                cost_norm = ((dist - mid) / amp).double()
                P = sinkhorn_knopp(cost_norm, eps=self.sk_epsilon, max_iter=self.sk_iters)
                sk_ids = P.argmax(dim=-1).long()
                emb = self.embedding(sk_ids)
                emb_out = x + (emb - x).detach()  # STE
                ids = sk_ids
            else:
                # Gumbel-Softmax: fully differentiable
                weights = gumbel_softmax_sample(-dist, temperature=gumbel_t, device=x.device)
                emb = weights @ self.embedding.weight
                emb_out = emb
        else:
            emb_out = self.embedding(ids)
            emb = emb_out

        # VQ loss: codebook_loss + commitment_weight * commitment_loss
        emb_loss = ((x.detach() - emb) ** 2).sum(dim=-1)
        commit_loss = ((x - emb.detach()) ** 2).sum(dim=-1)
        loss = emb_loss + self.commitment_weight * commit_loss

        return emb_out, loss, ids


# ---------------------------------------------------------------------------
# Residual Quantizer
# ---------------------------------------------------------------------------

class ResidualVQ(nn.Module):

    def __init__(self, n_e_list, embed_dim, commitment_weight=0.25,
                 sk_epsilon=0.003, sk_iters=100):
        super().__init__()
        n_layers = len(n_e_list)
        # All layers: Gumbel-Softmax; last layer: Sinkhorn
        self.vq_layers = nn.ModuleList([
            Quantize(n_e, embed_dim, commitment_weight=commitment_weight,
                     use_sinkhorn=(i == n_layers - 1),
                     sk_epsilon=sk_epsilon, sk_iters=sk_iters)
            for i, n_e in enumerate(n_e_list)
        ])

    def forward(self, z, gumbel_t=0.2):
        residual = z
        x_q_sum = torch.zeros_like(z)
        all_losses = []
        all_ids = []
        for vq in self.vq_layers:
            emb, loss, ids = vq(residual, gumbel_t=gumbel_t)
            residual = residual - emb
            x_q_sum = x_q_sum + emb
            all_losses.append(loss)
            all_ids.append(ids)
        total_loss = torch.stack(all_losses).sum(dim=0)  # sum across layers, per-sample
        codes = torch.stack(all_ids, dim=1)
        return x_q_sum, total_loss, codes

    @torch.no_grad()
    def encode(self, z):
        residual = z
        all_ids = []
        for vq in self.vq_layers:
            dist = vq._distances(residual)
            ids = dist.argmin(dim=1)
            emb = vq.embedding(ids)
            residual = residual - emb
            all_ids.append(ids)
        return torch.stack(all_ids, dim=1)


# ---------------------------------------------------------------------------
# RQ-VAE (actually an AE, following GenRec)
# ---------------------------------------------------------------------------

class RQVAE(nn.Module):

    def __init__(self, in_dim, embed_dim=32,
                 n_e_list=None, encoder_dims=None,
                 commitment_weight=0.25,
                 sk_epsilon=0.003, sk_iters=100,
                 quant_loss_weight=1.0):
        super().__init__()
        if n_e_list is None:
            n_e_list = [256, 256, 256, 256]
        if encoder_dims is None:
            encoder_dims = [1024, 512, 256, 128]

        self.quant_loss_weight = quant_loss_weight

        # Encoder: in_dim -> hidden -> embed_dim  (SiLU, no bias, like GenRec)
        dims = [in_dim] + encoder_dims + [embed_dim]
        enc = []
        for i in range(len(dims) - 1):
            enc.append(nn.Linear(dims[i], dims[i + 1], bias=False))
            if i < len(dims) - 2:
                enc.append(nn.SiLU())
        self.encoder = nn.Sequential(*enc)

        # Decoder: mirror
        dims_rev = dims[::-1]
        dec = []
        for i in range(len(dims_rev) - 1):
            dec.append(nn.Linear(dims_rev[i], dims_rev[i + 1], bias=False))
            if i < len(dims_rev) - 2:
                dec.append(nn.SiLU())
        self.decoder = nn.Sequential(*dec)

        self.rq = ResidualVQ(
            n_e_list=n_e_list, embed_dim=embed_dim,
            commitment_weight=commitment_weight,
            sk_epsilon=sk_epsilon, sk_iters=sk_iters,
        )

    def forward(self, x, gumbel_t=0.2):
        z = self.encoder(x)
        z_q, quant_loss, codes = self.rq(z, gumbel_t=gumbel_t)
        z_q_st = z + (z_q - z).detach()
        x_rec = self.decoder(z_q_st)
        recon_loss = ((x_rec - x) ** 2).sum(dim=-1)  # per-sample, sum over features
        loss = (recon_loss + self.quant_loss_weight * quant_loss).mean()
        return {
            "loss": loss,
            "recon_loss": recon_loss.mean().detach(),
            "quant_loss": quant_loss.mean().detach(),
            "codes": codes,
        }

    @torch.no_grad()
    def encode(self, x):
        z = self.encoder(x)
        return self.rq.encode(z)

    @torch.no_grad()
    def init_codebooks(self, x):
        """Explicitly init all codebooks from data (optional, forward also does lazy init)."""
        z = self.encoder(x)
        residual = z.clone()
        for vq in self.rq.vq_layers:
            if not vq.kmeans_initted:
                vq._kmeans_init(residual)
            ids = vq._distances(residual).argmin(dim=1)
            residual = residual - vq.embedding(ids)
