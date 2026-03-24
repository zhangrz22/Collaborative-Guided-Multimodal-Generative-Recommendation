import torch
from torch import nn
import torch.nn.functional as F


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


class VectorQuantizer(nn.Module):

    def __init__(self, n_e: int, e_dim: int, commitment_weight: float = 0.25,
                 sk_epsilon: float = 0.0, sk_iters: int = 50):
        super().__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.commitment_weight = commitment_weight
        self.sk_epsilon = sk_epsilon
        self.sk_iters = sk_iters
        self.embedding = nn.Embedding(n_e, e_dim)
        self.embedding.weight.data.uniform_(-1.0 / n_e, 1.0 / n_e)

    def _distances(self, x: torch.Tensor):
        return (
            (x ** 2).sum(1, keepdim=True)
            + (self.embedding.weight ** 2).sum(1).unsqueeze(0)
            - 2.0 * x @ self.embedding.weight.t()
        )

    @torch.no_grad()
    def init_codebook(self, x: torch.Tensor, n_iters: int = 50):
        centers = kmeans_torch(x.detach(), self.n_e, n_iters=n_iters)
        self.embedding.weight.copy_(centers)

    def forward(self, x: torch.Tensor, use_sk: bool = False):
        d = self._distances(x)
        if use_sk and self.sk_epsilon > 0 and x.shape[0] > 1:
            max_d, min_d = d.max(), d.min()
            centered = (d - (max_d + min_d) / 2) / ((max_d - min_d) / 2).clamp(min=1e-5)
            Q = sinkhorn(centered.double(), self.sk_epsilon, self.sk_iters)
            indices = Q.argmax(dim=1).long()
        else:
            indices = d.argmin(dim=1)
        x_q = self.embedding(indices)

        commitment_loss = F.mse_loss(x_q.detach(), x)
        codebook_loss = F.mse_loss(x_q, x.detach())
        loss = codebook_loss + self.commitment_weight * commitment_loss

        x_q = x + (x_q - x).detach()
        return x_q, loss, indices


class ResidualVQ(nn.Module):

    def __init__(self, n_e_list: list, e_dim: int, commitment_weight: float = 0.25,
                 sk_epsilons: list = None, sk_iters: int = 50):
        super().__init__()
        n_layers = len(n_e_list)
        if sk_epsilons is None:
            sk_epsilons = [0.0] * n_layers
        self.vq_layers = nn.ModuleList([
            VectorQuantizer(n_e, e_dim, commitment_weight=commitment_weight,
                            sk_epsilon=eps, sk_iters=sk_iters)
            for n_e, eps in zip(n_e_list, sk_epsilons)
        ])

    @torch.no_grad()
    def init_codebooks(self, z: torch.Tensor, n_iters: int = 50):
        residual = z.clone()
        for vq in self.vq_layers:
            vq.init_codebook(residual, n_iters=n_iters)
            x_q = vq.embedding(vq._distances(residual).argmin(dim=1))
            residual = residual - x_q

    def forward(self, z: torch.Tensor, use_sk: bool = False):
        residual = z
        x_q_sum = torch.zeros_like(z)
        all_losses = []
        all_indices = []
        for vq in self.vq_layers:
            x_q, loss, indices = vq(residual, use_sk=use_sk)
            residual = residual - x_q
            x_q_sum = x_q_sum + x_q
            all_losses.append(loss)
            all_indices.append(indices)
        return x_q_sum, torch.stack(all_losses).mean(), torch.stack(all_indices, dim=1)

    @torch.no_grad()
    def encode(self, z: torch.Tensor, use_sk: bool = False):
        _, _, codes = self.forward(z, use_sk=use_sk)
        return codes


class RQVAE(nn.Module):

    def __init__(self, in_dim: int, e_dim: int = 32,
                 n_e_list: list = None,
                 encoder_dims: list = None,
                 commitment_weight: float = 0.25,
                 sk_epsilons: list = None, sk_iters: int = 50,
                 quant_loss_weight: float = 1.0):
        super().__init__()
        if n_e_list is None:
            n_e_list = [256, 256, 256, 256]
        if encoder_dims is None:
            encoder_dims = [1024, 512, 256, 128]

        self.quant_loss_weight = quant_loss_weight

        # Encoder: in_dim -> ... -> e_dim
        enc_layers = []
        dims = [in_dim] + encoder_dims + [e_dim]
        for i in range(len(dims) - 1):
            enc_layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                enc_layers.append(nn.ReLU())
        self.encoder = nn.Sequential(*enc_layers)

        # Decoder: e_dim -> ... -> in_dim (mirror)
        dec_layers = []
        dims_rev = dims[::-1]
        for i in range(len(dims_rev) - 1):
            dec_layers.append(nn.Linear(dims_rev[i], dims_rev[i + 1]))
            if i < len(dims_rev) - 2:
                dec_layers.append(nn.ReLU())
        self.decoder = nn.Sequential(*dec_layers)

        self.rq = ResidualVQ(
            n_e_list=n_e_list, e_dim=e_dim,
            commitment_weight=commitment_weight,
            sk_epsilons=sk_epsilons, sk_iters=sk_iters,
        )

    def forward(self, x: torch.Tensor, use_sk: bool = False):
        z = self.encoder(x)
        z_q, quant_loss, codes = self.rq(z, use_sk=use_sk)
        z_q_st = z + (z_q - z).detach()
        x_rec = self.decoder(z_q_st)
        recon_loss = F.mse_loss(x_rec, x)
        loss = recon_loss + self.quant_loss_weight * quant_loss
        return {
            "loss": loss,
            "recon_loss": recon_loss.detach(),
            "quant_loss": quant_loss.detach(),
            "codes": codes,
        }

    @torch.no_grad()
    def encode(self, x: torch.Tensor, use_sk: bool = False):
        z = self.encoder(x)
        return self.rq.encode(z, use_sk=use_sk)

    @torch.no_grad()
    def init_codebooks(self, x: torch.Tensor, n_iters: int = 50):
        z = self.encoder(x)
        self.rq.init_codebooks(z, n_iters=n_iters)
