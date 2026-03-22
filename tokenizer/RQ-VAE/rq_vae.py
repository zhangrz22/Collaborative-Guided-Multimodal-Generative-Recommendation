import torch
from torch import nn
import torch.nn.functional as F


class ResidualVectorQuantizer(nn.Module):
    def __init__(self, num_layers: int, codebook_size: int, latent_dim: int):
        super().__init__()
        self.num_layers = num_layers
        self.codebook_size = codebook_size
        self.latent_dim = latent_dim
        self.codebooks = nn.Parameter(
            torch.empty(num_layers, codebook_size, latent_dim)
        )
        nn.init.xavier_uniform_(self.codebooks)

    def _quantize_once(self, x: torch.Tensor, codebook: torch.Tensor):
        x_sq = (x ** 2).sum(dim=1, keepdim=True)
        cb_sq = (codebook ** 2).sum(dim=1).unsqueeze(0)
        distances = x_sq + cb_sq - 2.0 * x @ codebook.t()
        indices = distances.argmin(dim=1)
        quantized = codebook[indices]
        return quantized, indices

    def quantize(self, z: torch.Tensor):
        residual = z
        quantized_sum = torch.zeros_like(z)
        all_indices = []
        all_quantized = []
        for layer in range(self.num_layers):
            q, idx = self._quantize_once(residual, self.codebooks[layer])
            residual = residual - q
            quantized_sum = quantized_sum + q
            all_indices.append(idx)
            all_quantized.append(q)
        codes = torch.stack(all_indices, dim=1)
        return quantized_sum, codes, all_quantized

    @torch.no_grad()
    def encode(self, z: torch.Tensor):
        _, codes, _ = self.quantize(z)
        return codes

    def decode(self, codes: torch.Tensor):
        out = torch.zeros(
            (codes.shape[0], self.latent_dim),
            dtype=self.codebooks.dtype,
            device=codes.device,
        )
        n_layers = min(codes.shape[1], self.num_layers)
        for layer in range(n_layers):
            out = out + self.codebooks[layer][codes[:, layer]]
        return out


class RQVAE(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 1024,
        latent_dim: int = 256,
        num_layers: int = 4,
        codebook_size: int = 256,
        commitment_weight: float = 0.25,
        kl_weight: float = 1e-4,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.num_layers = num_layers
        self.codebook_size = codebook_size
        self.commitment_weight = commitment_weight
        self.kl_weight = kl_weight

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.mu_head = nn.Linear(hidden_dim, latent_dim)
        self.logvar_head = nn.Linear(hidden_dim, latent_dim)

        self.rq = ResidualVectorQuantizer(
            num_layers=num_layers, codebook_size=codebook_size, latent_dim=latent_dim
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def encode_to_latent(self, x: torch.Tensor):
        h = self.encoder(x)
        mu = self.mu_head(h)
        logvar = self.logvar_head(h).clamp(-10.0, 10.0)
        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x: torch.Tensor):
        mu, logvar = self.encode_to_latent(x)
        z = self.reparameterize(mu, logvar)

        z_q, codes, layer_q = self.rq.quantize(z)
        z_q_st = z + (z_q - z).detach()
        x_rec = self.decoder(z_q_st)

        recon_loss = F.mse_loss(x_rec, x)
        codebook_loss = F.mse_loss(z_q, z.detach())
        commit_loss = F.mse_loss(z, z_q.detach())
        kl_loss = -0.5 * torch.mean(1.0 + logvar - mu.pow(2) - logvar.exp())

        loss = (
            recon_loss
            + codebook_loss
            + self.commitment_weight * commit_loss
            + self.kl_weight * kl_loss
        )
        return {
            "loss": loss,
            "recon_loss": recon_loss.detach(),
            "codebook_loss": codebook_loss.detach(),
            "commit_loss": commit_loss.detach(),
            "kl_loss": kl_loss.detach(),
            "codes": codes,
            "x_rec": x_rec,
            "mu": mu,
            "z_q": z_q,
            "layer_q": layer_q,
        }

    @torch.no_grad()
    def encode(self, x: torch.Tensor):
        mu, _ = self.encode_to_latent(x)
        return self.rq.encode(mu)

