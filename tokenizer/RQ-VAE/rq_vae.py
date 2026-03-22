import torch
from torch import nn
import torch.nn.functional as F


class ResidualVectorQuantizer(nn.Module):
    def __init__(
        self,
        num_layers: int,
        codebook_size: int,
        latent_dim: int,
        ema: bool = True,
        ema_decay: float = 0.99,
        restart_unused_codes: bool = True,
        dead_code_threshold: float = 1.0,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.codebook_size = codebook_size
        self.latent_dim = latent_dim
        self.ema = ema
        self.ema_decay = ema_decay
        self.restart_unused_codes = restart_unused_codes
        self.dead_code_threshold = dead_code_threshold

        self.register_buffer(
            "codebooks",
            torch.empty(num_layers, codebook_size, latent_dim),
        )
        nn.init.xavier_uniform_(self.codebooks)

        self.register_buffer(
            "ema_cluster_size",
            torch.zeros(num_layers, codebook_size),
        )
        self.register_buffer(
            "ema_embed_avg",
            self.codebooks.clone(),
        )

    @staticmethod
    def _center_distance_for_constraint(distances: torch.Tensor):
        max_distance = distances.max()
        min_distance = distances.min()
        middle = (max_distance + min_distance) / 2.0
        amplitude = (max_distance - middle).clamp(min=1e-5)
        return (distances - middle) / amplitude

    @staticmethod
    def _sinkhorn_algorithm(distances: torch.Tensor, epsilon: float, sinkhorn_iterations: int):
        q = torch.exp(-distances / epsilon)
        b = q.shape[0]
        k = q.shape[1]
        sum_q = q.sum(dim=1, keepdim=True).sum(dim=0, keepdim=True).clamp(min=1e-12)
        q = q / sum_q
        for _ in range(sinkhorn_iterations):
            q = q / q.sum(dim=1, keepdim=True).clamp(min=1e-12)
            q = q / b
            q = q / q.sum(dim=0, keepdim=True).clamp(min=1e-12)
            q = q / k
        q = q * b
        return q

    def _assign_indices(
        self,
        distances: torch.Tensor,
        use_sk: bool = False,
        sk_epsilon: float = 0.0,
        sk_iters: int = 50,
    ):
        if (not use_sk) or sk_epsilon <= 0.0 or distances.shape[0] <= 1:
            return distances.argmin(dim=1)
        centered = self._center_distance_for_constraint(distances).double()
        q = self._sinkhorn_algorithm(centered, sk_epsilon, sk_iters)
        return torch.argmax(q, dim=1).long()

    def _quantize_once(
        self,
        x: torch.Tensor,
        codebook: torch.Tensor,
        use_sk: bool = False,
        sk_epsilon: float = 0.0,
        sk_iters: int = 50,
    ):
        x_sq = (x ** 2).sum(dim=1, keepdim=True)
        cb_sq = (codebook ** 2).sum(dim=1).unsqueeze(0)
        distances = x_sq + cb_sq - 2.0 * x @ codebook.t()
        indices = self._assign_indices(
            distances, use_sk=use_sk, sk_epsilon=sk_epsilon, sk_iters=sk_iters
        )
        quantized = codebook[indices]
        return quantized, indices, distances

    @torch.no_grad()
    def _ema_update(self, layer: int, residual: torch.Tensor, indices: torch.Tensor):
        # Keep EMA buffers in stable fp32 even when AMP casts activations to fp16/bf16.
        residual = residual.to(self.ema_embed_avg.dtype)
        one_hot = F.one_hot(indices, self.codebook_size).type_as(residual)
        cluster_size = one_hot.sum(dim=0)
        embed_sum = one_hot.t() @ residual

        self.ema_cluster_size[layer].mul_(self.ema_decay).add_(cluster_size, alpha=1.0 - self.ema_decay)
        self.ema_embed_avg[layer].mul_(self.ema_decay).add_(embed_sum, alpha=1.0 - self.ema_decay)

        if self.restart_unused_codes:
            dead = self.ema_cluster_size[layer] < self.dead_code_threshold
            if dead.any():
                n_dead = int(dead.sum().item())
                rand_idx = torch.randint(
                    0,
                    residual.shape[0],
                    size=(n_dead,),
                    device=residual.device,
                )
                new_embed = residual[rand_idx]
                self.ema_embed_avg[layer, dead] = new_embed
                self.ema_cluster_size[layer, dead] = self.dead_code_threshold

        n = self.ema_cluster_size[layer].sum()
        denom = (self.ema_cluster_size[layer] + 1e-5) / (n + self.codebook_size * 1e-5) * n
        self.codebooks[layer] = self.ema_embed_avg[layer] / denom.unsqueeze(1)

    def quantize(
        self,
        z: torch.Tensor,
        use_sk: bool = False,
        sk_epsilon: float = 0.0,
        sk_iters: int = 50,
    ):
        residual = z
        quantized_sum = torch.zeros_like(z)
        all_indices = []
        all_layer_q = []
        all_cum_q = []

        for layer in range(self.num_layers):
            q, idx, distances = self._quantize_once(
                residual,
                self.codebooks[layer],
                use_sk=(use_sk and not self.training),
                sk_epsilon=sk_epsilon,
                sk_iters=sk_iters,
            )

            if self.training and self.ema:
                self._ema_update(layer=layer, residual=residual.detach(), indices=idx.detach())
                q = self.codebooks[layer][idx]

            residual = residual - q
            quantized_sum = quantized_sum + q
            all_indices.append(idx)
            all_layer_q.append(q)
            all_cum_q.append(quantized_sum.clone())

        codes = torch.stack(all_indices, dim=1)
        return quantized_sum, codes, all_layer_q, all_cum_q

    @torch.no_grad()
    def encode(
        self,
        z: torch.Tensor,
        use_sk: bool = False,
        sk_epsilon: float = 0.0,
        sk_iters: int = 50,
    ):
        _, codes, _, _ = self.quantize(
            z, use_sk=use_sk, sk_epsilon=sk_epsilon, sk_iters=sk_iters
        )
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
        kl_weight: float = 0.0,
        balance_weight: float = 0.05,
        ema: bool = True,
        ema_decay: float = 0.99,
        restart_unused_codes: bool = True,
        dead_code_threshold: float = 1.0,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.num_layers = num_layers
        self.codebook_size = codebook_size
        self.commitment_weight = commitment_weight
        self.kl_weight = kl_weight
        self.balance_weight = balance_weight
        self.ema = ema

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.mu_head = nn.Linear(hidden_dim, latent_dim)

        self.rq = ResidualVectorQuantizer(
            num_layers=num_layers,
            codebook_size=codebook_size,
            latent_dim=latent_dim,
            ema=ema,
            ema_decay=ema_decay,
            restart_unused_codes=restart_unused_codes,
            dead_code_threshold=dead_code_threshold,
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def encode_to_latent(self, x: torch.Tensor):
        h = self.encoder(x)
        return self.mu_head(h)

    def forward(self, x: torch.Tensor):
        z = self.encode_to_latent(x)

        z_q, codes, layer_q, cum_q = self.rq.quantize(z)
        z_q_st = z + (z_q - z).detach()
        x_rec = self.decoder(z_q_st)

        recon_loss = F.mse_loss(x_rec, x)

        if self.ema:
            codebook_loss = torch.zeros((), device=x.device)
        else:
            codebook_loss = F.mse_loss(z_q, z.detach())

        commit_terms = [F.mse_loss(z, cq.detach()) for cq in cum_q]
        commit_loss = sum(commit_terms) / max(len(commit_terms), 1)

        kl_loss = torch.zeros((), device=x.device)
        # Encourage balanced code usage per layer:
        # minimize (logK - H(p_layer)), where p_layer is batch code histogram.
        balance_terms = []
        log_k = torch.log(torch.tensor(float(self.codebook_size), device=x.device))
        for layer in range(codes.shape[1]):
            counts = torch.bincount(codes[:, layer], minlength=self.codebook_size).float()
            probs = (counts / counts.sum().clamp(min=1.0)).clamp(min=1e-12)
            entropy = -(probs * probs.log()).sum()
            balance_terms.append(log_k - entropy)
        balance_loss = sum(balance_terms) / max(len(balance_terms), 1)

        loss = (
            recon_loss
            + codebook_loss
            + self.commitment_weight * commit_loss
            + self.kl_weight * kl_loss
            + self.balance_weight * balance_loss
        )
        return {
            "loss": loss,
            "recon_loss": recon_loss.detach(),
            "codebook_loss": codebook_loss.detach(),
            "commit_loss": commit_loss.detach(),
            "kl_loss": kl_loss.detach(),
            "balance_loss": balance_loss.detach(),
            "codes": codes,
            "x_rec": x_rec,
            "mu": z,
            "z_q": z_q,
            "layer_q": layer_q,
            "cum_q": cum_q,
        }

    @torch.no_grad()
    def encode(
        self,
        x: torch.Tensor,
        use_sk: bool = False,
        sk_epsilon: float = 0.0,
        sk_iters: int = 50,
    ):
        z = self.encode_to_latent(x)
        return self.rq.encode(z, use_sk=use_sk, sk_epsilon=sk_epsilon, sk_iters=sk_iters)
