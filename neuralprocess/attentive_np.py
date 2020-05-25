
"""Attentive Neural Process.

H. Kim et al., "Attentive Neural Processes".
http://arxiv.org/abs/1901.05761
"""

from typing import Tuple, Dict

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.distributions import Normal

from .attention_layer import MultiHeadAttention, SelfAttention
from .base_np import BaseNP, kl_divergence_normal


class DeterministicEncoder(nn.Module):
    """Deterministic encoder.

    Args:
        x_dim (int): Dimension size of x.
        y_dim (int): Dimension size of y.
        r_dim (int): Dimension size of r (deterministic representation).
        n_head (int): Number of head in self-attention module.
    """

    def __init__(self, x_dim: int, y_dim: int, r_dim: int, n_head: int):
        super().__init__()

        self.fc = nn.Sequential(
            nn.Linear(x_dim + y_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, r_dim),
        )
        self.attention = SelfAttention(r_dim, r_dim, n_head)

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        """Forward method f(r|x, y).

        Args:
            x (torch.Tensor): x context data, size
                `(batch_size, num_context, x_dim)`.
            y (torch.Tensor): x context data, size
                `(batch_size, num_context, y_dim)`.

        Returns:
            r (torch.Tensor): Aggregated representation, size
                `(batch_size, num_context, r_dim)`.
        """

        h = torch.cat([x, y], dim=-1)
        h = self.fc(h)
        r = self.attention(h)

        return r


class StochasticEncoder(nn.Module):
    """Stochastic encoder.

    1. Encode each context to representation `s`.
    2. Aggregate all representations `s_C`.
    3. Sample stochastic latent `z` ~ p(z|s_C).

    Args:
        x_dim (int): Dimension size of x.
        y_dim (int): Dimension size of y.
        s_dim (int): Dimension size of s (deterministic representation).
        z_dim (int): Dimension size of z (stochastic latent).
        n_head (int): Number of head in self-attention module.
    """

    def __init__(self, x_dim: int, y_dim: int, s_dim: int, z_dim: int,
                 n_head: int):
        super().__init__()

        self.fc = nn.Sequential(
            nn.Linear(x_dim + y_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, s_dim),
        )
        self.fc_mu = nn.Linear(s_dim, z_dim)
        self.fc_var = nn.Linear(s_dim, z_dim)

        self.attention = SelfAttention(s_dim, s_dim, n_head)

    def forward(self, x: Tensor, y: Tensor) -> Tuple[Tensor, Tensor]:
        """Forward method p(z|x, y): p(z|s)f(s|x, y).

        Args:
            x (torch.Tensor): x context data, size
                `(batch_size, num_context, x_dim)`.
            y (torch.Tensor): x context data, size
                `(batch_size, num_context, y_dim)`.

        Returns:
            mu (torch.Tensor): Sampled aggregated mean, size
                `(batch_size, z_dim)`.
            var (torch.Tensor): Sampled aggregated variance, size
                `(batch_size, z_dim)`.
        """

        h = torch.cat([x, y], dim=-1)
        h = self.fc(h)
        s = self.attention(h)

        # Aggregate representations for all contexts per batch and dimension.
        # (batch_size, num_context, s_dim) -> (batch_size, s_dim)
        s = s.mean(dim=1)

        # Mean and variance of N(mu(s), var(s)^0.5)
        mu = self.fc_mu(s)
        var = F.softplus(self.fc_var(s))

        # Bounds variance > 0.01 (original code: sigma > 0.1)
        var = var + 0.01

        return mu, var


class Decoder(nn.Module):
    """Decoder p(y|x, r, z).

    Args:
        x_dim (int): Dimension size of x.
        y_dim (int): Dimension size of y.
        r_dim (int): Dimension size of r (deterministic representation).
        z_dim (int): Dimension size of z (stochastic latent).
    """

    def __init__(self, x_dim: int, y_dim: int, r_dim: int, z_dim: int):
        super().__init__()

        self.fc = nn.Sequential(
            nn.Linear(x_dim + r_dim + z_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
        )

        self.fc_mu = nn.Linear(128, y_dim)
        self.fc_var = nn.Linear(128, y_dim)

    def forward(self, x: Tensor, r: Tensor, z: Tensor
                ) -> Tuple[Tensor, Tensor]:
        """Forward method.

        Args:
            x (torch.Tensor): x context data, size
                `(batch_size, num_points, x_dim)`.
            r (torch.Tensor): Deterministic representation, size
                `(batch_size, num_points, r_dim)`.
            z (torch.Tensor): Stochastic latents, size `(batch_size, z_dim)`.

        Returns:
            mu (torch.Tensor): Decoded mean, size
                `(batch_size, num_points, x_dim)`.
            var (torch.Tensor): Decoded variance, size
                `(batch_size, num_points, x_dim)`.
        """

        # Data size
        num_points = x.size(1)

        # Concat inputs
        z = z.unsqueeze(1).repeat(1, num_points, 1)
        h = torch.cat([x, r, z], dim=-1)

        # Forward
        h = self.fc(h)
        mu = self.fc_mu(h)
        var = F.softplus(self.fc_var(h))

        # Bounds variance > 0.01 (original code: sigma > 0.1)
        var = var + 0.01

        return mu, var


class AttentiveNP(BaseNP):
    """Neural Process class.

    Args:
        x_dim (int): Dimension size of x.
        y_dim (int): Dimension size of y.
        r_dim (int): Dimension size of r (deterministic representation).
        s_dim (int): Dimension size of s (deterministic representation).
        z_dim (int): Dimension size of z (stochastic latent).
        n_head (int): Number of head in self-attention module.

    Attributes:
        encoder_r (DeterministicEncoder): Encoder for deterministic
            representation `r`.
        encoder_z (StochasticEncoder): Encoder for stochastic latent `z`.
        decoder (Decoder): Decoder for predicting y with representation and
            query.
        attention (MultiHeadAttention): Cross attention layer.
    """

    def __init__(self, x_dim: int, y_dim: int, r_dim: int, s_dim: int,
                 z_dim: int, n_head: int):
        super().__init__()

        self.encoder_r = DeterministicEncoder(x_dim, y_dim, r_dim, n_head)
        self.encoder_z = StochasticEncoder(x_dim, y_dim, s_dim, z_dim, n_head)
        self.decoder = Decoder(x_dim, y_dim, r_dim, z_dim)
        self.attention = MultiHeadAttention(x_dim, r_dim, x_dim, r_dim, n_head)

    def query(self, x_context: Tensor, y_context: Tensor, x_target: Tensor
              ) -> Tuple[Tensor, Tensor]:
        """Query y target.

        Args:
            x_context (torch.Tensor): x for context, size
                `(batch_size, num_context, x_dim)`.
            y_context (torch.Tensor): y for context, size
                `(batch_size, num_context, y_dim)`.
            x_target (torch.Tensor): x for target, size
                `(batch_size, num_target, x_dim)`.

        Returns:
            mu (torch.Tensor): y queried by target x and encoded
                representation, size `(batch_size, num_target, y_dim)`.
            var (torch.Tensor): Variance of y, size
                `(batch_size, num_target, y_dim)`.
        """

        # Encode representations
        r_c = self.encoder_r(x_context, y_context)
        r = self.attention(x_target, x_context, r_c)

        # Sample global latents
        mu_z, var_z = self.encoder_z(x_context, y_context)
        z = mu_z + (var_z ** 0.5) * torch.randn_like(var_z)

        # Query
        mu, var = self.decoder(x_target, r, z)
        return mu, var

    def loss_func(self, x_context: Tensor, y_context: Tensor, x_target: Tensor,
                  y_target: Tensor) -> Dict[str, Tensor]:
        """Loss function for the negative conditional log probability.

        Args:
            x_context (torch.Tensor): x for context, size
                `(batch_size, num_context, x_dim)`.
            y_context (torch.Tensor): y for context, size
                `(batch_size, num_context, y_dim)`.
            x_target (torch.Tensor): x for target, size
                `(batch_size, num_target, x_dim)`.
            y_target (torch.Tensor): y for target, size
                `(batch_size, num_target, y_dim)`.

        Returns:
            loss_dict (dict of [str, torch.Tensor]): Calculated loss.
        """

        # Concat context and target: (batch, num_context + num_target, dim)
        x_cat = torch.cat([x_context, x_target], dim=1)
        y_cat = torch.cat([y_context, y_target], dim=1)

        # Stochastic latents
        mu_z_t, var_z_t = self.encoder_z(x_cat, y_cat)
        z = mu_z_t + (var_z_t ** 0.5) * torch.randn_like(var_z_t)

        # Deterministic representations
        r_c = self.encoder_r(x_context, y_context)
        r = self.attention(x_cat, x_context, r_c)

        # Negative Log likelihood
        mu, var = self.decoder(x_cat, r, z)
        dist = Normal(mu, var ** 0.5)
        nll = -dist.log_prob(y_cat).mean()

        # KL divergence KL(N(mu_z_t, var_z_t^0.5) || N(mu_z_c, var_z_c^0.5))
        mu_z_c, var_z_c = self.encoder_z(x_context, y_context)
        kl_div = kl_divergence_normal(mu_z_t, var_z_t, mu_z_c, var_z_c)
        kl_div = kl_div.mean()

        # ELBO loss
        loss = nll + kl_div

        return {"loss": loss, "nll": nll, "kl": kl_div}
