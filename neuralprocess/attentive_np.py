
"""Neural Process."""

from typing import Tuple, Dict

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.distributions import Normal

from .attention_layer import MultiHeadAttention, SelfAttention
from .base_np import BaseNP


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
            representation (torch.Tensor): Aggregated representation, size
                `(batch_size, num_context, r_dim)`.
        """

        h = torch.cat([x, y], dim=-1)

        # Reshape tensor: (batch, num, dim) -> (batch * num, dim)
        batch_size, num_context, h_dim = h.size()
        h = h.reshape(batch_size * num_context, h_dim)

        # Pass through MLP
        h = self.fc(h)

        # Bring back into original shape
        h = h.reshape(batch_size, num_context, -1)

        # Self-attention
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
        """Forward method f(r|x, y).

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

        # Reshape tensor: (batch, num, dim) -> (batch * num, dim)
        batch_size, num_context, h_dim = h.size()
        h = h.reshape(batch_size * num_context, h_dim)

        # Pass through MLP
        h = self.fc(h)

        # Bring back into original shape
        h = h.reshape(batch_size, num_context, -1)

        # Self-attention
        s = self.attention(h)

        # Aggregate representations for each batch
        s = s.sum(dim=1)

        # Mean and variance of N(mu(s), var(s)^0.5)
        mu = self.fc_mu(s)
        var = F.softplus(self.fc_var(s))

        return mu, var


class Decoder(nn.Module):
    """Decoder.

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
        batch_size, num_points, _ = x.size()

        # Concat inputs
        z = z.unsqueeze(1).repeat(1, num_points, 1)
        h = torch.cat([x, r, z], dim=-1)

        # Reshape tensor: (batch, num, dim) -> (batch * num, dim)
        h = h.reshape(batch_size * num_points, -1)

        # Forward
        h = self.fc(h)
        mu = self.fc_mu(h)
        var = F.softplus(self.fc_var(h))

        # Bring back into original shape
        mu = mu.reshape(batch_size, num_points, -1)
        var = var.reshape(batch_size, num_points, -1)

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

        # Encode latents
        r_c = self.encoder_r(x_context, y_context)
        r = self.attention(x_target, x_context, r_c)
        mu_z, var_z = self.encoder_z(x_context, y_context)
        z = mu_z + (var_z ** 0.5) * torch.randn(var_z.size())

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

        # Forward
        r_c = self.encoder_r(x_context, y_context)
        r = self.attention(x_target, x_context, r_c)
        mu_z_c, var_z_c = self.encoder_z(x_context, y_context)
        z = mu_z_c + (var_z_c ** 0.5) * torch.randn(var_z_c.size())
        mu, var = self.decoder(x_target, r, z)

        # Log likelihood
        dist = Normal(mu, var ** 0.5)
        log_p = dist.log_prob(y_target).sum()

        # KL divergence KL(N(mu_z_t, var_z_t^0.5) || N(mu_z_c, var_z_c^0.5))
        mu_z_t, var_z_t = self.encoder_z(x_target, y_target)
        mu_diff = mu_z_c - mu_z_t
        kl_div = ((var_z_t / var_z_c).sum(1)
                  + (mu_diff / var_z_c * mu_diff).sum(1) - mu_z_c.size(1)
                  + (var_z_c.prod(1) / var_z_t.prod(1)).log()).sum() * 0.5

        # ELBO loss
        loss = -log_p + kl_div

        return {"loss": loss, "logp_loss": -log_p, "kl_loss": kl_div}
