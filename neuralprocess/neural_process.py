
"""Neural Process."""

from typing import Tuple, Dict

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.distributions import Normal

from .base_np import BaseNP, kl_divergence_normal


class Encoder(nn.Module):
    """Encoder and aggregator.

    Args:
        x_dim (int): Dimension size of x.
        y_dim (int): Dimension size of y.
        r_dim (int): Dimension size of r (deterministic representation).
        z_dim (int): Dimension size of z (stochastic latent).
    """

    def __init__(self, x_dim: int, y_dim: int, r_dim: int, z_dim: int):
        super().__init__()

        self.fc = nn.Sequential(
            nn.Linear(x_dim + y_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, r_dim),
        )
        self.fc_mu = nn.Linear(r_dim, z_dim)
        self.fc_var = nn.Linear(r_dim, z_dim)

    def forward(self, x: Tensor, y: Tensor) -> Tuple[Tensor, Tensor]:
        """Forward method f(r|x, y).

        Args:
            x (torch.Tensor): x context data, size
                `(batch_size, num_context, x_dim)`.
            y (torch.Tensor): x context data, size
                `(batch_size, num_context, y_dim)`.

        Returns:
            mu (torch.Tensor): Aggregated mean, size
                `(batch_size, z_dim)`.
           var (torch.Tensor): Aggregated variance, size
                `(batch_size, z_dim)`.
        """

        h = torch.cat([x, y], dim=-1)
        h = self.fc(h)

        # Aggregate representations for each batch
        r = h.sum(dim=1)

        # Mu and var of N(z|mu(r), var(r)^0.5)
        mu = self.fc_mu(r)
        var = F.softplus(self.fc_var(r))

        return mu, var


class Decoder(nn.Module):
    """Decoder.

    Args:
        x_dim (int): Dimension size of x.
        y_dim (int): Dimension size of y.
        z_dim (int): Dimension size of z (stochastic latent).
    """

    def __init__(self, x_dim: int, y_dim: int, z_dim: int):
        super().__init__()

        self.fc = nn.Sequential(
            nn.Linear(x_dim + z_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
        )

        self.fc_mu = nn.Linear(128, y_dim)
        self.fc_var = nn.Linear(128, y_dim)

    def forward(self, x: Tensor, z: Tensor) -> Tuple[Tensor, Tensor]:
        """Forward method.

        Args:
            x (torch.Tensor): x context data, size
                `(batch_size, num_points, x_dim)`.
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
        h = torch.cat([x, z], dim=-1)

        # Forward
        h = self.fc(h)
        mu = self.fc_mu(h)
        var = F.softplus(self.fc_var(h))

        return mu, var


class NeuralProcess(BaseNP):
    """Neural Process class.

    Args:
        x_dim (int): Dimension size of x.
        y_dim (int): Dimension size of y.
        r_dim (int): Dimension size of r (deterministic representation).
        z_dim (int): Dimension size of z (stochastic latent).

    Attributes:
        encoder (Encoder): Encoder for representation.
        decoder (Decoder): Decoder for predicting y with representation and
            query.
    """

    def __init__(self, x_dim: int, y_dim: int, r_dim: int, z_dim: int):
        super().__init__()

        self.encoder = Encoder(x_dim, y_dim, r_dim, z_dim)
        self.decoder = Decoder(x_dim, y_dim, z_dim)

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
        mu_z, var_z = self.encoder(x_context, y_context)
        z = mu_z + (var_z ** 0.5) * torch.randn(var_z.size())

        # Query
        mu, var = self.decoder(x_target, z)
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
        mu_z_c, var_z_c = self.encoder(x_context, y_context)
        z = mu_z_c + (var_z_c ** 0.5) * torch.randn(var_z_c.size())
        mu, var = self.decoder(x_target, z)

        # Negative Log likelihood
        dist = Normal(mu, var ** 0.5)
        nll = -dist.log_prob(y_target).sum()

        # KL divergence KL(N(mu_z_t, var_z_t^0.5) || N(mu_z_c, var_z_c^0.5))
        mu_z_t, var_z_t = self.encoder(x_target, y_target)
        kl_div = kl_divergence_normal(mu_z_t, var_z_t, mu_z_c, var_z_c)
        kl_div = kl_div.sum()

        # ELBO loss
        loss = nll + kl_div

        return {"loss": loss, "nll": nll, "kl": kl_div}
