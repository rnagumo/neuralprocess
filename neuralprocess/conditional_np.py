
"""Conditional Neural Process."""

from typing import Tuple, Dict

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.distributions import MultivariateNormal

from .base_np import BaseNP


class Encoder(nn.Module):
    """Encoder and aggregator.

    Args:
        x_dim (int): Dimension size of x.
        y_dim (int): Dimension size of y.
        r_dim (int): Dimension size of r (representation).
    """

    def __init__(self, x_dim: int, y_dim: int, r_dim: int):
        super().__init__()

        self.fc = nn.Sequential(
            nn.Linear(x_dim + y_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, r_dim),
        )

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        """Forward method f(r|x, y).

        Args:
            x (torch.Tensor): x context data, size
                `(batch_size, num_context, x_dim)`.
            y (torch.Tensor): x context data, size
                `(batch_size, num_context, y_dim)`.

        Returns:
            representation (torch.Tensor): Aggregated representation, size
                `(batch_size, r_dim)`.
        """

        h = torch.cat([x, y], dim=-1)

        # Reshape tensor: (batch, num, dim) -> (batch * num, dim)
        batch_size, num_context, h_dim = h.size()
        h = h.reshape(batch_size * num_context, h_dim)

        # Pass through MLP
        h = self.fc(h)

        # Bring back into original shape
        h = h.reshape(batch_size, num_context, -1)

        # Aggregate representations for each batch
        r = h.sum(dim=1)

        return r


class Decoder(nn.Module):
    """Decoder.

    Args:
        x_dim (int): Dimension size of x.
        y_dim (int): Dimension size of y.
        r_dim (int): Dimension size of r (representation).
    """

    def __init__(self, x_dim: int, y_dim: int, r_dim: int):
        super().__init__()

        self.fc = nn.Sequential(
            nn.Linear(x_dim + r_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
        )

        self.fc_mu = nn.Linear(128, y_dim)
        self.fc_var = nn.Linear(128, y_dim)

    def forward(self, x: Tensor, r: Tensor) -> Tuple[Tensor, Tensor]:
        """Forward method.

        Args:
            x (torch.Tensor): x context data, size
                `(batch_size, num_points, x_dim)`.
            r (torch.Tensor): Aggregated representation, size
                `(batch_size, r_dim)`.

        Returns:
           mu (torch.Tensor): Decoded mean, size
                `(batch_size, num_points, x_dim)`.
           var (torch.Tensor): Decoded variance, size
                `(batch_size, num_points, x_dim)`.
        """

        # Data size
        batch_size, num_points, _ = x.size()

        # Concat inputs
        r = r.unsqueeze(1).repeat(1, num_points, 1)
        h = torch.cat([x, r], dim=-1)

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


class ConditionalNP(BaseNP):
    """Conditional Neural Process class.

    Args:
        x_dim (int): Dimension size of x.
        y_dim (int): Dimension size of y.
        r_dim (int): Dimension size of r (representation).

    Attributes:
        encoder (Encoder): Encoder for representation.
        decoder (Decoder): Decoder for predicting y with representation and
            query.
    """

    def __init__(self, x_dim: int, y_dim: int, r_dim: int):
        super().__init__()

        self.encoder = Encoder(x_dim, y_dim, r_dim)
        self.decoder = Decoder(x_dim, y_dim, r_dim)

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

        representation = self.encoder(x_context, y_context)
        mu, var = self.decoder(x_target, representation)
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

        mu, var = self.query(x_context, y_context, x_target)

        # Distribution
        batch_size, num_target, y_dim = var.size()
        cov = (torch.eye(y_dim).repeat(batch_size, num_target, 1, 1)
               * var.unsqueeze(-1))
        dist = MultivariateNormal(mu, cov)
        log_p = dist.log_prob(y_target).sum()

        return {"loss": -log_p}
