
"""Convoluional Conditional Neural Process.

J. Gordon et al., "Convolutional Conditional Neural Processes"
http://arxiv.org/abs/1910.13556

Reference)
https://github.com/cambridge-mlg/convcnp
"""

from typing import Tuple, Dict, Union

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .base_np import BaseNP, nll_normal


def rbf_kernel(x0: Tensor, x1: Tensor, sigma: Tensor) -> Tensor:
    """RBF kernel.

    Args:
        x0 (torch.Tensor): First input data of size `(b, n, x_dim)`.
        x1 (torch.Tensor): Second input data of size `(b, m, x_dim)`.
        sigma (torch.Tensor): Sigma scale parameter of size `(c,)`.

    Returns:
        kernel (torch.Tensor): kernel matrix of size `(b, n, m, c)`.
    """

    # Data size
    batch_size, num_points, x_dim = x0.size()

    # Expand and take diff (b, n, m, 1)
    x0_unsq = x0.unsqueeze(2)  # (b, n, 1, x_dim)
    x1_unsq = x1.unsqueeze(1)  # (b, 1, m, x_dim)
    diff = (x1_unsq - x0_unsq).sum(-1).unsqueeze(-1)

    # Scale parameter (1, 1, 1, c)
    scale = sigma.exp().view(1, 1, 1, -1)

    # Kernel value (b, n, m, c)
    kernel = (-0.5 * (diff / scale) ** 2).exp()

    return kernel


def to_multiple(x: Union[int, float], multiple: int) -> Union[int, float]:
    """Convert `x` to the nearest above multiple.

    Args:
        x (number): Number to round up.
        multiple (int): Multiple to round up to.

    Returns:
        number: `x` rounded to the nearest above multiple of `multiple`.
    """

    if x % multiple == 0:
        return x
    else:
        return x + multiple - x % multiple


class Encoder(nn.Module):
    """Encoder for ConvCNP.

    Args:
        y_dim (int): Dimension size of y.
        h_dim (int): Dimension size of h (representation).
        length_scale (int): Initial length scale of RBF kernel.
    """

    def __init__(self, y_dim: int, h_dim: int, length_scale: int):
        super().__init__()

        self.fc = nn.Sequential(
            nn.Linear(y_dim + 1, h_dim),
        )

        # Scale parameter for RBF kernel
        self.sigma = nn.Parameter(
            torch.ones(y_dim + 1) * math.log(length_scale),
            requires_grad=True)

    def forward(self, x: Tensor, y: Tensor, t: Tensor) -> Tensor:
        """Forward

        Args:
            x (torch.Tensor): Input observation, size `(b, n, x_dim)`.
            y (torch.Tensor): Output observation, size `(b, n, y_dim)`.
            t (torch.Tensor): Uniform grid, size `(b, m, x_dim)`.

        Returns:
            z (torch.Tensor): Encoded representation, size `(b, m, h_dim)`.
        """

        # Data size
        batch_size, n_in, _ = x.size()

        # Compute weights (b, n, m, y_dim + 1)
        weight = rbf_kernel(x, t, self.sigma)

        # Add extra 'density' channel (b, n, y_dim + 1)
        density = y.new_ones((batch_size, n_in, 1))
        h = torch.cat([density, y], dim=-1)

        # Perform compute (b, n, m, y_dim + 1)
        h = h.unsqueeze(2) * weight

        # Sum over inputs (b, m, y_dim + 1)
        h = h.sum(1)

        # Normalize 'convoluion' channels excluding 'density' channel
        density = h[..., :1]
        conv = h[..., 1:]
        normalized = conv / (density + 1e-8)
        h = torch.cat([density, normalized], dim=-1)

        # Apply point-wise function
        h = self.fc(h)

        return h


class Decoder(nn.Module):
    """Encoder for ConvCNP.

    Args:
        h_dim (int): Dimension size of representation h.
        y_dim (int): Dimension size of output y.
        length_scale (int): Initial length scale of RBF kernel.
    """

    def __init__(self, h_dim: int, y_dim: int, length_scale: int):
        super().__init__()

        self.fc = nn.Sequential(
            nn.Linear(h_dim, y_dim),
        )

        # Scale parameter for RBF kernel
        self.sigma = nn.Parameter(
            torch.ones(h_dim) * math.log(length_scale),
            requires_grad=True)

    def forward(self, t: Tensor, h: Tensor, x: Tensor) -> Tensor:
        """Forward

        Args:
            t (torch.Tensor): Uniform grid, size `(b, n, x_dim)`.
            h (torch.Tensor): Latent, size `(b, n, h_dim)`.
            x (torch.Tensor): X target data, size `(b, m, x_dim)`.

        Returns:
            y_out (torch.Tensor): Encoded representation, size `(b, m, y_dim)`.
        """

        # Data size
        batch_size, n_in, _ = t.size()

        # Compute weights (b, n, m, y_dim)
        weight = rbf_kernel(t, x, self.sigma)

        # Perform compute (b, n, m, y_dim)
        y_out = h.unsqueeze(2) * weight

        # Sum over inputs (b, m, y_dim)
        y_out = y_out.sum(1)

        # Apply point-wise function
        y_out = self.fc(y_out)

        return y_out


class ConvCNP(BaseNP):
    """One-dimensional Convolutional Conditional Neural Process model.

    Args:
        x_dim (int): Dimension size of x input.
        y_dim (int): Dimension size of y output.
        h_dim (int, optional): Dimension size of hidden representation.
        points_per_unit (int, optional): Number of grid points per unit.
    """

    def __init__(self, x_dim: int, y_dim: int, h_dim: int = 8,
                 points_per_unit: int = 64):
        super().__init__()

        self.points_per_unit = points_per_unit

        # Convolution layer
        self.conv = nn.Sequential(
            nn.Conv1d(h_dim, 16, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.Conv1d(32, 16, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.Conv1d(16, h_dim, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
        )
        num_layers = 0
        self.multipler = 2 ** num_layers

        # Layers
        length_scale = 2.0 / points_per_unit
        self.encoder = Encoder(y_dim, h_dim, length_scale)
        self.decoder_mu = Decoder(h_dim, y_dim, length_scale)
        self.decoder_var = Decoder(h_dim, y_dim, length_scale)

    def sample(self, x_context: Tensor, y_context: Tensor, x_target: Tensor
               ) -> Tuple[Tensor, Tensor]:
        """Samples queried y target.

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

        # Data size
        batch = x_context.size(0)

        # Device
        device = x_context.device

        # Determin thg grid (batch, num_points, 1)
        x_min = min(x_context.min().item(), x_target.min().item(), -2) - 0.1
        x_max = max(x_context.max().item(), x_target.max().item(), 2) + 0.1
        num_points = int(to_multiple(self.points_per_unit * (x_max - x_min),
                                     self.multipler))
        x_grid = torch.linspace(x_min, x_max, num_points).to(device)
        x_grid = x_grid.view(1, -1, 1).repeat(batch, 1, 1)

        # Encode (batch, num_points, h_dim)
        h = torch.sigmoid(self.encoder(x_context, y_context, x_grid))

        # Convolution (batch, num_points, h_dim)
        h = h.permute(0, 2, 1)
        h = self.conv(h)
        h = h.permute(0, 2, 1)

        # Decode
        mu = self.decoder_mu(x_grid, h, x_target)
        var = F.softplus(self.decoder_var(x_grid, h, x_target))

        return mu, var

    def loss_func(self, x_context: Tensor, y_context: Tensor, x_target: Tensor,
                  y_target: Tensor) -> Dict[str, Tensor]:
        """Loss function for ELBO.

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

        mu, var = self.sample(x_context, y_context, x_target)

        # Negative Log likelihood
        nll = nll_normal(y_target, mu, var)
        nll = nll.mean()

        return {"loss": nll}
