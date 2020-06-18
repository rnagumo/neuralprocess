
"""Sequential Neural Processes.

G. Singh et al., "Sequential Neural Processes".
http://arxiv.org/abs/1906.10264
"""

from typing import Tuple, Dict, Optional

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .base_np import BaseNP, kl_divergence_normal, nll_normal


class DeterministicEncoder(nn.Module):
    """Encoder and aggregator r = f(x, y).

    Args:
        x_dim (int): Dimension size of x.
        y_dim (int): Dimension size of y.
        r_dim (int): Dimension size of r (representation).
    """

    def __init__(self, x_dim: int, y_dim: int, r_dim: int) -> None:
        super().__init__()

        self.fc = nn.Sequential(
            nn.Linear(x_dim + y_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, r_dim),
        )

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        """Forward method r = f(x, y).

        Args:
            x (torch.Tensor): x context data, size
                `(batch_size, num_context, x_dim)`.
            y (torch.Tensor): x context data, size
                `(batch_size, num_context, y_dim)`.

        Returns:
            r (torch.Tensor): Aggregated representation, size
                `(batch_size, r_dim)`.
        """

        h = torch.cat([x, y], dim=-1)
        h = self.fc(h)

        # Aggregate representations for all contexts per batch and dimension.
        # (batch_size, num_context, r_dim) -> (batch_size, r_dim)
        r = h.mean(dim=1)

        return r


class StochasticEncoder(nn.Module):
    """Stochastic encoder p(z|h, r).

    Args:
        h_dim (int): Dimension size of h (rnn hidden state).
        r_dim (int): Dimension size of r (representation).
        z_dim (int): Dimension size of z (stochastic latent).
    """

    def __init__(self, h_dim: int, r_dim: int, z_dim: int) -> None:
        super().__init__()

        self.fc = nn.Sequential(
            nn.Linear(h_dim + r_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
        )
        self.fc_mu = nn.Linear(128, z_dim)
        self.fc_var = nn.Linear(128, z_dim)

    def forward(self, h: Tensor, r: Tensor) -> Tuple[Tensor, Tensor]:
        """Forward method p(z|h, r).

        Args:
            h (torch.Tensor): Hidden state, size `(batch_size, h_dim)`.
            r (torch.Tensor): Representation, size `(batch_size, r_dim)`.

        Returns:
            mu (torch.Tensor): Encoded aggregated mean, size
                `(batch_size, z_dim)`.
            var (torch.Tensor): Encoded aggregated variance, size
                `(batch_size, z_dim)`.
        """

        h = torch.cat([h, r], dim=-1)
        s = self.fc(h)

        # Mean and variance of N(mu(s), var(s)^0.5)
        mu = self.fc_mu(s)
        var = F.softplus(self.fc_var(s))

        return mu, var


class Decoder(nn.Module):
    """Decoder.

    Args:
        x_dim (int): Dimension size of x.
        y_dim (int): Dimension size of y.
        h_dim (int): Dimension size of h (rnn hidden state).
        z_dim (int): Dimension size of z (stochastic latent).
    """

    def __init__(self, x_dim: int, y_dim: int, h_dim: int, z_dim: int) -> None:
        super().__init__()

        self.fc = nn.Sequential(
            nn.Linear(x_dim + h_dim + z_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
        )

        self.fc_mu = nn.Linear(64, y_dim)
        self.fc_var = nn.Linear(64, y_dim)

    def forward(self, x: Tensor, h: Tensor, z: Tensor
                ) -> Tuple[Tensor, Tensor]:
        """Forward method.

        Args:
            x (torch.Tensor): x context data, size
                `(batch_size, num_points, x_dim)`.
            h (torch.Tensor): RNN hidden state, size `(batch_size, h_dim)`.
            z (torch.Tensor): Stochastic latent, size `(batch_size, z_dim)`.

        Returns:
            mu (torch.Tensor): Decoded mean, size
                `(batch_size, num_points, y_dim)`.
            var (torch.Tensor): Decoded variance, size
                `(batch_size, num_points, y_dim)`.
        """

        # Data size
        num_points = x.size(1)

        # Concat inputs
        h = h.unsqueeze(1).repeat(1, num_points, 1)
        z = z.unsqueeze(1).repeat(1, num_points, 1)
        h = torch.cat([x, h, z], dim=-1)

        # Forward
        h = self.fc(h)
        mu = self.fc_mu(h)
        var = F.softplus(self.fc_var(h))

        return mu, var


class SequentialNP(BaseNP):
    """Sequential Neural Process class.

    Args:
        x_dim (int): Dimension size of x.
        y_dim (int): Dimension size of y.
        r_dim (int): Dimension size of r (representation).
        z_dim (int): Dimension size of z (stochastic latent).
        h_dim (int): Dimension size of h (rnn hidden state).

    Attributes:
        encoder_r (DeterministicEncoder): Encoder for deterministic
            representation `r`.
        encoder_z (StochasticEncoder): Encoder for stochastic latent `z`.
        decoder (Decoder): Decoder for predicting y with representation and
            query.
        rnn_cell (nn.RNNCell): RNN for sequence.
    """

    def __init__(self, x_dim: int, y_dim: int, r_dim: int, z_dim: int,
                 h_dim: int) -> None:
        super().__init__()

        self.z_dim = z_dim
        self.h_dim = h_dim

        self.encoder_r = DeterministicEncoder(x_dim, y_dim, r_dim)
        self.encoder_z = StochasticEncoder(h_dim, r_dim, z_dim)
        self.decoder = Decoder(x_dim, y_dim, h_dim, z_dim)
        self.rnn_cell = nn.RNNCell(r_dim + z_dim, h_dim)

    def sample(self, x_context: Tensor, y_context: Tensor, x_target: Tensor,
               y_target: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        """Samples queried y target.

        Args:
            x_context (torch.Tensor): x for context, size
                `(batch_size, seq_len, num_context, x_dim)`.
            y_context (torch.Tensor): y for context, size
                `(batch_size, seq_len, num_context, y_dim)`.
            x_target (torch.Tensor): x for target, size
                `(batch_size, seq_len, num_target, x_dim)`.
            y_target (torch.Tensor, optional): y for target, size
                `(batch_size, recon_len, num_target, y_dim)`.

        Returns:
            mu (torch.Tensor): Mean of y, size
                `(batch_size, seq_len, num_target, y_dim)`.
            var (torch.Tensor): Variance of y, size
                `(batch_size, seq_len, num_target, y_dim)`.
        """

        # Initial parameters
        batch, seq_len, num_target, _ = x_target.size()
        recon_len = y_target.size(1) if y_target is not None else 0

        h_t = x_target.new_zeros((batch, self.h_dim))
        z_t = x_target.new_zeros((batch, self.z_dim))

        # Sample
        # t < recon_len: Reconstruct observations
        # t >= recon_len: Sample from prior
        y_mu_list = []
        y_var_list = []
        for t in range(seq_len):
            # 1. Encode context: r = f(x, y)
            if y_target is not None and t < recon_len:
                r_t = self.encoder_r(x_target[:, t], y_target[:, t])
            else:
                r_t = self.encoder_r(x_context[:, t], y_context[:, t])

            # 2. Update rnn: h_t = rnn(z, r, h_{t-1})
            h_t = self.rnn_cell(torch.cat([z_t, r_t], dim=-1), h_t)

            # 3. Sample stochastic latent: z ~ p(h, r)
            z_t_mu, z_t_var = self.encoder_z(h_t, r_t)
            z_t = z_t_mu + z_t_var ** 0.5 * torch.randn_like(z_t_var)

            # 4. Render target y: y = renderer(x, z, h)
            y_t_mu, y_t_var = self.decoder(x_target[:, t], z_t, h_t)

            y_mu_list.append(y_t_mu)
            y_var_list.append(y_t_var)

        # Stack and resize
        y_mu = torch.stack(y_mu_list)
        y_var = torch.stack(y_var_list)

        y_mu = y_mu.transpose(0, 1)
        y_var = y_var.transpose(0, 1)

        return y_mu, y_var

    def loss_func(self, x_context: Tensor, y_context: Tensor, x_target: Tensor,
                  y_target: Tensor) -> Dict[str, Tensor]:
        """Loss function for the negative conditional log probability.

        Args:
            x_context (torch.Tensor): x for context, size
                `(batch_size, seq_len, num_context, x_dim)`.
            y_context (torch.Tensor): y for context, size
                `(batch_size, seq_len, num_context, y_dim)`.
            x_target (torch.Tensor): x for target, size
                `(batch_size, seq_len, num_target, x_dim)`.
            y_target (torch.Tensor): y for target, size
                `(batch_size, seq_len, num_target, y_dim)`.

        Returns:
            loss_dict (dict of [str, torch.Tensor]): Calculated loss.
        """

        # Initial parameters
        batch, seq_len, *_ = x_context.size()
        h_t = x_target.new_zeros((batch, self.h_dim))
        z_t = x_target.new_zeros((batch, self.z_dim))

        nll_loss = x_target.new_zeros((batch,))
        kl_loss = x_target.new_zeros((batch,))

        for t in range(seq_len):
            # 1. Encode context and target: r = f(x, y)
            r_t_ctx = self.encoder_r(x_context[:, t], y_context[:, t])
            r_t_tgt = self.encoder_r(x_target[:, t], y_target[:, t])

            # 2. Update rnn: h_t = rnn(z, r, h_{t-1})
            h_t = self.rnn_cell(torch.cat([z_t, r_t_ctx], dim=-1), h_t)

            # 3. Sample stochastic latent z ~ p(h, r), q(h, r)
            z_t_mu_ctx, z_t_var_ctx = self.encoder_z(h_t, r_t_ctx)

            z_t_mu_tgt, z_t_var_tgt = self.encoder_z(h_t, r_t_tgt)
            z_t = (z_t_mu_tgt
                   + z_t_var_tgt ** 0.5 * torch.randn_like(z_t_var_tgt))

            # 4. Render target y: y = renderer(x, z, h)
            y_t_mu, y_t_var = self.decoder(x_target[:, t], z_t, h_t)

            # Loss
            nll_loss += nll_normal(y_target[:, t], y_t_mu, y_t_var).sum(-1)
            kl_loss += kl_divergence_normal(
                z_t_mu_tgt, z_t_var_tgt, z_t_mu_ctx, z_t_var_ctx)

        loss_dict = {
            "loss": (nll_loss + kl_loss).mean(),
            "nll": nll_loss.mean(),
            "kl": kl_loss.mean(),
        }

        return loss_dict
