
"""Base Neural Process class."""

from typing import Tuple, Dict

import math

from torch import nn, Tensor


class BaseNP(nn.Module):
    """Base Neural Process class."""

    def forward(self, x_context: Tensor, y_context: Tensor, x_target: Tensor
                ) -> Tensor:
        """Forward method.

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
        """

        mu, _ = self.query(x_context, y_context, x_target)
        return mu

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

        raise NotImplementedError

    def loss_func(self, x_context: Tensor, y_context: Tensor, x_target: Tensor,
                  y_target: Tensor) -> Dict[str, Tensor]:
        """Loss function for the negative conditional log probability.

        **Note**

        Returned `loss_dict` must include `loss` key-value.

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

        raise NotImplementedError


def kl_divergence_normal(mu0: Tensor, var0: Tensor, mu1: Tensor, var1: Tensor
                         ) -> Tensor:
    """Kullback Leibler divergence for Normal distributions with diagonal
    covariance matrix.

    p = N(mu0, var0)
    q = N(mu1, var1)
    KL(p||q) = 1/2 * (Tr(var1^{-1} var0) + (mu1-mu0)^T var1^{-1} (mu1-mu0)
                      - d + log(|var1| / |var0|))

    Args:
        mu0 (torch.Tensor): Mean vector of p, size.
        var0 (torch.Tensor): Diagonal variance of p.
        mu1 (torch.Tensor): Mean vector of q.
        var1 (torch.Tensor): Diagonal variance of q.

    Returns:
        kl (torch.Tensor): Calculated kl divergence for each data.
    """

    diff = mu1 - mu0
    kl = ((var0 / var1).sum(-1) + (diff / var1 * diff).sum(-1) - mu1.size(-1)
          + (var1.prod(-1) / var0.prod(-1)).log()) * 0.5

    return kl


def nll_normal(x: Tensor, mu: Tensor, var: Tensor, reduce: bool = True
               ) -> Tensor:
    """Negative log likelihood for 1-D Normal distribution.

    Args:
        mu (torch.Tensor): Mean vector.
        var (torch.Tensor): Variance vector.
        reduce (bool, optional): If `True`, sum calculated loss for each
            data point.

    Returns:
        nll (torch.Tensor): Calculated nll for each data.
    """

    nll = 0.5 * ((2 * math.pi * var).log() + (x - mu) ** 2 / var)

    if reduce:
        return nll.sum(-1)
    return nll
