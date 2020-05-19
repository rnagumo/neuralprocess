
"""Gaussian Process.

Mainly used for data generation and model comparison.
"""

from typing import Optional

import torch
from torch import Tensor


class GaussianProcess(torch.nn.Module):
    """Gaussian Process class.

    Args:
        y_dim (int, optional): y dimension.
        l1_scale (float, optional): Scale parameter of the Gaussian kernel.
        sigma (float, optional): Magnitude of std.
    """

    def __init__(self, y_dim: int = 1, l1_scale: float = 0.4,
                 sigma: float = 1.0):
        super().__init__()

        self.y_dim = y_dim
        self.l1_scale = l1_scale
        self.sigma = sigma

    def gaussian_kernel(self, x0: Tensor, x1: Optional[Tensor] = None,
                        eps: float = 1e-2) -> Tensor:
        """Gaussian kernel.

        Args:
            x0 (torch.Tensor): First input data of size
                `(batch_size, num_points_0, x_dim)`.
            x1 (torch.Tensor, optional): Second input data of size
                `(batch_size, num_points_1, x_dim)`.
            eps (float, optional): Noise scale.

        Returns:
            kernel (torch.Tensor): kernel matrix of size
                `(batch_size, y_dim, num_points_0, num_points_1)` if x1 is not
                `None`, `(batch_size, y_dim, num_points_0, num_points_0)`
                otherwise.
        """

        # Data size
        batch_size, num_points, x_dim = x0.size()

        # Kernel parameters
        l1 = torch.ones(batch_size, self.y_dim, x_dim) * self.l1_scale
        sigma = torch.ones(batch_size, self.y_dim) * self.sigma

        # Expand and take diff (batch_size, num_points, num_points, x_dim)
        if x1 is None:
            x1_unsq = x0.unsqueeze(1)  # (batch_size, 1, num_points_0, x_dim)
        else:
            x1_unsq = x1.unsqueeze(1)  # (batch_size, 1, num_points_1, x_dim)

        x0_unsq = x0.unsqueeze(2)  # (batch_size, num_points_0, 1, x_dim)
        diff = x1_unsq - x0_unsq

        # (batch_size, y_dim, num_points, num_points)
        norm = ((diff[:, None] / l1[:, :, None, None]) ** 2).sum(-1)

        # (batch_size, y_dim, num_points, num_points)
        kernel = (sigma ** 2)[:, :, None, None] * torch.exp(-0.5 * norm)

        # Add some noise to the diagonal to make the cholesky work
        if x1 is None:
            kernel += (eps ** 2) * torch.eye(num_points, device=x0.device)

        return kernel

    def inference(self, x: Tensor) -> Tensor:
        """Inference p(y|x).

        Args:
            x (torch.Tensor): Input tensor of size
                `(batch_size. num_points, x_dim)`.

        Returns:
            y (torch.Tensor): Sampled y `(batch_size, num_points, y_dim)`.
        """

        batch_size, num_points, _ = x.size()

        # Gaussian kernel (batch_size, y_dim, num_points, num_points)
        kernel = self.gaussian_kernel(x)

        # Calculate cholesky using double precision
        cholesky = torch.cholesky(kernel.double()).float()

        # Sample curve (batch_size, y_size, num_points, 1)
        y = cholesky.matmul(
            torch.randn(batch_size, self.y_dim, num_points, 1))

        # (batch_size, num_points, y_size)
        y = y.squeeze(3).permute(0, 2, 1)

        return y

    def fit(self, x: Tensor, y: Tensor):
        """Fit Gaussian Process to the given training data.

        This method only saves given data.

        Args:
            x (torch.Tensor): Input data for training, size
                `(num_points, x_dim)`.
            y (torch.Tensor): Output data for training, size
                `(num_points, y_dim)`.
        """

        self._x_train = x
        self._y_train = y

    def predict(self, x: Tensor) -> Tensor:
        """Predict y for given x.

        Args:
            x (torch.Tensor): Input data for test, size `(num_points, x_dim)`.

        Returns:
            y (torch.Tensor): Predicted output, size `(num_points, y_dim)`.
        """

        # Shift mean of output
        y_mean = self._y_train.mean()
