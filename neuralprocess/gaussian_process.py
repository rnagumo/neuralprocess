
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

        # Saved training data
        self._x_train = None
        self._y_train = None

    def gaussian_kernel(self, x0: Tensor, x1: Tensor, eps: float = 1e-2
                        ) -> Tensor:
        """Gaussian kernel.

        Args:
            x0 (torch.Tensor): First input data of size
                `(batch_size, num_points_0, x_dim)`.
            x1 (torch.Tensor): Second input data of size
                `(batch_size, num_points_1, x_dim)`.
            eps (float, optional): Noise scale.

        Returns:
            kernel (torch.Tensor): kernel matrix of size
                `(batch_size, num_points_0, num_points_1)`.
        """

        # Data size
        batch_size, num_points, x_dim = x0.size()

        # Kernel parameters
        l1 = torch.ones(batch_size, x_dim) * self.l1_scale
        sigma = torch.ones(batch_size) * self.sigma

        # Expand and take diff (batch_size, num_points_0, num_points_1, x_dim)
        x0_unsq = x0.unsqueeze(2)  # (batch_size, num_points_0, 1, x_dim)
        x1_unsq = x1.unsqueeze(1)  # (batch_size, 1, num_points_1, x_dim)
        diff = x1_unsq - x0_unsq

        # (batch_size, num_points_0, num_points_1)
        norm = ((diff / l1[:, None, None, :]) ** 2).sum(-1)

        # (batch_size, num_points_0, num_points_1)
        kernel = (sigma ** 2)[:, None, None] * torch.exp(-0.5 * norm)

        # Add some noise to the diagonal to make the cholesky work
        if kernel.size(1) == kernel.size(2):
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

        # Gaussian kernel (batch_size, num_points, num_points)
        kernel = self.gaussian_kernel(x, x)

        # Calculate cholesky using double precision
        cholesky = torch.cholesky(kernel.double()).float()

        # Sample curve (batch_size, num_points, y_size)
        y = cholesky.matmul(torch.randn(batch_size, num_points, self.y_dim))

        return y

    def fit(self, x: Tensor, y: Tensor) -> None:
        """Fit Gaussian Process to the given training data.

        This method only saves given data.

        Args:
            x (torch.Tensor): Input data for training, size
                `(batch_size, num_points, x_dim)`.
            y (torch.Tensor): Output data for training, size
                `(batch_size, num_points, y_dim)`.
        """

        self._x_train = x
        self._y_train = y

    def predict(self, x: Tensor, return_cov: bool = False) -> Tensor:
        """Predict y for given x with previously seen training data.

        Args:
            x (torch.Tensor): Input data for test, size
                `(batch_size, num_points, x_dim)`.
            return_cov (bool, optional): If true, covariance of the joint
                predictive distribution at the query points is returned.

        Returns:
            y (torch.Tensor): Predicted output, size
                `(batch_size, num_points, y_dim)`.
        """

        if self._x_train is None or self._y_train is None:
            return self.inference(x)

        # Shift mean of y train to 0
        y_mean = self._y_train.mean(dim=[0, 1])
        y_train = self._y_train - y_mean

        # Kernel
        K_nn = self.gaussian_kernel(self._x_train, self._x_train)
        K_xx = self.gaussian_kernel(x, x)
        K_xn = self.gaussian_kernel(x, self._x_train)

        # Solve cholesky for each y_dim
        L_ = torch.cholesky(K_nn.double()).float()
        alpha_ = torch.cholesky_solve(y_train, L_)

        # Mean prediction with undoing normalization
        y_mean = K_xn.matmul(alpha_) + y_mean

        if not return_cov:
            return y_mean

        # Cov
        v = torch.cholesky_solve(K_xn.transpose(1, 2), L_)
        y_cov = K_xx - K_xn.matmul(v)

        return y_mean, y_cov
