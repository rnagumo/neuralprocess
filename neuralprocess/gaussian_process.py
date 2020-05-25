
"""Gaussian Process.

Mainly used for data generation and model comparison.
"""

from typing import Tuple

import torch
from torch import Tensor


class GaussianProcess(torch.nn.Module):
    """Gaussian Process class.

    Args:
        l2_scale (float, optional): Scale parameter of the Gaussian kernel.
        variance (float, optional): Magnitude of std.
    """

    def __init__(self, l2_scale: float = 0.4, variance: float = 1.0):
        super().__init__()

        self.l2_scale = l2_scale
        self.variance = variance

        # Saved training data
        self._x_train = None
        self._y_train = None

        # Check parameters
        if l2_scale <= 0:
            raise ValueError("L2 scale parameter should be >0")

        if variance <= 0:
            raise ValueError("Variance should be >0")

    def forward(self, x: Tensor) -> Tensor:
        """Forward method for prediction.

        Args:
            x (torch.Tensor): Input tensor

        Returns:
            y_mean (torch.Tensor): Predicted output, size
                `(batch_size, num_points, y_dim)`.
        """

        y_mean, _ = self.predict(x)
        return y_mean

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

        if x0.size(0) != x1.size(0):
            raise ValueError("Batch size of x0 and x1 should be same: "
                             f"x0 size = {x0.size()}, x1 size = {x1.size()}")

        if x0.size(2) != x1.size(2):
            raise ValueError("Dimension size of x0 and x1 should be same: "
                             f"x0 size = {x0.size()}, x1 size = {x1.size()}")

        # Data size
        batch_size, num_points, x_dim = x0.size()

        # Expand and take diff (batch_size, num_points_0, num_points_1, x_dim)
        x0_unsq = x0.unsqueeze(2)  # (batch_size, num_points_0, 1, x_dim)
        x1_unsq = x1.unsqueeze(1)  # (batch_size, 1, num_points_1, x_dim)
        diff = x1_unsq - x0_unsq

        # Norm (batch_size, num_points_0, num_points_1)
        norm = ((diff / self.l2_scale) ** 2).sum(-1)

        # Gaussian kernel (batch_size, num_points_0, num_points_1)
        kernel = self.variance * torch.exp(-0.5 * norm)

        # Add some noise to the diagonal to make the cholesky work
        if kernel.size(1) == kernel.size(2):
            kernel += (eps ** 2) * torch.eye(num_points, device=x0.device)

        return kernel

    def fit(self, x: Tensor, y: Tensor) -> None:
        """Fits Gaussian Process to the given training data.

        This method only saves given data.

        Args:
            x (torch.Tensor): Input data for training, size
                `(batch_size, num_points, x_dim)`.
            y (torch.Tensor): Output data for training, size
                `(batch_size, num_points, y_dim)`.
        """

        if x.dim() != 3:
            raise ValueError("Dim of x should be 3 (batch_size, num_points, "
                             f"x_dim), but given {x.size()}.")

        if y.dim() != 3:
            raise ValueError("Dim of y should be 3 (batch_size, num_points, "
                             f"y_dim), but given {y.size()}.")

        self._x_train = x
        self._y_train = y

    def predict(self, x: Tensor, y_dim: int = 1) -> Tuple[Tensor, Tensor]:
        """Predicts mean and covariance.

        Args:
            x (torch.Tensor): Input data for test, size
                `(batch_size, num_points, x_dim)`.
            y_dim (int, optional): Output y dim size for prior.

        Returns:
            y_mean (torch.Tensor): Predicted output, size
                `(batch_size, num_points, y_dim)`.
            y_cov (torch.Tensor): Covariance of the joint predictive
                distribution at the query points, size
                `(batch_size, num_points, num_points)`.
        """

        if x.dim() != 3:
            raise ValueError("Dim of x should be 3 (batch_size, num_points, "
                             f"x_dim), but given {x.size()}.")

        # Predict y|x based on GP prior
        if self._x_train is None or self._y_train is None:
            batch_size, num_points, _ = x.size()
            y_mean = torch.zeros(batch_size, num_points, y_dim)
            y_cov = self.gaussian_kernel(x, x)
            return y_mean, y_cov

        # Predict y*|x*, x, y based on GP posterior

        # Shift mean of y_train to 0
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

        # Cov
        v = torch.cholesky_solve(K_xn.transpose(1, 2), L_)
        y_cov = K_xx - K_xn.matmul(v)

        return y_mean, y_cov

    def sample(self, x: Tensor, y_dim: int = 1) -> Tensor:
        """Samples function from GP.

        Args:
            x (torch.Tensor): Input tensor of size
                `(batch_size. num_points, x_dim)`.
            y_dim (int, optional): Output y dim size.

        Returns:
            y (torch.Tensor): Sampled y `(batch_size, num_points, y_dim)`.
        """

        # Sample mean and cov
        mean, cov = self.predict(x, y_dim=y_dim)

        # Dims
        batch_size, num_points, _ = x.size()

        # Calculate cholesky using double precision
        chol = torch.cholesky(cov.double()).float()

        # Sample y (batch_size, num_points, y_dim)
        y = chol.matmul(torch.randn(batch_size, num_points, y_dim)) + mean

        return y

    def resample_params(self, l2_scale: float = 1.0, variance: float = 1.0,
                        eps: float = 1e-2) -> None:
        """Resamples `l2_scale` and `variance` params.

        Args:
            l2_scale (float): Upper bounds of `l2_scale` value.
            variance (float): Upper bounds of `variance` value.
            eps (float): Lower bounds of sampled values.
        """

        self.l2_scale = torch.rand((1,)).item() * l2_scale + eps
        self.variance = torch.rand((1,)).item() * variance + eps
