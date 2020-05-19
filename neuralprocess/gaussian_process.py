
"""Gaussian Process.

Mainly used for data generation and model comparison.
"""


import torch
from torch import Tensor


class GaussianProcess(torch.nn.Module):
    """Gaussian Process class.

    Args:
        y_dim (int, optional): y dimension.
        l1_scale (float, optional): Scale parameter of the Gaussian kernel.
        sigma (float, optional): Magnitude of std.
    """

    def __init__(self, y_dim: int = 1, l1_scale: float = 0.1,
                 sigma: float = 1.0):
        super().__init__()

        self.y_dim = y_dim
        self.l1_scale = l1_scale
        self.sigma = sigma

    def gaussian_kernel(self, x: Tensor, eps: float = 1e-2) -> Tensor:
        """Gaussian kernel.

        Args:
            x (torch.Tensor): Input data of size
                `(batch_size, num_points, x_dim)`.
            eps (float, optional): Noise scale.

        Returns:
            kernel (torch.Tensor): kernel matrix of size
                `(batch_size, y_dim, num_points, num_points)`.
        """

        # Data size
        batch_size, num_points, x_dim = x.size()

        # Kernel parameters
        l1 = torch.ones(batch_size, self.y_dim, x_dim) * self.l1_scale
        sigma = torch.ones(batch_size, self.y_dim) * self.sigma

        # Expand and take diff
        x_1 = x.unsqueeze(1)  # (batch_size, 1, num_points, x_dim)
        x_2 = x.unsqueeze(2)  # (batch_size, num_points, 1, x_dim)
        diff = x_1 - x_2

        # (batch_size, y_dim, num_points, num_points)
        norm = ((diff[:, None] / l1[:, :, None, None]) ** 2).sum(-1)

        # (batch_size, y_dim, num_points, num_points)
        kernel = (sigma ** 2)[:, :, None, None] * torch.exp(-0.5 * norm)

        # Add some noise to the diagonal to make the cholesky work
        kernel += (eps ** 2) * torch.eye(num_points, device=x.device)

        return kernel

    def inference(self, x: Tensor) -> Tensor:
        """Inference p(y|x).

        Args:
            x (torch.Tensor): Input tensor of size
                `(batch_size. num_points, x_dim)`.

        Returns:
            y (torch.Tensor): Sampled y `(batch_size, num_points, y_size)`.
        """

        batch_size, num_points, x_dim = x.size()

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
