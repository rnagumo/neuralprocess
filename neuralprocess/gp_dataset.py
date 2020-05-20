
"""Dataset with Gaussian Process."""

from typing import Tuple

import torch
from torch import Tensor

from .gaussian_process import GaussianProcess


class GPDataset(torch.utils.data.Dataset):
    """Gaussian Process dataset class.

    Args:
        train (bool): Boolean for specifying train or test.
        batch_size (int): Number of total batch.
        num_context (int): Number of context data.
        num_target (int): Number of target data.
        x_dim (int, optional): Dimension size of input x.
        y_dim (int, optional): Dimension size of output y.
        gp_params (dict, optional): Parameters dict for GP class.

    Attributes:
        gp (neuralprocess.GaussianProcess): GP object.
        x_context (torch.Tensor): x data for context.
        y_context (torch.Tensor): y data for context.
        x_target (torch.Tensor): x data for target.
        y_target (torch.Tensor): y data for target.
    """

    def __init__(self, train: bool, batch_size: int, num_context: int,
                 num_target: int, x_dim: int = 1, y_dim: int = 1,
                 gp_params: dict = {}):
        super().__init__()

        # Args
        self.train = train
        self.batch_size = batch_size
        self.num_context = num_context
        self.num_target = num_target
        self.x_dim = x_dim
        self.y_dim = y_dim

        # Attributes
        self.gp = GaussianProcess(**gp_params)
        self.x_context = None
        self.y_context = None
        self.x_target = None
        self.y_target = None

        # Init dataset
        self._init_dataset()

    def _init_dataset(self, x_ub: float = 2.0, x_lb: float = -2.0) -> None:
        """Initializes dataset.

        Dataset sizes.

        * x_context: `(batch_size, num_context, x_dim)`
        * y_context: `(batch_size, num_context, y_dim)`
        * x_target: `(batch_size, num_target, x_dim)`
        * y_target: `(batch_size, num_target, y_dim)`

        Args:
            x_ub (float, optional): Upper bound of x range.
            x_lb (float, optional): Lower bound of x range.
        """

        if self.num_context > self.num_target:
            raise ValueError(
                f"Dataset size error: "
                f"num_context ({self.num_context}) should be equal or "
                f"larger than num_target ({self.num_target})")

        # Sample input x
        if self.train:
            # For training, sample random points
            x = (torch.rand(self.batch_size, self.num_target, self.x_dim)
                 * (x_ub - x_lb) + x_lb)
        else:
            # For test, sample evenly distributed array for plot

            # Uniformly distributed x
            x = torch.arange(x_lb, x_ub, (x_ub - x_lb) / self.num_target)

            # Expand x_dim, size (num_target, x_dim)
            x = x.view(-1, 1).repeat(1, self.x_dim)

            # Expand batch, size (batch_size, num_target, x_dim)
            x = x.repeat(self.batch_size, 1, 1)

        # Sample y from GP prior
        y = self.gp.sample(x, y_dim=self.y_dim)

        # Context dataset
        if self.train:
            self.x_context = x[:, :self.num_context]
            self.y_context = y[:, :self.num_context]
        else:
            # Random sample from curves
            _x_context = torch.empty(
                self.batch_size, self.num_context, self.x_dim)
            _y_context = torch.empty(
                self.batch_size, self.num_context, self.y_dim)
            for i in range(self.batch_size):
                indices = torch.randperm(self.num_target)
                _x_context[i] = x[i, indices[:self.num_context]]
                _y_context[i] = y[i, indices[:self.num_context]]

            self.x_context = _x_context
            self.y_context = _y_context

        # Target dataset contains all sample data.
        self.x_target = x
        self.y_target = y

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """Gets item with specified index.

        Args:
            index (int): Index number.

        Returns:
            x_context (torch.Tensor): x data for context.
            y_context (torch.Tensor): y data for context.
            x_target (torch.Tensor): x data for target.
            y_target (torch.Tensor): y data for target.
        """

        return (self.x_context[index], self.y_context[index],
                self.x_target[index], self.y_target[index])

    def __len__(self) -> int:
        """Length of dataset.

        Returns:
            batch_size (int): Batch size of this dataset.
        """

        return self.batch_size
