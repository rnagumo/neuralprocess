
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
        num_context_max (int): Upper bound of number of context data.
        num_target_max (int): Upper bound of number of target data.
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

    def __init__(self, train: bool, batch_size: int, num_context_max: int,
                 num_target_max: int, x_dim: int = 1, y_dim: int = 1,
                 gp_params: dict = {}):
        super().__init__()

        # Args
        self.train = train
        self.batch_size = batch_size
        self.num_context_max = num_context_max
        self.num_target_max = num_target_max
        self.x_dim = x_dim
        self.y_dim = y_dim

        # Attributes
        self.gp = GaussianProcess(**gp_params)
        self.x_context = None
        self.y_context = None
        self.x_target = None
        self.y_target = None

        # Initialize dataset
        self.generate_dataset()

    def generate_dataset(self, x_ub: float = 2.0, x_lb: float = -2.0) -> None:
        """Initializes dataset.

        Dataset sizes.

        * x_context: `(batch_size, num_context, x_dim)`
        * y_context: `(batch_size, num_context, y_dim)`
        * x_target: `(batch_size, num_target, x_dim)`
        * y_target: `(batch_size, num_target, y_dim)`

        **Note**

        `num_context` and `num_target` are sampled from uniform distributions.
        Therefore, these two values might be changed at each time this function
        is called.

        Args:
            x_ub (float, optional): Upper bound of x range.
            x_lb (float, optional): Lower bound of x range.
        """

        # Sample number of data points
        num_context = torch.randint(3, self.num_context_max, (1,)).item()
        num_target = (torch.randint(2, self.num_target_max, (1,)).item()
                      if self.train else self.num_target_max)

        # Sample input x
        if self.train:
            # For training, sample random points
            x = torch.rand(self.batch_size, num_context + num_target,
                           self.x_dim)
            x = x * (x_ub - x_lb) + x_lb
        else:
            # For test, sample evenly distributed array for plot
            num_points = max(num_context, num_target)

            # Uniformly distributed x
            x = torch.arange(x_lb, x_ub, (x_ub - x_lb) / num_points)

            # Expand x_dim, size (num_points, x_dim)
            x = x.view(-1, 1).repeat(1, self.x_dim)

            # Expand batch, size (batch_size, num_points, x_dim)
            x = x.repeat(self.batch_size, 1, 1)

        # Sample y from GP prior
        y = self.gp.sample(x, y_dim=self.y_dim)

        # Split sampled data into context and target set
        if self.train:
            # Context dataset
            self.x_context = x[:, :num_context]
            self.y_context = y[:, :num_context]

            # Target dataset
            self.x_target = x[:, num_context:]
            self.y_target = y[:, num_context:]
        else:
            # For context dataset, sample random data points from uniformly
            # distributed x and y
            _x_context = torch.empty(
                self.batch_size, num_context, self.x_dim)
            _y_context = torch.empty(
                self.batch_size, num_context, self.y_dim)
            for i in range(self.batch_size):
                indices = torch.randperm(max(num_context, num_target))
                _x_context[i] = x[i, indices[:num_context]]
                _y_context[i] = y[i, indices[:num_context]]

            self.x_context = _x_context
            self.y_context = _y_context

            # Target dataset
            self.x_target = x[:, :num_target]
            self.y_target = y[:, :num_target]

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

    @property
    def num_context(self):
        return self.x_context.size(1)

    @property
    def num_target(self):
        return self.x_target.size(1)
