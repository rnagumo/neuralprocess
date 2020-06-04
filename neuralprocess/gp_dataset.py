
"""Dataset with Gaussian Process."""

from typing import Tuple

import random

import torch
from torch import Tensor

from .gaussian_process import GaussianProcess


class GPDataset(torch.utils.data.Dataset):
    """Gaussian Process dataset class.

    Args:
        train (bool): Boolean for specifying train or test.
        batch_size (int): Number of total batch.
        num_context_min (int, optional): Lower bound of number of context data.
        num_context_max (int, optional): Upper bound of number of context data.
        num_target_min (int, optional): Lower bound of number of target data.
        num_target_max (int, optional): Upper bound of number of target data.
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

    def __init__(self, train: bool, batch_size: int, num_context_min: int = 3,
                 num_context_max: int = 10, num_target_min: int = 2,
                 num_target_max: int = 10, x_dim: int = 1, y_dim: int = 1,
                 gp_params: dict = {}):
        super().__init__()

        # Args
        self.train = train
        self.batch_size = batch_size
        self.num_context_min = num_context_min
        self.num_context_max = num_context_max
        self.num_target_min = num_target_min
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

    def generate_dataset(self, x_ub: float = 2.0, x_lb: float = -2.0,
                         resample_params: bool = False) -> None:
        """Initializes dataset.

        Dataset sizes.

        * x_context: `(batch_size, num_context, x_dim)`
        * y_context: `(batch_size, num_context, y_dim)`
        * x_target: `(batch_size, num_target, x_dim)`
        * y_target: `(batch_size, num_target, y_dim)`

        **Note**

        * `num_context` and `num_target` are sampled from uniform
        distributions. Therefore, these two values might be changed at each
        time this function is called.

        * Target dataset includes context dataset, that is, context is a
        subset of target.

        Args:
            x_ub (float, optional): Upper bound of x range.
            x_lb (float, optional): Lower bound of x range.
            resample_params (bool, optional): If true, resample gaussian
                kernel parameters.
        """

        # Bounds number of dataset
        num_context_max = max(self.num_context_min, self.num_context_max)
        num_target_max = max(self.num_target_min, self.num_target_max)

        # Sample number of data points
        num_context = random.randint(self.num_context_min, num_context_max)
        num_target = (
            random.randint(self.num_target_min, num_target_max)
            if self.train else max(self.num_target_min, self.num_target_max))

        # Sample input x for target
        if self.train:
            # For training, sample random points in range of [x_lb, x_ub]
            x = torch.rand(self.batch_size, num_target, self.x_dim)
            x = x * (x_ub - x_lb) + x_lb
        else:
            # For test, sample uniformly distributed array in range of
            # [x_lb, x_ub] for target dataset

            # Uniformly distributed x
            x = torch.arange(x_lb, x_ub, (x_ub - x_lb) / num_target)

            # Expand x size (batch_size, num_points, x_dim)
            x = x.view(1, -1, 1).repeat(self.batch_size, 1, self.x_dim)

        # Sample y from GP prior
        y = self.gp.sample(
            x, y_dim=self.y_dim, resample_params=resample_params)

        # Sample random data points as context from target
        _x_context = torch.empty(
            self.batch_size, num_context, self.x_dim)
        _y_context = torch.empty(
            self.batch_size, num_context, self.y_dim)
        for i in range(self.batch_size):
            indices = torch.randint(0, num_target, (num_context,))
            _x_context[i] = x[i, indices]
            _y_context[i] = y[i, indices]

        self.x_context = _x_context
        self.y_context = _y_context

        # Target dataset
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

    @property
    def num_context(self) -> int:
        """Number of context for current dataset."""
        return self.x_context.size(1)

    @property
    def num_target(self) -> int:
        """Number of target for current dataset."""
        return self.x_target.size(1)
