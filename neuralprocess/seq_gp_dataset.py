
"""Dataset for sequential data with Gaussian Process."""

from typing import Tuple

import random

import torch
from torch import Tensor

from .gaussian_process import GaussianProcess


class SequentialGPDataset(torch.utils.data.Dataset):
    """Sequential dataset class with Gaussian Process.

    Args:
        train (bool): Boolean for specifying train or test.
        batch_size (int): Number of total batch.
        seq_len (int): Length of sequence.
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

    def __init__(self, train: bool, batch_size: int, seq_len: int,
                 num_context_min: int = 3, num_context_max: int = 10,
                 num_target_min: int = 2, num_target_max: int = 10,
                 x_dim: int = 1, y_dim: int = 1, gp_params: dict = {}):
        super().__init__()

        # Args
        self.train = train
        self.batch_size = batch_size
        self.seq_len = seq_len
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
                         resample_params: bool = False,
                         single_params: bool = True) -> None:
        """Initializes dataset.

        Dataset sizes.

        * x_context: `(batch_size, seq_len, num_context, x_dim)`
        * y_context: `(batch_size, seq_len, num_context, y_dim)`
        * x_target: `(batch_size, seq_len, num_target, x_dim)`
        * y_target: `(batch_size, seq_len, num_target, y_dim)`

        **Note**

        * `num_context` and `num_target` are sampled from uniform
        distributions. Therefore, these two values might be changed at each
        time this function is called.

        * Target dataset includes context dataset, that is, context is a
        subset of target.

        Args:
            x_ub (float, optional): Upper bound of x range.
            x_lb (float, optional): Lower bound of x range.
            resample_params (bool, optional): If `True`, resample gaussian
                kernel parameters.
            single_params (bool, optional): If `True`, resampled kernel
                parameters are single values (default = `True`).
        """

        # Bounds number of dataset
        num_context_max = max(self.num_context_min, self.num_context_max)
        num_target_max = max(self.num_target_min, self.num_target_max)

        # Sample number of data points
        num_context = random.randint(self.num_context_min, num_context_max)
        num_target = (
            random.randint(self.num_target_min, num_target_max)
            if self.train else max(self.num_target_min, self.num_target_max))

        # Sequential hyper-parameter setting
        l2_scale = torch.empty(self.batch_size).uniform_(0.7, 1.2)
        delta_l2_scale = torch.empty(self.batch_size).uniform_(-0.03, 0.03)

        variance = torch.empty(self.batch_size).uniform_(1.0, 2.56)
        delta_variance = torch.empty(self.batch_size).uniform_(-0.0025, 0.0025)

        # Sample data
        x_context = []
        y_context = []
        x_target = []
        y_target = []

        for _ in range(self.seq_len):
            # Update gp params
            self.gp.l2_scale = l2_scale
            self.gp.variance = variance

            # Sample single time step data
            x_c, y_c, x_t, y_t = self._generate_single_step(
                num_context, num_target, x_ub, x_lb)

            x_context.append(x_c)
            y_context.append(y_c)
            x_target.append(x_t)
            y_target.append(y_t)

            # Update sequential parameters
            l2_scale = (l2_scale + delta_l2_scale
                        + torch.randn(self.batch_size) * 1e-5)
            variance = (variance + delta_variance
                        + torch.randn(self.batch_size) * 1e-5)

        # Reshape: (l, b, n, d) -> (b, l, n, d)
        self.x_context = torch.stack(x_context).permute(1, 0, 2, 3)
        self.y_context = torch.stack(y_context).permute(1, 0, 2, 3)
        self.x_target = torch.stack(x_target).permute(1, 0, 2, 3)
        self.y_target = torch.stack(y_target).permute(1, 0, 2, 3)

    def _generate_single_step(self, num_context: int, num_target: int,
                              x_ub: float, x_lb: float) -> Tuple[Tensor]:
        """Generates single time step data.

        Args:
            num_context (int): Number of context.
            num_target (int): Number of target.
            x_ub (float): Upper bound of x range.
            x_lb (float): Lower bound of x range.

        Returns:
            x_context (torch.Tensor): x data for context.
            y_context (torch.Tensor): y data for context.
            x_target (torch.Tensor): x data for target.
            y_target (torch.Tensor): y data for target.
        """

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
        y = self.gp.sample(x, y_dim=self.y_dim)

        # Sample random data points as context from target
        _x_context = torch.empty(
            self.batch_size, num_context, self.x_dim)
        _y_context = torch.empty(
            self.batch_size, num_context, self.y_dim)
        for i in range(self.batch_size):
            indices = torch.randint(0, num_target, (num_context,))
            _x_context[i] = x[i, indices]
            _y_context[i] = y[i, indices]

        return _x_context, _y_context, x, y

    def __getitem__(self, index: int) -> Tuple[Tensor]:
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
        return self.x_context.size(2)

    @property
    def num_target(self) -> int:
        """Number of target for current dataset."""
        return self.x_target.size(2)
