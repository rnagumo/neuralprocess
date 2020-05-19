
import unittest

import torch

import neuralprocess as npr


class TestGaussianProcess(unittest.TestCase):

    def setUp(self):
        self.y_dim = 2
        self.model = npr.GaussianProcess(y_dim=self.y_dim)

    def test_gaussian_kernel(self):
        batch_size = 5
        num_points_0 = 10
        x_dim = 3
        x0 = torch.randn(batch_size, num_points_0, x_dim)

        kernel = self.model.gaussian_kernel(x0, x0)
        self.assertTupleEqual(
            kernel.size(),
            (batch_size, self.y_dim, num_points_0, num_points_0))

        num_points_1 = 4
        x1 = torch.randn(batch_size, num_points_1, x_dim)
        kernel = self.model.gaussian_kernel(x0, x1)
        self.assertTupleEqual(
            kernel.size(),
            (batch_size, self.y_dim, num_points_0, num_points_1))

    def test_inference(self):
        batch_size = 5
        num_points = 10
        x_dim = 3
        x = torch.randn(batch_size, num_points, x_dim)
        y = self.model.inference(x)

        self.assertTupleEqual(y.size(), (batch_size, num_points, self.y_dim))
