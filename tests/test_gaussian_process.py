
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
            kernel.size(), (batch_size, num_points_0, num_points_0))

        num_points_1 = 4
        x1 = torch.randn(batch_size, num_points_1, x_dim)
        kernel = self.model.gaussian_kernel(x0, x1)
        self.assertTupleEqual(
            kernel.size(), (batch_size, num_points_0, num_points_1))

    def test_inference(self):
        batch_size = 5
        num_points = 10
        x_dim = 3
        x = torch.randn(batch_size, num_points, x_dim)
        y = self.model.inference(x)

        self.assertTupleEqual(y.size(), (batch_size, num_points, self.y_dim))

    def test_fit(self):
        batch_size = 5
        num_points = 10
        x_dim = 3
        x = torch.randn(batch_size, num_points, x_dim)
        y = torch.randn(batch_size, num_points, 1)

        self.model.fit(x, y)
        self.assertTrue((self.model._x_train == x).all().item)
        self.assertTrue((self.model._y_train == y).all().item)

    def test_predict(self):
        batch_size = 5
        num_points = 10
        x_dim = 3
        y_dim = 2
        x = torch.randn(batch_size, num_points, x_dim)
        y = torch.randn(batch_size, num_points, y_dim)

        self.model.fit(x, y)
        y_mean = self.model.predict(x)
        self.assertTupleEqual(y_mean.size(), (batch_size, num_points, y_dim))

        # Return cov
        y_mean, y_cov = self.model.predict(x, return_cov=True)
        self.assertTupleEqual(y_mean.size(), (batch_size, num_points, y_dim))
        self.assertTupleEqual(
            y_cov.size(), (batch_size, num_points, num_points))
