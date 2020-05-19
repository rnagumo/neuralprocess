
import unittest

import torch

import neuralprocess as npr


class TestGaussianProcess(unittest.TestCase):

    def setUp(self):
        self.model = npr.GaussianProcess()

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

    def test_gaussian_kernel_with_raises(self):
        batch_size = 5
        num_points_0 = 10
        num_points_1 = 4
        x_dim = 3
        x0 = torch.randn(batch_size, num_points_0, x_dim)

        with self.assertRaises(ValueError):
            x1 = torch.randn(batch_size + 5, num_points_1, x_dim)
            self.model.gaussian_kernel(x0, x1)

        with self.assertRaises(ValueError):
            x1 = torch.randn(batch_size, num_points_1, x_dim + 5)
            self.model.gaussian_kernel(x0, x1)

    def test_inference(self):
        batch_size = 5
        num_points = 10
        x_dim = 3
        x = torch.randn(batch_size, num_points, x_dim)

        y_dim = 2
        y = self.model.inference(x, y_dim)
        self.assertTupleEqual(y.size(), (batch_size, num_points, y_dim))

        with self.assertRaises(ValueError):
            x = torch.randn(batch_size, x_dim)
            self.model.inference(x)

    def test_fit(self):
        batch_size = 5
        num_points = 10
        x_dim = 3
        x = torch.randn(batch_size, num_points, x_dim)
        y = torch.randn(batch_size, num_points, 1)

        self.model.fit(x, y)
        self.assertTrue((self.model._x_train == x).all().item)
        self.assertTrue((self.model._y_train == y).all().item)

        with self.assertRaises(ValueError):
            x = torch.randn(batch_size, num_points, x_dim)
            y = torch.randn(batch_size, num_points)
            self.model.fit(x, y)

        with self.assertRaises(ValueError):
            x = torch.randn(batch_size, num_points)
            y = torch.randn(batch_size, num_points, 1)
            self.model.fit(x, y)

    def test_predict(self):
        batch_size = 5
        num_points = 10
        x_dim = 3
        y_dim = 2
        x = torch.randn(batch_size, num_points, x_dim)
        y = torch.randn(batch_size, num_points, y_dim)

        self.model.fit(x, y)
        y_mean, y_cov = self.model.predict(x)
        self.assertTupleEqual(y_mean.size(), (batch_size, num_points, y_dim))
        self.assertTupleEqual(
            y_cov.size(), (batch_size, num_points, num_points))

        with self.assertRaises(ValueError):
            x = torch.randn(batch_size, x_dim)
            self.model.predict(x)

    def test_predict_with_raise(self):
        batch_size = 5
        num_points = 10
        x_dim = 3
        x = torch.randn(batch_size, num_points, x_dim)
        with self.assertRaises(ValueError):
            self.model.predict(x)

    def test_forward(self):
        batch_size = 5
        num_points = 10
        x_dim = 3
        y_dim = 2
        x = torch.randn(batch_size, num_points, x_dim)
        y = torch.randn(batch_size, num_points, y_dim)

        self.model.fit(x, y)
        y_mean = self.model(x)
        self.assertTupleEqual(y_mean.size(), (batch_size, num_points, y_dim))
