
import unittest

import torch

import neuralprocess as npr


class TestGaussianProcess(unittest.TestCase):

    def setUp(self):
        self.model = npr.GaussianProcess()

    def test_init(self):

        with self.assertRaises(ValueError):
            npr.GaussianProcess(l2_scale=0.)

        with self.assertRaises(ValueError):
            npr.GaussianProcess(variance=0.)

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

    def test_predict_prior(self):
        batch_size = 5
        num_points = 10
        x_dim = 3
        y_dim = 2
        x = torch.randn(batch_size, num_points, x_dim)
        y_mean, y_cov = self.model.predict(x, y_dim)
        self.assertTupleEqual(y_mean.size(), (batch_size, num_points, y_dim))
        self.assertTupleEqual(
            y_cov.size(), (batch_size, num_points, num_points))

        self.assertTrue((y_mean == 0).all().item)

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

    def test_sample(self):
        batch_size = 5
        num_points = 10
        x_dim = 3
        y_dim = 2
        x = torch.randn(batch_size, num_points, x_dim)
        y = torch.randn(batch_size, num_points, y_dim)

        # Sample from prior
        y_sample = self.model.sample(x, y_dim, single_params=True)
        self.assertTupleEqual(y_sample.size(), (batch_size, num_points, y_dim))

        # Sample params
        y_sample = self.model.sample(x, y_dim, single_params=False)
        self.assertTupleEqual(y_sample.size(), (batch_size, num_points, y_dim))

        # Sample from posterior
        x_sample = torch.randn(batch_size, num_points + 4, x_dim)
        self.model.fit(x, y)
        y_sample = self.model.sample(x_sample)
        self.assertTupleEqual(
            y_sample.size(), (batch_size, num_points + 4, y_dim))

    def test_sample_without_resample(self):
        batch_size = 5
        num_points = 10
        x_dim = 3
        y_dim = 2
        x = torch.randn(batch_size, num_points, x_dim)

        # Sample from prior
        y_sample = self.model.sample(x, y_dim, resample_params=False)
        self.assertTupleEqual(y_sample.size(), (batch_size, num_points, y_dim))

    def test_l2_scale(self):
        # Getter
        self.assertEqual(self.model.l2_scale, 0.4)

        # Setter
        self.model.l2_scale = 0.5
        self.assertEqual(self.model.l2_scale, 0.5)
        self.assertEqual(self.model.l2_scale_param, 0.5)

        self.model.l2_scale = torch.randn(10)
        self.assertTupleEqual(self.model.l2_scale.size(), (10,))

    def test_variance(self):
        # Getter
        self.assertEqual(self.model.variance, 1.0)

        # Setter
        self.model.variance = 0.5
        self.assertEqual(self.model.variance, 0.5)
        self.assertEqual(self.model.variance_param, 0.5)

        self.model.variance = torch.randn(10)
        self.assertTupleEqual(self.model.variance.size(), (10,))


if __name__ == "__main__":
    unittest.main()
