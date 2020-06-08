
import unittest

import torch

import neuralprocess as npr


class TestFunctionalNP(unittest.TestCase):

    def setUp(self):
        self.x_dim = 3
        self.y_dim = 2
        self.h_dim = 4
        self.u_dim = 5
        self.z_dim = 6
        self.model = npr.FunctionalNP(
            self.x_dim, self.y_dim, self.h_dim, self.u_dim, self.z_dim)

    def test_sample(self):
        # Data
        batch_size = 12
        num_context = 6
        num_target = 10
        x_context = torch.randn(batch_size, num_context, self.x_dim)
        y_context = torch.randn(batch_size, num_context, self.y_dim)
        x_target = torch.randn(batch_size, num_target, self.x_dim)

        # Forward
        mu, var = self.model.sample(x_context, y_context, x_target)

        self.assertTupleEqual(mu.size(), (batch_size, num_target, self.y_dim))
        self.assertTupleEqual(
            var.size(), (batch_size, num_target, self.y_dim))

    def test_loss_func(self):
        # Data
        batch_size = 12
        num_context = 6
        num_target = 10
        x_context = torch.randn(batch_size, num_context, self.x_dim)
        y_context = torch.randn(batch_size, num_context, self.y_dim)
        x_target = torch.randn(batch_size, num_target, self.x_dim)
        y_target = torch.randn(batch_size, num_target, self.y_dim)

        # Calculate loss
        loss_dict = self.model.loss_func(
            x_context, y_context, x_target, y_target)

        self.assertIsInstance(loss_dict, dict)
        self.assertIsInstance(loss_dict["loss"], torch.Tensor)
        self.assertIsInstance(loss_dict["nll_loss_c"], torch.Tensor)
        self.assertIsInstance(loss_dict["kl_loss_c"], torch.Tensor)
        self.assertIsInstance(loss_dict["nll_loss_t"], torch.Tensor)
        self.assertIsInstance(loss_dict["kl_loss_t"], torch.Tensor)

    def test_forward(self):
        # Data
        batch_size = 12
        num_context = 6
        num_target = 10
        x_context = torch.randn(batch_size, num_context, self.x_dim)
        y_context = torch.randn(batch_size, num_context, self.y_dim)
        x_target = torch.randn(batch_size, num_target, self.x_dim)

        # Forward
        mu = self.model(x_context, y_context, x_target)
        self.assertTupleEqual(mu.size(), (batch_size, num_target, self.y_dim))


if __name__ == "__main__":
    unittest.main()
