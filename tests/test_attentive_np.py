
import unittest

import torch

import neuralprocess as npr


class TestAttentiveNP(unittest.TestCase):

    def setUp(self):
        self.x_dim = 3
        self.y_dim = 2
        self.r_dim = 4
        self.s_dim = 5
        self.z_dim = 4
        self.n_head = 6
        self.model = npr.AttentiveNP(
            self.x_dim, self.y_dim, self.r_dim, self.s_dim, self.z_dim,
            self.n_head)

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
        self.assertTrue(loss_dict["loss"] > 0)
        self.assertIsInstance(loss_dict["nll"], torch.Tensor)
        self.assertTrue(loss_dict["nll"] > 0)
        self.assertIsInstance(loss_dict["kl"], torch.Tensor)
        self.assertTrue(loss_dict["kl"] > 0)

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
