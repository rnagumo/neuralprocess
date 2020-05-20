
import unittest

import torch

import neuralprocess as npr


class TestEncoder(unittest.TestCase):

    def setUp(self):
        self.x_dim = 3
        self.y_dim = 2
        self.r_dim = 4
        self.model = npr.ConditionalNP(self.x_dim, self.y_dim, self.r_dim)

    def test_query(self):
        # Data
        batch_size = 12
        num_context = 6
        num_target = 10
        x_context = torch.randn(batch_size, num_context, self.x_dim)
        y_context = torch.randn(batch_size, num_context, self.y_dim)
        x_target = torch.randn(batch_size, num_target, self.x_dim)

        # Forward
        mu, logvar = self.model.query(x_context, y_context, x_target)

        self.assertTupleEqual(mu.size(), (batch_size, num_target, self.y_dim))
        self.assertTupleEqual(
            logvar.size(), (batch_size, num_target, self.y_dim))

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
        self.assertTupleEqual(
            loss_dict["loss"].size(), (batch_size, num_target))
        self.assertTrue((loss_dict["loss"] > 0).all())


if __name__ == "__main__":
    unittest.main()
