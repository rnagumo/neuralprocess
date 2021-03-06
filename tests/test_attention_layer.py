
import unittest

import torch

import neuralprocess as npr


class TestScaledDotProductAttention(unittest.TestCase):

    def setUp(self):
        self.attention = npr.ScaledDotProductAttention()

    def test_forward_3d(self):
        batch = 10
        len_q = 5
        len_k = 4
        d_k = 3
        d_v = 2

        # Data
        q = torch.ones(batch, len_q, d_k)
        k = torch.ones(batch, len_k, d_k)
        v = torch.ones(batch, len_k, d_v)

        y = self.attention(q, k, v)
        self.assertTupleEqual(y.size(), (batch, len_q, d_v))
        self.assertTrue((y == 1).all())

    def test_forward_4d(self):
        batch = 10
        num = 8
        len_q = 5
        len_k = 4
        d_k = 3
        d_v = 2

        # Data
        q = torch.ones(batch, num, len_q, d_k)
        k = torch.ones(batch, num, len_k, d_k)
        v = torch.ones(batch, num, len_k, d_v)

        y = self.attention(q, k, v)
        self.assertTupleEqual(y.size(), (batch, num, len_q, d_v))
        self.assertTrue((y == 1).all())


class TestMultiHeadAttention(unittest.TestCase):

    def setUp(self):
        self.k_dim = 3
        self.v_dim = 2
        self.d_k = 6
        self.d_v = 5
        self.n_head = 4
        self.attention = npr.MultiHeadAttention(
            self.k_dim, self.v_dim, self.d_k, self.d_v, self.n_head)

    def test_forward(self):
        # Data
        batch = 10
        len_q = 5
        len_k = 4
        q = torch.ones(batch, len_q, self.k_dim)
        k = torch.ones(batch, len_k, self.k_dim)
        v = torch.ones(batch, len_k, self.v_dim)

        y = self.attention(q, k, v)
        self.assertTupleEqual(y.size(), (batch, len_q, self.v_dim))

    def test_size_error(self):
        # Data
        batch = 10
        len_q = 5
        len_k = 4

        # Test case 1. q_dim != k_dim
        q = torch.ones(batch, len_q, self.k_dim)
        k = torch.ones(batch, len_k, self.k_dim + 1)
        v = torch.ones(batch, len_k, self.v_dim)

        with self.assertRaises(ValueError):
            _ = self.attention(q, k, v)

        # Test case 2. len_k != len_v
        q = torch.ones(batch, len_q, self.k_dim)
        k = torch.ones(batch, len_k, self.k_dim)
        v = torch.ones(batch, len_k + 1, self.v_dim)

        with self.assertRaises(ValueError):
            _ = self.attention(q, k, v)


class TestSelfAttention(unittest.TestCase):

    def setUp(self):
        self.x_dim = 3
        self.d_x = 6
        self.n_head = 4
        self.attention = npr.SelfAttention(
            self.x_dim, self.d_x, self.n_head)

    def test_forward(self):
        # Data
        batch = 10
        len_x = 5
        x = torch.ones(batch, len_x, self.x_dim)

        y = self.attention(x)
        self.assertTupleEqual(y.size(), (batch, len_x, self.x_dim))


if __name__ == "__main__":
    unittest.main()
