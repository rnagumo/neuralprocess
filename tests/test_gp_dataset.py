
import unittest

import neuralprocess as npr


class TestGPDataset(unittest.TestCase):

    def setUp(self):
        self.params = {
            "batch_size": 20,
            "num_context": 10,
            "num_target": 400,
            "x_dim": 3,
            "y_dim": 2,
        }

    def test_train_dataset(self):
        dataset = npr.GPDataset(train=True, **self.params)

        batch_size = self.params["batch_size"]
        num_context = self.params["num_context"]
        num_target = self.params["num_target"]
        x_dim = self.params["x_dim"]
        y_dim = self.params["y_dim"]

        self.assertTupleEqual(
            dataset.x_context.size(), (batch_size, num_context, x_dim))
        self.assertTupleEqual(
            dataset.y_context.size(), (batch_size, num_context, y_dim))
        self.assertTupleEqual(
            dataset.x_target.size(), (batch_size, num_target, x_dim))
        self.assertTupleEqual(
            dataset.y_target.size(), (batch_size, num_target, y_dim))

    def test_test_dataset(self):
        dataset = npr.GPDataset(train=False, **self.params)

        batch_size = self.params["batch_size"]
        num_context = self.params["num_context"]
        num_target = self.params["num_target"]
        x_dim = self.params["x_dim"]
        y_dim = self.params["y_dim"]

        self.assertTupleEqual(
            dataset.x_context.size(), (batch_size, num_context, x_dim))
        self.assertTupleEqual(
            dataset.y_context.size(), (batch_size, num_context, y_dim))
        self.assertTupleEqual(
            dataset.x_target.size(), (batch_size, num_target, x_dim))
        self.assertTupleEqual(
            dataset.y_target.size(), (batch_size, num_target, y_dim))

    def test_init_raise(self):
        self.params["num_context"] = 1000
        with self.assertRaises(ValueError):
            npr.GPDataset(train=True, **self.params)

    def test_get_item_train(self):
        dataset = npr.GPDataset(train=True, **self.params)
        index = [2, 4, 6]
        x_ctx, y_ctx, x_tgt, y_tgt = dataset[index]

        num_context = self.params["num_context"]
        num_target = self.params["num_target"]
        x_dim = self.params["x_dim"]
        y_dim = self.params["y_dim"]

        self.assertTupleEqual(x_ctx.size(), (3, num_context, x_dim))
        self.assertTupleEqual(y_ctx.size(), (3, num_context, y_dim))
        self.assertTupleEqual(x_tgt.size(), (3, num_target, x_dim))
        self.assertTupleEqual(y_tgt.size(), (3, num_target, y_dim))

    def test_get_item_test(self):
        dataset = npr.GPDataset(train=False, **self.params)
        index = [2, 4, 6]
        x_ctx, y_ctx, x_tgt, y_tgt = dataset[index]

        num_context = self.params["num_context"]
        num_target = self.params["num_target"]
        x_dim = self.params["x_dim"]
        y_dim = self.params["y_dim"]

        self.assertTupleEqual(x_ctx.size(), (3, num_context, x_dim))
        self.assertTupleEqual(y_ctx.size(), (3, num_context, y_dim))
        self.assertTupleEqual(x_tgt.size(), (3, num_target, x_dim))
        self.assertTupleEqual(y_tgt.size(), (3, num_target, y_dim))

    def test_len(self):
        dataset = npr.GPDataset(train=True, **self.params)
        self.assertEqual(len(dataset), self.params["batch_size"])


if __name__ == "__main__":
    unittest.main()
