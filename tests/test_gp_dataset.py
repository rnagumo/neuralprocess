
import unittest

import neuralprocess as npr


class TestGPDataset(unittest.TestCase):

    def setUp(self):
        self.params = {
            "batch_size": 100,
            "num_context": 10,
            "num_target": 20,
            "x_dim": 3,
            "y_dim": 2,
        }

    def test_get_item_train(self):
        self._base_method(train=True)

    def test_get_item_test(self):
        self._base_method(train=False)

    def test_large_context_train(self):
        self.params["num_context"] = 50
        self._base_method(train=True)

    def test_large_context_test(self):
        self.params["num_context"] = 50
        self._base_method(train=False)

    def test_large_target_train(self):
        self.params["num_target"] = 50
        self._base_method(train=True)

    def test_large_target_test(self):
        self.params["num_target"] = 50
        self._base_method(train=False)

    def test_len(self):
        dataset = npr.GPDataset(train=True, **self.params)
        self.assertEqual(len(dataset), self.params["batch_size"])

    def _base_method(self, train):
        indices = [0, 1, 2]
        dataset = npr.GPDataset(train=train, **self.params)
        x_ctx, y_ctx, x_tgt, y_tgt = dataset[indices]

        num_context = self.params["num_context"]
        num_target = self.params["num_target"]
        x_dim = self.params["x_dim"]
        y_dim = self.params["y_dim"]

        self.assertEqual(x_ctx.size(0), len(indices))
        self.assertLessEqual(x_ctx.size(1), num_context)
        self.assertEqual(x_ctx.size(2), x_dim)

        self.assertEqual(y_ctx.size(0), len(indices))
        self.assertLessEqual(y_ctx.size(1), num_context)
        self.assertEqual(y_ctx.size(2), y_dim)

        self.assertEqual(x_tgt.size(0), len(indices))
        self.assertLessEqual(x_tgt.size(1), num_target)
        self.assertEqual(x_tgt.size(2), x_dim)

        self.assertEqual(y_tgt.size(0), len(indices))
        self.assertLessEqual(y_tgt.size(1), num_target)
        self.assertEqual(y_tgt.size(2), y_dim)


if __name__ == "__main__":
    unittest.main()
