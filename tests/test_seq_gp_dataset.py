
import unittest

import neuralprocess as npr


class TestSequentialGPDataset(unittest.TestCase):

    def setUp(self):
        self.params = {
            "seq_len": 20,
            "batch_size": 100,
            "num_context_min": 3,
            "num_context_max": 10,
            "num_target_max": 20,
            "num_target_min": 2,
            "x_dim": 3,
            "y_dim": 2,
        }

    def _base_case(self, train):
        indices = [0, 1, 2]
        dataset = npr.SequentialGPDataset(train=train, **self.params)
        x_ctx, y_ctx, x_tgt, y_tgt = dataset[indices]

        seq_len = self.params["seq_len"]
        num_context_max = self.params["num_context_max"]
        num_target_max = self.params["num_target_max"]
        x_dim = self.params["x_dim"]
        y_dim = self.params["y_dim"]

        self.assertEqual(x_ctx.size(0), len(indices))
        self.assertEqual(x_ctx.size(1), seq_len)
        self.assertLessEqual(x_ctx.size(2), num_context_max)
        self.assertEqual(x_ctx.size(3), x_dim)

        self.assertEqual(y_ctx.size(0), len(indices))
        self.assertEqual(y_ctx.size(1), seq_len)
        self.assertLessEqual(y_ctx.size(2), num_context_max)
        self.assertEqual(y_ctx.size(3), y_dim)

        self.assertEqual(x_tgt.size(0), len(indices))
        self.assertEqual(x_tgt.size(1), seq_len)
        self.assertLessEqual(x_tgt.size(2), num_target_max)
        self.assertEqual(x_tgt.size(3), x_dim)

        self.assertEqual(y_tgt.size(0), len(indices))
        self.assertEqual(y_tgt.size(1), seq_len)
        self.assertLessEqual(y_tgt.size(2), num_target_max)
        self.assertEqual(y_tgt.size(3), y_dim)

    def test_train(self):
        self._base_case(train=True)

    def test_test(self):
        self._base_case(train=False)


if __name__ == "__main__":
    unittest.main()
