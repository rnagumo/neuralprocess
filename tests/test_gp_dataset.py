
import unittest

import neuralprocess as npr


class TestGPDataset(unittest.TestCase):

    def setUp(self):
        self.params = {
            "batch_size": 100,
            "num_context_max": 10,
            "num_target_max": 20,
            "x_dim": 3,
            "y_dim": 2,
        }

    def _base_case(self, train):
        indices = [0, 1, 2]
        dataset = npr.GPDataset(train=train, **self.params)
        x_ctx, y_ctx, x_tgt, y_tgt = dataset[indices]

        num_context_max = self.params["num_context_max"]
        num_target_max = self.params["num_target_max"]
        x_dim = self.params["x_dim"]
        y_dim = self.params["y_dim"]

        self.assertEqual(x_ctx.size(0), len(indices))
        self.assertLessEqual(x_ctx.size(1), num_context_max)
        self.assertEqual(x_ctx.size(2), x_dim)

        self.assertEqual(y_ctx.size(0), len(indices))
        self.assertLessEqual(y_ctx.size(1), num_context_max)
        self.assertEqual(y_ctx.size(2), y_dim)

        self.assertEqual(x_tgt.size(0), len(indices))
        self.assertLessEqual(x_tgt.size(1), num_target_max)
        self.assertEqual(x_tgt.size(2), x_dim)

        self.assertEqual(y_tgt.size(0), len(indices))
        self.assertLessEqual(y_tgt.size(1), num_target_max)
        self.assertEqual(y_tgt.size(2), y_dim)

    def _small_case(self, train):
        indices = [0, 1, 2]
        dataset = npr.GPDataset(train=train, **self.params)

        num_context_min = 30
        num_target_min = 30
        x_dim = self.params["x_dim"]
        y_dim = self.params["y_dim"]

        dataset.generate_dataset(num_context_min, num_target_min)
        x_ctx, y_ctx, x_tgt, y_tgt = dataset[indices]

        self.assertEqual(x_ctx.size(0), len(indices))
        self.assertEqual(x_ctx.size(1), num_context_min)
        self.assertEqual(x_ctx.size(2), x_dim)

        self.assertEqual(y_ctx.size(0), len(indices))
        self.assertEqual(y_ctx.size(1), num_context_min)
        self.assertEqual(y_ctx.size(2), y_dim)

        self.assertEqual(x_tgt.size(0), len(indices))
        self.assertEqual(x_tgt.size(1), num_target_min)
        self.assertEqual(x_tgt.size(2), x_dim)

        self.assertEqual(y_tgt.size(0), len(indices))
        self.assertEqual(y_tgt.size(1), num_target_min)
        self.assertEqual(y_tgt.size(2), y_dim)

    def test_get_item_train(self):
        self._base_case(train=True)

    def test_get_item_test(self):
        self._base_case(train=False)

    def test_large_context(self):
        self.params["num_context_max"] = 50
        self._base_case(train=True)
        self._base_case(train=False)

    def test_large_target(self):
        self.params["num_target_max"] = 50
        self._base_case(train=True)
        self._base_case(train=False)

    def test_len(self):
        dataset = npr.GPDataset(train=True, **self.params)
        self.assertEqual(len(dataset), self.params["batch_size"])

    def test_num_context(self):
        dataset = npr.GPDataset(train=True, **self.params)
        self.assertLessEqual(
            dataset.num_context, self.params["num_context_max"])

        dataset = npr.GPDataset(train=False, **self.params)
        self.assertLessEqual(
            dataset.num_context, self.params["num_context_max"])

    def test_num_target(self):
        dataset = npr.GPDataset(train=True, **self.params)
        self.assertLessEqual(
            dataset.num_target, self.params["num_target_max"])

        dataset = npr.GPDataset(train=False, **self.params)
        self.assertLessEqual(
            dataset.num_target, self.params["num_target_max"])

    def test_small_num_context(self):
        self.params["num_context_max"] = 1
        self._small_case(train=True)
        self._small_case(train=False)

    def test_small_num_target(self):
        self.params["num_target_max"] = 1
        self._small_case(train=True)
        self._small_case(train=False)

    def test_generate_with_resample_params(self):
        dataset = npr.GPDataset(train=True, **self.params)
        dataset.generate_dataset(resample_params=True)

        indices = [0, 1, 2]
        x_ctx, y_ctx, x_tgt, y_tgt = dataset[indices]

        num_context_max = self.params["num_context_max"]
        num_target_max = self.params["num_target_max"]
        x_dim = self.params["x_dim"]
        y_dim = self.params["y_dim"]

        self.assertEqual(x_ctx.size(0), len(indices))
        self.assertLessEqual(x_ctx.size(1), num_context_max)
        self.assertEqual(x_ctx.size(2), x_dim)

        self.assertEqual(y_ctx.size(0), len(indices))
        self.assertLessEqual(y_ctx.size(1), num_context_max)
        self.assertEqual(y_ctx.size(2), y_dim)

        self.assertEqual(x_tgt.size(0), len(indices))
        self.assertLessEqual(x_tgt.size(1), num_target_max)
        self.assertEqual(x_tgt.size(2), x_dim)

        self.assertEqual(y_tgt.size(0), len(indices))
        self.assertLessEqual(y_tgt.size(1), num_target_max)
        self.assertEqual(y_tgt.size(2), y_dim)


if __name__ == "__main__":
    unittest.main()
