"""Integration test on utility functions."""

import unittest

import numpy as np
import torch

from tabdpt.utils import (
    FAISS,
    Log1pScaler,
    clip_outliers,
    convert_to_torch_tensor,
    flash_context,
    generate_random_permutation,
    maskmean,
    maskstd,
    normalize_data,
    pad_x,
)


class TestPermutation(unittest.TestCase):
    def test_seeded_is_reproducible(self):
        a = generate_random_permutation(50, seed=7)
        b = generate_random_permutation(50, seed=7)
        self.assertTrue(torch.equal(a, b))
        self.assertEqual(sorted(a.tolist()), list(range(50)))

    def test_unseeded_length(self):
        self.assertEqual(generate_random_permutation(10).shape[0], 10)


class TestFlashContext(unittest.TestCase):
    def test_non_flash_passthrough(self):
        class Dummy:
            use_flash = False

            @flash_context
            def run(self, x):
                return x * 2

        self.assertEqual(Dummy().run(3), 6)


class TestMaskedStats(unittest.TestCase):
    def setUp(self):
        self.x = torch.tensor([[1.0, 2.0], [3.0, 10.0], [5.0, 6.0]])
        self.mask = torch.tensor([[True, True], [True, False], [True, True]])

    def test_maskmean_ignores_masked(self):
        mean = maskmean(self.x, self.mask, dim=0)
        # Column 1 excludes the masked 10.0 -> mean of {2, 6} = 4.
        self.assertAlmostEqual(mean[0, 0].item(), 3.0, places=5)
        self.assertAlmostEqual(mean[0, 1].item(), 4.0, places=5)

    def test_maskstd_positive(self):
        std = maskstd(self.x, self.mask, dim=0)
        self.assertTrue(torch.all(std > 0))


class TestNormalizeData(unittest.TestCase):
    def test_zero_mean_unit_std(self):
        data = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
        out = normalize_data(data)
        self.assertAlmostEqual(out.mean().item(), 0.0, places=5)

    def test_return_mean_std(self):
        data = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
        out, mean, std = normalize_data(data, return_mean_std=True)
        self.assertEqual(mean.shape, (1, 1))
        self.assertEqual(std.shape, (1, 1))

    def test_eval_pos_uses_prefix(self):
        data = torch.tensor([[0.0], [0.0], [100.0]])
        # Stats computed on the first two rows only, which are 0's, so std collapses to ~1e-6.
        out = normalize_data(data, eval_pos=2)
        self.assertGreater(out[2, 0].item(), 1e6)


class TestClipOutliers(unittest.TestCase):
    def test_clips_extreme_value(self):
        data = torch.cat([torch.zeros(20, 1), torch.tensor([[1000.0]])], dim=0)
        clipped = clip_outliers(data, n_sigma=4)
        self.assertLess(clipped.max().item(), 1000.0)


class TestConvertToTorch(unittest.TestCase):
    def test_from_numpy(self):
        out = convert_to_torch_tensor(np.array([1.0, 2.0]))
        self.assertTrue(torch.is_tensor(out))

    def test_tensor_passthrough(self):
        t = torch.tensor([1.0])
        self.assertIs(convert_to_torch_tensor(t), t)

    def test_invalid_type_raises(self):
        with self.assertRaises(TypeError):
            convert_to_torch_tensor([1, 2, 3])


class TestPadX(unittest.TestCase):
    def test_pads_to_num_features(self):
        out = pad_x(torch.ones(2, 3), num_features=5)
        self.assertEqual(out.shape, (2, 5))
        self.assertEqual(out[:, 3:].sum().item(), 0.0)

    def test_none_is_noop(self):
        x = torch.ones(2, 3)
        self.assertIs(pad_x(x, None), x)


class TestFAISS(unittest.TestCase):
    def setUp(self):
        self.X = np.arange(20, dtype=np.float32).reshape(10, 2)

    def test_l2_self_nearest(self):
        index = FAISS(self.X, metric="l2")
        idx = index.get_knn_indices(self.X, k=1)
        self.assertTrue(np.array_equal(idx[:, 0], np.arange(10)))

    def test_ip_metric(self):
        index = FAISS(self.X, metric="ip")
        idx = index.get_knn_indices(self.X[:3], k=2)
        self.assertEqual(idx.shape, (3, 2))

    def test_torch_query(self):
        index = FAISS(self.X, metric="l2")
        idx = index.get_knn_indices(torch.from_numpy(self.X), k=1)
        self.assertEqual(idx.shape, (10, 1))

    def test_invalid_metric_raises(self):
        with self.assertRaises(ValueError):
            FAISS(self.X, metric="cosine")


class TestLog1pScaler(unittest.TestCase):
    def test_sign_preserving(self):
        scaler = Log1pScaler()
        scaler.fit(np.array([[1.0]]))
        out = scaler.fit_transform(np.array([[-1.0], [0.0], [np.e - 1]]))
        self.assertAlmostEqual(out[0, 0], -np.log1p(1.0), places=5)
        self.assertAlmostEqual(out[1, 0], 0.0, places=5)
        self.assertAlmostEqual(out[2, 0], 1.0, places=5)

    def test_transform_matches_fit_transform(self):
        scaler = Log1pScaler()
        X = np.array([[2.0], [-3.0]])
        np.testing.assert_allclose(scaler.transform(X), scaler.fit_transform(X))


if __name__ == "__main__":
    unittest.main()
