"""Integration tests exercising the inference code paths on small synthetic data."""
import unittest

import numpy as np

from tabdpt import TabDPTClassifier, TabDPTRegressor

from device_utils import pick_device, seed_everything

DEVICE = pick_device()
N_TRAIN = 60
N_CLASSES = 3


class TestClassifier(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        seed_everything(0)
        cls.X = np.random.normal(size=(N_TRAIN, 5)).astype(np.float32)
        cls.y = np.random.randint(0, N_CLASSES, N_TRAIN)
        cls.model = TabDPTClassifier(device=DEVICE)  # verbose=True covers the tqdm ensemble path
        cls.model.fit(cls.X, cls.y)

    def test_full_context_probs_sum_to_one(self):
        p = self.model.predict_proba(self.X, context_size=2048)
        self.assertEqual(p.shape, (N_TRAIN, N_CLASSES))
        np.testing.assert_allclose(p.sum(axis=1), 1.0, atol=1e-4)

    def test_retrieval_path(self):
        p = self.model.predict_proba(self.X, context_size=20)
        self.assertEqual(p.shape, (N_TRAIN, N_CLASSES))
        np.testing.assert_allclose(p.sum(axis=1), 1.0, atol=1e-4)

    def test_return_logits(self):
        logits = self.model.predict_proba(self.X, context_size=2048, return_logits=True)
        self.assertEqual(logits.shape[0], N_TRAIN)
        # Generally, rows shouldn't all sum to one - if they do, we're probably returning probabilities instead of logits
        self.assertFalse(np.allclose(logits.sum(axis=1), 1.0, atol=1e-4))

    def test_single_sample_is_2d(self):
        p = self.model.predict_proba(self.X[:1], context_size=2048)
        self.assertEqual(p.shape, (1, N_CLASSES))

    def test_predict_single_and_ensemble(self):
        single = self.model.predict(self.X, n_ensembles=1, context_size=2048, seed=0)
        ens = self.model.predict(self.X, n_ensembles=4, context_size=2048, seed=0)
        for pred in (single, ens):
            self.assertEqual(pred.shape, (N_TRAIN,))
            self.assertTrue(set(np.unique(pred)).issubset({0, 1, 2}))

    def test_ensemble_predict_proba_permute_flag(self):
        for permute in (True, False):
            p = self.model.ensemble_predict_proba(
                self.X, n_ensembles=2, context_size=2048, permute_classes=permute, seed=1
            )
            self.assertEqual(p.shape, (N_TRAIN, N_CLASSES))
            np.testing.assert_allclose(p.sum(axis=1), 1.0, atol=1e-4)

    def test_class_perm(self):
        p = self.model.predict_proba(self.X, context_size=2048, class_perm=np.array([1, 2, 0]), seed=0)
        self.assertEqual(p.shape, (N_TRAIN, N_CLASSES))
        p_noperm = self.model.predict_proba(self.X, context_size=2048, seed=0)
        np.testing.assert_allclose(p, p_noperm[:, [1, 2, 0]], atol=1e-2)


class TestLargeClass(unittest.TestCase):
    """num_classes > max_num_classes triggers the digit-decomposition path."""

    @classmethod
    def setUpClass(cls):
        seed_everything(1)
        n_classes = 18  # > max_num_classes (16) for the v1.2 weights
        n_rows = n_classes * 10
        cls.n_classes = n_classes
        cls.X = np.random.normal(size=(n_rows, 5)).astype(np.float32)
        cls.y = np.tile(np.arange(n_classes), n_rows // n_classes)
        cls.model = TabDPTClassifier(device=DEVICE, verbose=False)
        cls.model.fit(cls.X, cls.y)

    def test_large_class_probs(self):
        self.assertGreater(self.model.num_classes, self.model.max_num_classes)
        p = self.model.predict_proba(self.X, context_size=2048)
        self.assertEqual(p.shape, (self.X.shape[0], self.n_classes))
        np.testing.assert_allclose(p.sum(axis=1), 1.0, atol=1e-4)


class TestRegressor(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        seed_everything(2)
        cls.X = np.random.normal(size=(N_TRAIN, 5)).astype(np.float32)
        cls.y = np.random.normal(size=N_TRAIN).astype(np.float32)
        cls.model = TabDPTRegressor(device=DEVICE, verbose=False)
        cls.model.fit(cls.X, cls.y)

    def test_full_context(self):
        pred = self.model.predict(self.X, n_ensembles=1, context_size=2048)
        self.assertEqual(pred.shape, (N_TRAIN,))

    def test_retrieval_path(self):
        pred = self.model.predict(self.X, n_ensembles=1, context_size=20)
        self.assertEqual(pred.shape, (N_TRAIN,))

    def test_ensemble(self):
        pred = self.model.predict(self.X, n_ensembles=3, context_size=2048, seed=0)
        self.assertEqual(pred.shape, (N_TRAIN,))
        self.assertTrue(np.all(np.isfinite(pred)))


class TestEstimatorConfig(unittest.TestCase):
    """Estimator construction options: feature reduction, missing indicators, faiss metric."""

    def _cls_data(self, n_features, seed=3):
        seed_everything(seed)
        X = np.random.normal(size=(N_TRAIN, n_features)).astype(np.float32)
        y = np.random.randint(0, N_CLASSES, N_TRAIN)
        return X, y

    def test_feature_reduction_pca(self):
        X, y = self._cls_data(n_features=140)  # > max_features (128) -> PCA reduction
        model = TabDPTClassifier(device=DEVICE, feature_reduction="pca", verbose=False)
        model.fit(X, y)
        self.assertIsNotNone(model.projection)
        model.to("cpu")  # exercises the V.to(device) branch
        p = model.predict_proba(X, context_size=2048)
        self.assertEqual(p.shape, (N_TRAIN, N_CLASSES))

    def test_feature_reduction_subsample(self):
        X, y = self._cls_data(n_features=140)
        model = TabDPTClassifier(device=DEVICE, feature_reduction="subsample", verbose=False)
        model.fit(X, y)
        p = model.predict_proba(X, context_size=2048, seed=1)
        self.assertEqual(p.shape, (N_TRAIN, N_CLASSES))

    def test_missing_indicators(self):
        X, y = self._cls_data(n_features=5)
        X[::4, 0] = np.nan
        X[1::5, 2] = np.nan
        model = TabDPTClassifier(device=DEVICE, missing_indicators=True, verbose=False)
        model.fit(X, y)
        p = model.predict_proba(X, context_size=2048)
        self.assertEqual(p.shape, (N_TRAIN, N_CLASSES))

    def test_faiss_ip_metric(self):
        X, y = self._cls_data(n_features=5)
        model = TabDPTClassifier(device=DEVICE, faiss_metric="ip", verbose=False)
        model.fit(X, y)
        p = model.predict_proba(X, context_size=20)  # retrieval -> uses the ip index
        self.assertEqual(p.shape, (N_TRAIN, N_CLASSES))


if __name__ == "__main__":
    unittest.main()
