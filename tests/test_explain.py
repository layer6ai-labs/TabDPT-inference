import unittest

import numpy as np

from tabdpt import TabDPTClassifier, TabDPTRegressor
from tabdpt.explain import integrated_gradients

from device_utils import pick_device, seed_everything

DEVICE = pick_device()
N_TRAIN = 2000
N_TEST = 100
N_COL = 4

class TestIntegratedGradients(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        seed_everything(0)
        cls.gen = np.random.default_rng(0)
        cls.X_train = cls.gen.normal(size=(N_TRAIN, N_COL))
        cls.X_test = cls.gen.normal(size=(N_TEST, N_COL))
        # Expected attributions: feature 1 > feature 2, feature 0 and 3 uninmportant
        cls.y_reg = cls.X_train[:,1] + 0.1*cls.X_train[:,2]
        cls.y_cls = cls.y_reg > 0

    def validate_attrs(self, attrs):
        self.assertEqual(attrs.shape, (N_TEST, N_COL))
        agg = np.mean(np.abs(attrs), axis=0)
        self.assertGreater(agg[1], agg[2])
        self.assertGreater(agg[2], agg[0])
        self.assertGreater(agg[2], agg[3])
        self.assertGreater(agg[1]/10, agg[0], "most important feature should be clearly above unimportant ones")
        self.assertGreater(agg[1]/10, agg[3], "most important feature should be clearly above unimportant ones")

    def test_regression(self):
        model = TabDPTRegressor(device=DEVICE)
        model.fit(self.X_train, self.y_reg)
        attrs = integrated_gradients(model, self.X_test)
        self.validate_attrs(attrs)

    def test_classification(self):
        model = TabDPTClassifier(device=DEVICE)
        model.fit(self.X_train, self.y_cls)
        attrs = integrated_gradients(model, self.X_test, class_index=1)
        self.validate_attrs(attrs)

    def test_no_retrieval(self):
        model = TabDPTClassifier(context_reduction="retrieval", device=DEVICE)
        model.fit(self.X_train, self.y_cls)
        with self.assertRaises(NotImplementedError):
            integrated_gradients(model, self.X_test, context_size=10)

    def test_no_large_cls(self):
        model = TabDPTClassifier(device=DEVICE)
        n_classes = model.max_num_classes + 2
        y = np.tile(np.arange(n_classes), len(self.y_cls) // n_classes + 1)[:len(self.y_cls)]
        model.fit(self.X_train, y)
        with self.assertRaises(NotImplementedError):
            integrated_gradients(model, self.X_test)

    def test_pca(self):
        model = TabDPTRegressor(device=DEVICE)
        n_features = model.max_features + 10
        X_train = np.hstack((self.X_train, self.gen.normal(size=(N_TRAIN, n_features-N_COL))))
        X_test = np.hstack((self.X_test, self.gen.normal(size=(N_TEST, n_features-N_COL))))
        model.fit(X_train, self.y_reg)
        attrs = integrated_gradients(model, X_test)
        self.assertEqual(attrs.shape, (N_TEST, n_features))
        agg = np.mean(np.abs(attrs), axis=0)
        self.assertGreater(agg[1], agg[2])
        self.assertGreater(agg[2], max(agg[0], np.max(agg[3:])))

    def test_feature_subsample(self):
        model = TabDPTRegressor(feature_reduction="subsample", device=DEVICE, compile=False, verbose=False)
        n_features = model.max_features + 10
        X_train = np.hstack((self.X_train, self.gen.normal(size=(N_TRAIN, n_features-N_COL))))
        X_test = np.hstack((self.X_test, self.gen.normal(size=(N_TEST, n_features-N_COL))))
        model.fit(X_train, self.y_reg)
        attrs = integrated_gradients(model, X_test)
        self.assertEqual(attrs.shape, (N_TEST, n_features))
        # Can't assert feature ordering because relevant features might not have been sampled

    def test_missing_indicators(self):
        model = TabDPTRegressor(missing_indicators=True, device=DEVICE)
        X_train = self.X_train.copy()
        X_train[self.gen.uniform(size=X_train.shape[0]) > 0.9, 1] = np.nan
        X_train[self.gen.uniform(size=X_train.shape[0]) > 0.5, 3] = np.nan
        X_test = self.X_test.copy()
        X_test[self.gen.uniform(size=X_test.shape[0]) > 0.9, 1] = np.nan
        X_test[self.gen.uniform(size=X_test.shape[0]) > 0.5, 3] = np.nan
        model.fit(X_train, self.y_reg)
        attrs = integrated_gradients(model, X_test)
        self.validate_attrs(attrs)

    def test_batching(self):
        model = TabDPTRegressor(device=DEVICE)
        model.fit(self.X_train, self.y_reg)
        attrs = integrated_gradients(model, self.X_test, batch_size=N_TEST // 10)
        self.validate_attrs(attrs)
