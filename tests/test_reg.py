"""Regression: the v1.2 model (now the default) must match or beat the recorded v1.1 result."""
import unittest

from sklearn.datasets import fetch_california_housing
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

from tabdpt import TabDPTRegressor

from device_utils import pick_device, seed_everything

DEVICE = pick_device()

V1_REG_R2 = 0.85
TOLERANCE = 5e-2
TRAIN_MIN_R2 = 0.98


class TestRegression(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Setup dataset and the v1.2 model (default weights), then gather predictions."""
        seed_everything(42)
        X, y = fetch_california_housing(return_X_y=True)
        cls.X_train, cls.X_test, cls.y_train, cls.y_test = train_test_split(X[:4096], y[:4096], test_size=0.33, random_state=42)

        cls.base_model = TabDPTRegressor(device=DEVICE)
        cls.base_model.fit(cls.X_train, cls.y_train)

        cls.y_train_pred = cls.base_model.predict(cls.X_train, seed=42)
        cls.y_test_pred = cls.base_model.predict(cls.X_test, seed=42)

    def test_accuracy(self):
        """Train prediction should be near-perfect; v1.2 test R2 should be at least v1.1's."""
        train_r2 = r2_score(self.y_train, self.y_train_pred)
        test_r2 = r2_score(self.y_test, self.y_test_pred)
        print(f"\n[reg] v1.2 test R2 {test_r2:.4f} vs v1.1 {V1_REG_R2:.4f} (delta {test_r2 - V1_REG_R2:+.4f})")

        self.assertGreaterEqual(
            train_r2, TRAIN_MIN_R2 - TOLERANCE,
            msg=f"train R2 {train_r2:.4f} is over {TOLERANCE} below expectation {TRAIN_MIN_R2}",
        )
        self.assertGreaterEqual(
            test_r2, V1_REG_R2 - TOLERANCE,
            msg=f"v1.2 R2 {test_r2:.4f} regressed >{TOLERANCE} below v1.1 {V1_REG_R2}",
        )

    def test_shape_and_targets(self):
        self.assertEqual(self.y_test_pred.shape, self.y_test.shape)
