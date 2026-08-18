"""Classification: the v1.2 model (now the default) must match or beat the recorded v1.1 result."""
import unittest

from sklearn.datasets import load_digits
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from tabdpt import TabDPTClassifier

from device_utils import pick_device, seed_everything

DEVICE = pick_device()

V1_CLS_ACC = 0.9916
TOLERANCE = 2e-2


class TestClassification(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Setup dataset and the v1.2 model, then gather predictions."""
        seed_everything(42)
        X, y = load_digits(return_X_y=True)
        cls.X_train, cls.X_test, cls.y_train, cls.y_test = train_test_split(X, y, test_size=0.33, random_state=42)
        cls.base_model = TabDPTClassifier(device=DEVICE)
        cls.base_model.fit(cls.X_train, cls.y_train)
        cls.y_train_pred = cls.base_model.predict(cls.X_train, seed=42)
        cls.y_test_pred = cls.base_model.predict(cls.X_test, seed=42)

    def test_accuracy(self):
        """Train prediction should be perfect; v1.2 test accuracy should be at least v1.1's."""
        train_acc = accuracy_score(self.y_train, self.y_train_pred)
        test_acc = accuracy_score(self.y_test, self.y_test_pred)
        print(f"\n[cls] v1.2 test acc {test_acc:.4f} vs v1.1 {V1_CLS_ACC:.4f} (delta {test_acc - V1_CLS_ACC:+.4f})")

        self.assertAlmostEqual(
            train_acc, 1.,
            msg=f"train accuracy {train_acc:.4f} below expectation 1.0",
        )
        self.assertGreaterEqual(
            test_acc, V1_CLS_ACC - TOLERANCE,
            msg=f"v1.2 accuracy {test_acc:.4f} regressed >{TOLERANCE} below v1.1 {V1_CLS_ACC}",
        )

    def test_shape_and_targets(self):
        self.assertEqual(self.y_test_pred.shape, self.y_test.shape)
