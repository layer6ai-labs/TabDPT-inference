"""Semi-supervised few-shot classification: pseudo-labelling must help, and misuse must be loud."""
import inspect
import unittest

import numpy as np
from sklearn.datasets import load_digits
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from tabdpt import TabDPTClassifier, TabDPTSelfTrainingClassifier
from tabdpt.self_training import UNLABELLED

from device_utils import pick_device, seed_everything

DEVICE = pick_device()

N_SHOTS = 10
SEED = 42

# Over 20 seeds of this setup on CPU the gain averaged +0.034 (std 0.010, min +0.015, positive on every
# seed); seed 42 gives +0.034. The floor leaves headroom for hardware differences while still failing
# if the method loses most of its benefit.
MIN_GAIN = 0.02


def mask_all_but_n_shots(y, n_shots=N_SHOTS, seed=SEED):
    """Keep `n_shots` labels per class, marking every other row unlabelled."""
    rng = np.random.default_rng(seed)
    masked = np.full_like(y, UNLABELLED)
    for c in np.unique(y):
        masked[rng.permutation(np.flatnonzero(y == c))[:n_shots]] = c
    return masked


class TestSelfTrainingQuality(unittest.TestCase):
    """The method must actually beat the few-shot baseline it is meant to improve on."""

    @classmethod
    def setUpClass(cls):
        seed_everything(SEED)
        X, y = load_digits(return_X_y=True)
        cls.X_train, cls.X_test, cls.y_train, cls.y_test = train_test_split(
            X, y, test_size=0.33, random_state=SEED
        )
        cls.y_few_shot = mask_all_but_n_shots(cls.y_train)
        cls.labelled = cls.y_few_shot != UNLABELLED

        baseline = TabDPTClassifier(device=DEVICE)
        baseline.fit(cls.X_train[cls.labelled], cls.y_train[cls.labelled])
        cls.baseline_acc = accuracy_score(cls.y_test, baseline.predict(cls.X_test, seed=SEED))

        cls.model = TabDPTSelfTrainingClassifier(n_pseudo=1000, device=DEVICE)
        cls.model.fit(cls.X_train, cls.y_few_shot, seed=SEED)
        cls.self_training_acc = accuracy_score(cls.y_test, cls.model.predict(cls.X_test, seed=SEED))

    def test_beats_few_shot_baseline(self):
        """A larger pseudo-labelled context must improve on the labelled rows alone, not tie it."""
        gain = self.self_training_acc - self.baseline_acc
        print(f"\n[self-training] {self.self_training_acc:.4f} vs few-shot baseline {self.baseline_acc:.4f} "
              f"(gain {gain:+.4f})")
        self.assertGreater(
            self.self_training_acc, self.baseline_acc + MIN_GAIN,
            msg=f"self-training {self.self_training_acc:.4f} did not beat the {N_SHOTS}-shot baseline "
                f"{self.baseline_acc:.4f} by at least {MIN_GAIN}; pseudo-labelling is not helping",
        )

    def test_pseudo_labels_beat_chance(self):
        """The pseudo-labels must be mostly right, or the context is being poisoned."""
        pseudo_acc = accuracy_score(self.y_train[self.model.pseudo_label_indices_], self.model.pseudo_labels_)
        chance = 1 / len(np.unique(self.y_train))
        print(f"[self-training] pseudo-label accuracy {pseudo_acc:.4f} vs chance {chance:.4f}")
        self.assertGreater(pseudo_acc, chance, msg=f"pseudo-label accuracy {pseudo_acc:.4f} at chance {chance:.4f}")

    def test_pseudo_label_attributes(self):
        """Fitted attributes must line up with each other and point only at unlabelled rows."""
        idx, labels = self.model.pseudo_label_indices_, self.model.pseudo_labels_
        n_unlabelled = int((~self.labelled).sum())

        self.assertEqual(len(idx), len(labels))
        self.assertEqual(len(idx), min(1000, n_unlabelled))
        self.assertTrue(np.all(~self.labelled[idx]), "a labelled row was pseudo-labelled")
        self.assertTrue(set(np.unique(labels)).issubset(set(np.unique(self.y_train))))


class TestSelfTrainingContract(unittest.TestCase):
    """Edge cases and misuse, on small synthetic data."""

    @classmethod
    def setUpClass(cls):
        seed_everything(0)
        cls.X = np.random.normal(size=(80, 5)).astype(np.float32)
        cls.y = np.repeat(np.arange(4), 20)
        cls.y_masked = mask_all_but_n_shots(cls.y, n_shots=5, seed=0)
        cls.n_unlabelled = int((cls.y_masked == UNLABELLED).sum())
        cls.model = TabDPTSelfTrainingClassifier(n_pseudo=10, device=DEVICE)

    def set_n_pseudo(self, n_pseudo):
        """Temporarily change the pseudo-label budget without building another model."""
        original = self.model.n_pseudo
        self.addCleanup(setattr, self.model, "n_pseudo", original)
        self.model.n_pseudo = n_pseudo

    def test_n_pseudo_clamps_to_pool(self):
        """Asking for more pseudo-labels than there are unlabelled rows must clamp, not fail."""
        self.set_n_pseudo(10_000)
        self.model.fit(self.X, self.y_masked, seed=0)
        self.assertEqual(len(self.model.pseudo_labels_), self.n_unlabelled)

    def test_no_unlabelled_rows_warns_and_falls_back(self):
        """A fully labelled y is valid but pointless here, so it should say so and still work."""
        with self.assertWarns(UserWarning):
            self.model.fit(self.X, self.y, seed=0)
        self.assertEqual(len(self.model.pseudo_labels_), 0)
        self.assertEqual(self.model.predict(self.X, seed=0).shape, self.y.shape)

    def test_no_labelled_rows_raises(self):
        with self.assertRaises(ValueError):
            self.model.fit(self.X, np.full(len(self.y), UNLABELLED))

    def test_single_labelled_class_raises(self):
        """One class is not a classification problem; the message must say so."""
        y_one_class = np.where(self.y == 0, 0, UNLABELLED)
        with self.assertRaises(ValueError):
            self.model.fit(self.X, y_one_class)

    def test_non_contiguous_labels_raise(self):
        """Labels must be 0..n_classes-1, since pseudo-labels come back as class indices."""
        y_bad = np.where(self.y_masked == UNLABELLED, UNLABELLED, self.y_masked * 2)  # 0, 2, 4, 6
        with self.assertRaises(ValueError):
            self.model.fit(self.X, y_bad)

    def test_mismatched_x_and_y_raise(self):
        with self.assertRaises(AssertionError):
            self.model.fit(self.X[:-1], self.y_masked)

    def test_invalid_arguments(self):
        """n_pseudo is validated before the checkpoint loads; n_ensembles at fit time."""
        with self.assertRaises(AssertionError):
            TabDPTSelfTrainingClassifier(n_pseudo=0, device=DEVICE)
        with self.assertRaises(AssertionError):
            self.model.fit(self.X, self.y_masked, n_ensembles=0)

    def test_constructor_mirrors_parent(self):
        """The constructor restates TabDPTClassifier's arguments so sklearn can see them; they must not drift."""
        parent = inspect.signature(TabDPTClassifier.__init__).parameters
        child = dict(inspect.signature(TabDPTSelfTrainingClassifier.__init__).parameters)
        child.pop("n_pseudo")
        self.assertEqual(list(child), list(parent))
        for name in parent:
            self.assertEqual(child[name].default, parent[name].default, msg=f"default for {name} differs")

    def test_get_params_includes_every_argument(self):
        """sklearn's clone() rebuilds the estimator from get_params(), so nothing may be missing."""
        params = self.model.get_params()
        self.assertEqual(params["n_pseudo"], 10)
        self.assertEqual(params["device"], self.model.device)
        expected = set(inspect.signature(TabDPTSelfTrainingClassifier.__init__).parameters) - {"self"}
        self.assertEqual(set(params), expected)


if __name__ == "__main__":
    unittest.main()
