import numpy as np
from sklearn.datasets import load_digits
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from tabdpt import TabDPTClassifier, TabDPTSelfTrainingClassifier

SEED = 42
N_SHOTS = 10

X, y = load_digits(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=SEED)

# Simulate the few-shot setting: keep N_SHOTS labels per class and mark every other row unlabelled.
rng = np.random.default_rng(SEED)
y_few_shot = np.full_like(y_train, -1)
for c in np.unique(y_train):
    y_few_shot[rng.permutation(np.flatnonzero(y_train == c))[:N_SHOTS]] = c
labelled = y_few_shot != -1
print(f"{labelled.sum()} labelled rows ({N_SHOTS} per class), {(~labelled).sum()} unlabelled")

# Baseline: a standard classifier only ever sees the few labelled rows.
baseline = TabDPTClassifier()
baseline.fit(X_train[labelled], y_train[labelled])
baseline_accuracy = accuracy_score(y_test, baseline.predict(X_test, seed=SEED))

# Self-training: pseudo-label the unlabelled rows, then keep the most confident ones as context.
model = TabDPTSelfTrainingClassifier(n_pseudo=1000)
model.fit(X_train, y_few_shot, seed=SEED)
self_training_accuracy = accuracy_score(y_test, model.predict(X_test, seed=SEED))

# The pseudo-labels are not all correct; the method works because most of them are.
pseudo_accuracy = accuracy_score(y_train[model.pseudo_label_indices_], model.pseudo_labels_)

print(f"few-shot baseline accuracy:  {baseline_accuracy:.4f}  ({labelled.sum()} context rows)")
print(f"self-training accuracy:      {self_training_accuracy:.4f}  "
      f"({labelled.sum() + len(model.pseudo_labels_)} context rows)")
print(f"pseudo-label accuracy:       {pseudo_accuracy:.4f}  ({len(model.pseudo_labels_)} pseudo-labels)")
