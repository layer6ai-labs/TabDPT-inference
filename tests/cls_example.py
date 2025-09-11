from sklearn.datasets import load_digits
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from tabdpt import TabDPTClassifier

X, y = load_digits(return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

model = TabDPTClassifier()
model.fit(X_train, y_train)
y_pred = model.predict(
    X_test,
    n_ensembles=8,
    temperature=0.3,
    context_size=2048,
    permute_classes=True,
    seed=42,
)
print(accuracy_score(y_test, y_pred))
