from sklearn.datasets import load_digits
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import train_test_split

from tabdpt import TabDPTClassifier

X, y = load_digits(return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

model = TabDPTClassifier()
temperature = model.get_optimum_temperature_cv(X, y)
model.fit(X_train, y_train)
print("Selected Temp: ", temperature)

lls = []
for temp in [x / 10 for x in range(1, 13)]:
    y_pred = model.predict(
        X_test,
        n_ensembles=8,
        context_size=2048,
        permute_classes=True,
        return_probs=True,
        seed=42,
        temperature=temp,
    )
    lls.append(log_loss(y_test, y_pred))
print(accuracy_score(y_test, y_pred.argmax(-1)))
print(lls)
