from sklearn.datasets import fetch_california_housing
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

from tabdpt import TabDPTRegressor

X, y = fetch_california_housing(return_X_y=True)
X, y = X[:2048], y[:2048]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

model = TabDPTRegressor()
model.fit(X_train, y_train)
y_pred = model.predict(X_test, n_ensembles=2, context_size=1024, seed=42)
print(r2_score(y_test, y_pred))
