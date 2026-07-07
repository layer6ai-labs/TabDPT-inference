from sklearn.datasets import fetch_california_housing
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

from tabdpt import TabDPTRegressor

X, y = fetch_california_housing(return_X_y=True)
X, y = X[:4096], y[:4096]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

model = TabDPTRegressor()
model.fit(X_train, y_train)

predict_kwargs = dict(n_ensembles=2, seed=42)
quantiles = [0.1, 0.5, 0.9]

# Default: expected value of the binned predictive distribution
y_mean = model.predict(X_test, **predict_kwargs)
print("mean R2:", r2_score(y_test, y_mean))

# Other point estimates from the same distribution
y_median = model.predict(X_test, output_type="median", **predict_kwargs)
y_mode = model.predict(X_test, output_type="mode", **predict_kwargs)
print("median R2:", r2_score(y_test, y_median))
print("mode R2:", r2_score(y_test, y_mode))

# Arbitrary quantile levels (one array per quantile)
quantile_preds = model.predict(
    X_test,
    output_type="quantiles",
    quantiles=quantiles,
    **predict_kwargs,
)
print("quantile shapes:", [q.shape for q in quantile_preds])

# All main statistics in one call
main_out = model.predict(
    X_test,
    output_type="main",
    quantiles=quantiles,
    **predict_kwargs,
)
print("main keys:", list(main_out.keys()))

# Full distributional output: main stats plus logits and BarDistribution
full_out = model.predict(
    X_test,
    output_type="full",
    quantiles=quantiles,
    **predict_kwargs,
)
print("full keys:", list(full_out.keys()))
print("logits shape:", tuple(full_out["logits"].shape))
