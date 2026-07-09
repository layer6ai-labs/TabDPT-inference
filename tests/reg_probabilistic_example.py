import numpy as np
import torch
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
    X_test, output_type="quantiles", quantiles=quantiles, **predict_kwargs,
)
print("quantile shapes:", [q.shape for q in quantile_preds])

# All main statistics in one call
main_out = model.predict(X_test, output_type="main", quantiles=quantiles, **predict_kwargs)
print("main keys:", list(main_out.keys()))

# Full distributional output: logits and BarDistribution over the target
full_out = model.predict(X_test, output_type="full", quantiles=quantiles, **predict_kwargs)
print("full keys:", list(full_out.keys()))
print("logits shape:", tuple(full_out["logits"].shape))

# Interval calibration and sharpness from the predictive histogram
probas = torch.softmax(full_out["logits"], dim=-1).cpu().numpy()
edges = full_out["criterion"].borders.cpu().numpy()
mids = (edges[:-1] + edges[1:]) / 2
cdf, y = np.cumsum(probas, axis=-1), np.asarray(y_test, float)

for level in (90, 95):
    alpha = 1 - level / 100
    ql, qu = alpha / 2, 1 - alpha / 2
    il = np.clip((cdf >= ql).argmax(1), 0, len(edges) - 1)
    iu = np.clip((cdf >= qu).argmax(1) + 1, 0, len(edges) - 1)
    lo, hi = edges[il], edges[iu]
    print(f"coverage_{level}:", np.mean((y >= lo) & (y <= hi)))
    iscore = (hi - lo) + (2 / alpha) * np.clip(lo - y, 0, None) + (2 / alpha) * np.clip(y - hi, 0, None)
    print(f"interval_score_{level}:", iscore.mean())

mu = (probas * mids).sum(1)
print("sharpness:", np.sqrt(np.clip((probas * mids**2).sum(1) - mu**2, 0, None)).mean())
