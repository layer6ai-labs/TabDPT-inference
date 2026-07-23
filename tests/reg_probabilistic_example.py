import numpy as np
import torch
from sklearn.datasets import fetch_california_housing
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

from tabdpt import (
    TabDPTRegressor,
    distribution_mean,
    distribution_median,
    distribution_mode,
    distribution_quantiles,
)

X, y = fetch_california_housing(return_X_y=True)
X, y = X[:4096], y[:4096]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

model = TabDPTRegressor()
model.fit(X_train, y_train)

predict_kwargs = dict(n_ensembles=2, seed=42)

# Default: expected value of the binned predictive distribution
y_mean = model.predict(X_test, **predict_kwargs)
print("mean R2:", r2_score(y_test, y_mean))

# Full distributional output: logits and BarDistribution over the target
full = model.predict(X_test, output_type="full", **predict_kwargs)
logits, criterion = full["logits"], full["criterion"]
print("logits shape:", tuple(logits.shape))
print("num bins:", criterion.num_bars)

# Derive other point estimates and quantiles from the same distribution
print("mean R2 (helper):", r2_score(y_test, distribution_mean(logits, criterion)))
print("median R2:", r2_score(y_test, distribution_median(logits, criterion)))
print("mode R2:", r2_score(y_test, distribution_mode(logits, criterion)))

quantiles = [0.1, 0.5, 0.9]
quantile_preds = distribution_quantiles(logits, criterion, quantiles)
print("quantile shapes:", [q.shape for q in quantile_preds])

# Interval calibration and sharpness from the predictive histogram
probas = torch.softmax(logits, dim=-1).cpu().numpy()
edges = criterion.borders.cpu().numpy()
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
