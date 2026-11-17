from functools import wraps
from typing import Literal

import faiss
import numpy as np
import torch
from sklearn.preprocessing import OrdinalEncoder
from torch.nn.attention import SDPBackend, sdpa_kernel


def generate_random_permutation(N, seed=None):
    generator = torch.Generator()
    if seed is not None:
        generator.manual_seed(seed)

    return torch.randperm(N, generator=generator)


def flash_context(func):
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        if getattr(self, "use_flash", False):
            assert torch.cuda.is_available(), "FlashAttention requires CUDA support"
            bf_support = torch.cuda.get_device_capability()[0] >= 8
            dtype = torch.bfloat16 if bf_support else torch.float16
            with (
                torch.autocast(device_type="cuda", dtype=dtype),
                sdpa_kernel(SDPBackend.FLASH_ATTENTION),
            ):
                return func(self, *args, **kwargs)
        else:
            return func(self, *args, **kwargs)

    return wrapper


def maskmean(x, mask, dim):
    x = torch.where(mask, x, 0)
    return x.sum(dim=dim, keepdim=True) / mask.sum(dim=dim, keepdim=True)


def maskstd(x, mask, dim=0):
    num = mask.sum(dim=dim, keepdim=True)
    mean = maskmean(x, mask, dim=dim)
    diffs = torch.where(mask, mean - x, 0)
    return ((diffs**2).sum(dim=dim, keepdim=True) / (num - 1)) ** 0.5


def normalize_data(data, eval_pos=-1, dim=0, return_mean_std: bool = False):
    X = data[:eval_pos] if eval_pos > 0 else data
    X = X.detach()  # Allow gradient computations wrt inputs
    mask = ~torch.isnan(X)
    mean = maskmean(X, mask, dim=dim)
    std = maskstd(X, mask, dim=dim) + 1e-6
    data = (data - mean) / std
    if return_mean_std:
        return data, mean, std
    return data


def clip_outliers(data, eval_pos=-1, n_sigma=4, dim=0):
    X = data[:eval_pos] if eval_pos > 0 else data
    X = X.detach()  # Allow gradient computations wrt inputs
    mask = ~torch.isnan(X)
    mean = maskmean(X, mask, dim=dim)
    cutoff = n_sigma * maskstd(X, mask, dim=dim)
    mask &= cutoff >= torch.abs(X - mean)
    cutoff = n_sigma * maskstd(X, mask, dim=dim)
    return torch.clip(data, mean - cutoff, mean + cutoff)


def convert_to_torch_tensor(input):
    if isinstance(input, np.ndarray):
        return torch.from_numpy(input)
    elif torch.is_tensor(input):
        return input
    raise TypeError("Input must be a NumPy array or a PyTorch tensor.")


def pad_x(X: torch.Tensor, num_features: int = 100):
    if num_features is None:
        return X
    n_features = X.shape[-1]
    zero_feature_padding = torch.zeros((*X.shape[:-1], num_features - n_features), device=X.device)
    return torch.cat([X, zero_feature_padding], dim=-1)


class FAISS:
    def __init__(self, X, metric: Literal["l2", "ip"] = "l2"):
        assert isinstance(X, np.ndarray), "X must be a numpy array"
        X = np.ascontiguousarray(X)
        X = X.astype(np.float32)
        if metric == "l2":
            self.index = faiss.IndexFlatL2(X.shape[1])
        elif metric == "ip":
            self.index = faiss.IndexFlatIP(X.shape[1])
        else:
            raise ValueError('metric must be "l2" or "ip"')
        self.index.add(X)

    def get_knn_indices(self, queries, k):
        if isinstance(queries, torch.Tensor):
            queries = queries.cpu().numpy()
        queries = np.ascontiguousarray(queries)
        assert isinstance(k, int)

        knns = self.index.search(queries, k)
        indices_Xs = knns[1]
        return indices_Xs

class Log1pScaler:
    def fit(self, X: np.ndarray):
        pass

    def fit_transform(self, X: np.ndarray):
        return np.log1p(np.abs(X)) * np.sign(X)

    def transform(self, X: np.ndarray):
        return np.log1p(np.abs(X)) * np.sign(X)


def _column_kind(dtype) -> str | None:
    """How a DataFrame column is encoded, from its dtype (pandas dtypes expose a numpy-style ``kind``)."""
    kind = getattr(dtype, "kind", None)
    if kind in ("i", "u", "f"):  # also the nullable Int64 / Float64 dtypes
        return "numeric"
    if kind in ("b", "O", "S", "U"):  # bool, object, string and category dtypes
        return "categorical"
    if kind in ("M", "m"):
        return "temporal"
    return None


def _to_seconds(col):
    """Seconds since 1970-01-01 (UTC if timezone-aware) for a datetime column, total seconds for a timedelta one."""
    if col.dtype.kind == "M":
        if getattr(col.dt, "tz", None) is not None:
            col = col.dt.tz_convert(None)
        col = col - np.datetime64(0, "s")
    return col.dt.total_seconds().to_numpy(dtype=np.float64, na_value=np.nan)


class DataFrameEncoder:
    """Encode a pandas DataFrame as a float array with one column per input column, in the same order.

    - Numeric columns, including the nullable ``Int64``/``Float64`` dtypes, are cast to float.
    - String, object, category and boolean columns are ordinal-encoded. Missing values and categories not seen
      during ``fit`` become NaN, so they get the same missing-value handling as numeric columns.
    - Datetime columns become seconds since 1970-01-01 (UTC if timezone-aware), timedelta columns total seconds.

    Missing values are NaN in the output. Columns of any other dtype raise a ``TypeError``.
    """

    def fit(self, X):
        self.feature_names_in_ = list(X.columns)
        kinds = [_column_kind(dtype) for dtype in X.dtypes]
        unsupported = [f"{name!r} ({dtype})" for name, dtype, kind in zip(X.columns, X.dtypes, kinds) if kind is None]
        if unsupported:
            raise TypeError(f"Unsupported column dtypes: {', '.join(unsupported)}. Convert them to numbers or strings.")
        self.kinds_ = np.array(kinds)
        self.ordinal_encoder_ = None
        if (self.kinds_ == "categorical").any():
            self.ordinal_encoder_ = OrdinalEncoder(
                handle_unknown="use_encoded_value", unknown_value=np.nan, encoded_missing_value=np.nan
            ).fit(self._categorical(X))
        return self

    def _categorical(self, X):
        cat = X.loc[:, self.kinds_ == "categorical"]
        return cat.astype(object).where(cat.notna(), np.nan)  # pd.NA and None become NaN

    def transform(self, X) -> np.ndarray:
        if list(X.columns) != self.feature_names_in_:
            missing = [name for name in self.feature_names_in_ if name not in X.columns]
            if missing:
                raise ValueError(f"X is missing columns seen during fit: {missing}")
            X = X[self.feature_names_in_]
        out = np.empty(X.shape, dtype=np.float64)
        numeric = self.kinds_ == "numeric"
        if numeric.any():
            out[:, numeric] = X.loc[:, numeric].to_numpy(dtype=np.float64, na_value=np.nan)
        if self.ordinal_encoder_ is not None:
            out[:, self.kinds_ == "categorical"] = self.ordinal_encoder_.transform(self._categorical(X))
        for j in np.flatnonzero(self.kinds_ == "temporal"):
            out[:, j] = _to_seconds(X.iloc[:, j])
        return out
