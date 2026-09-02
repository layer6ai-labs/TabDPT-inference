"""Helpers for piecewise-uniform distributions over binned continuous targets."""

from __future__ import annotations

from typing import TypedDict

import numpy as np
import torch

_DEFAULT_QUANTILES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


class FullPrediction(TypedDict):
    """Full distributional output of `TabDPTRegressor.predict`.

    A piecewise-uniform histogram: model logits over bins, with bin edges in `borders`.
    Softmax turns logits into bin probabilities; within each bin the density is uniform,
    so the support is `[borders[0], borders[-1]]`.

    Attributes:
        logits: Unnormalized log-probabilities of shape `(n_test, num_bars)`.
        borders: Strictly increasing bin edges of shape `(num_bars + 1,)`, in raw target space.
    """

    logits: torch.Tensor
    borders: torch.Tensor


def _to_numpy(values: torch.Tensor) -> np.ndarray:
    return values.detach().cpu().numpy()


def _bucket_widths(borders: torch.Tensor) -> torch.Tensor:
    return borders[1:] - borders[:-1]


def _mean(logits: torch.Tensor, borders: torch.Tensor) -> torch.Tensor:
    bucket_means = borders[:-1] + _bucket_widths(borders) / 2
    probs = torch.softmax(logits, dim=-1)
    return probs @ bucket_means


def _icdf(logits: torch.Tensor, borders: torch.Tensor, left_prob: float | torch.Tensor) -> torch.Tensor:
    probs = logits.softmax(dim=-1)
    cumprobs = torch.cumsum(probs, dim=-1)

    # Normalize left_prob to shape (*leading,) then (*leading, 1) for searchsorted.
    # Avoid `tensor(N,) * ones(N, 1)`, which broadcasts to (N, N) in PyTorch.
    if not torch.is_tensor(left_prob):
        left_prob = torch.full(
            cumprobs.shape[:-1], float(left_prob), device=logits.device, dtype=cumprobs.dtype,
        )
    else:
        left_prob = left_prob.to(device=logits.device, dtype=cumprobs.dtype)
        left_prob = torch.broadcast_to(left_prob, cumprobs.shape[:-1])

    idx = torch.searchsorted(cumprobs, left_prob.unsqueeze(-1)).squeeze(-1).clamp(0, cumprobs.shape[-1] - 1)
    cumprobs = torch.cat(
        [torch.zeros(*cumprobs.shape[:-1], 1, device=logits.device, dtype=cumprobs.dtype), cumprobs],
        dim=-1,
    )

    rest_prob = left_prob - cumprobs.gather(-1, idx.unsqueeze(-1)).squeeze(-1)
    left_border = borders[idx]
    right_border = borders[idx + 1]
    return left_border + (right_border - left_border) * rest_prob / probs.gather(-1, idx.unsqueeze(-1)).squeeze(-1)


def _mode(logits: torch.Tensor, borders: torch.Tensor) -> torch.Tensor:
    density = logits.softmax(dim=-1) / _bucket_widths(borders)
    # argmax returns the lowest-index bin when several share the max density.
    mode_inds = density.argmax(dim=-1)
    bucket_means = borders[:-1] + _bucket_widths(borders) / 2
    return bucket_means[mode_inds]


def _sample(logits: torch.Tensor, borders: torch.Tensor) -> torch.Tensor:
    u = torch.rand(*logits.shape[:-1], device=logits.device)
    return _icdf(logits, borders, u)


def distribution_mean(pred: FullPrediction) -> np.ndarray:
    """Expected value of each predictive distribution, as a NumPy array.

    Args:
        pred: Full distributional output from `TabDPTRegressor.predict`.

    Returns:
        Expected target values of shape `(n_test,)`.
    """
    return _to_numpy(_mean(pred["logits"], pred["borders"]))


def distribution_median(pred: FullPrediction) -> np.ndarray:
    """Median (50th percentile) of each predictive distribution, as a NumPy array.

    Args:
        pred: Full distributional output from `TabDPTRegressor.predict`.

    Returns:
        Median target values of shape `(n_test,)`.
    """
    return _to_numpy(_icdf(pred["logits"], pred["borders"], 0.5))


def distribution_mode(pred: FullPrediction) -> np.ndarray:
    """Mode of each predictive distribution, as a NumPy array.

    Args:
        pred: Full distributional output from `TabDPTRegressor.predict`.

    Returns:
        Midpoint of the highest-density bin for each distribution, shape `(n_test,)`.
    """
    return _to_numpy(_mode(pred["logits"], pred["borders"]))


def distribution_quantiles(pred: FullPrediction, quantiles: list[float] | None = None) -> list[np.ndarray]:
    """Quantile values for each predictive distribution.

    Args:
        pred: Full distributional output from `TabDPTRegressor.predict`.
        quantiles: Probability levels in `[0, 1]`. Defaults to `[0.1, 0.2, ..., 0.9]`.

    Returns:
        One NumPy array per quantile level, each of shape `(n_test,)`.
    """
    quantiles = _DEFAULT_QUANTILES if quantiles is None else quantiles
    if not all((0 <= q <= 1) and isinstance(q, float) for q in quantiles):
        raise ValueError("All quantiles must be between 0 and 1 and floats.")
    logits, borders = pred["logits"], pred["borders"]
    return [_to_numpy(_icdf(logits, borders, q)) for q in quantiles]


def distribution_sample(pred: FullPrediction) -> np.ndarray:
    """Draw one sample from each predictive distribution, as a NumPy array.

    Args:
        pred: Full distributional output from `TabDPTRegressor.predict`.

    Returns:
        Sampled target values of shape `(n_test,)`.
    """
    return _to_numpy(_sample(pred["logits"], pred["borders"]))
