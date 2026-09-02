"""Distribution over binned (bucketed) continuous targets."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

_DEFAULT_QUANTILES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


class BarDistribution(nn.Module):
    """Piecewise-uniform distribution over fixed-width bins.

    Each bin is defined by consecutive entries in `borders`. Model logits over
    bins are turned into a categorical distribution via softmax; within each bin
    the density is uniform, so the overall distribution is a histogram with
    continuous support on `[borders[0], borders[-1]]`.
    """

    def __init__(self, borders: torch.Tensor):
        """Build a bar distribution from bin edges.

        Args:
            borders: Strictly increasing bin edges of shape `(num_bars + 1,)`.
                Defines `num_bars` contiguous intervals
                `[borders[i], borders[i + 1])`.
        """
        super().__init__()
        borders = borders.contiguous()
        self.register_buffer("borders", borders)
        assert (self.bucket_widths > 0.0).all(), "borders must be sorted"

    @property
    def bucket_widths(self) -> torch.Tensor:
        """Width of each bin, shape `(num_bars,)`."""
        return self.borders[1:] - self.borders[:-1]

    @property
    def num_bars(self) -> int:
        """Number of bins."""
        return len(self.borders) - 1

    def mean(self, logits: torch.Tensor) -> torch.Tensor:
        """Expected value of the distribution.

        Args:
            logits: Unnormalized log-probabilities of shape `(*leading, num_bars)`,
                where `*leading` indexes independent distributions (e.g. test
                points, or eval positions with optional model batching).

        Returns:
            Expected target values of shape `(*leading,)`.
        """
        bucket_means = self.borders[:-1] + self.bucket_widths / 2
        probs = torch.softmax(logits, dim=-1)
        return probs @ bucket_means

    def median(self, logits: torch.Tensor) -> torch.Tensor:
        """Median of the distribution (50th percentile).

        Args:
            logits: Unnormalized log-probabilities of shape `(*leading, num_bars)`,
                where `*leading` indexes independent distributions (e.g. test
                points, or eval positions with optional model batching).

        Returns:
            Median target values of shape `(*leading,)`.
        """
        return self.icdf(logits, 0.5)

    def icdf(
        self,
        logits: torch.Tensor,
        left_prob: float | torch.Tensor,
    ) -> torch.Tensor:
        """Inverse CDF (quantile function).

        Args:
            logits: Unnormalized log-probabilities of shape `(*leading, num_bars)`,
                where `*leading` indexes independent distributions (e.g. test
                points, or eval positions with optional model batching).
            left_prob: Probability level(s) in `[0, 1]`. A scalar applies the
                same level to every distribution. A tensor must be broadcastable
                to `logits.shape[:-1]` (for example shape `(*leading,)`) so
                each distribution can use its own level.

        Returns:
            Target values at the requested quantile(s), shape `(*leading,)`.
        """
        probs = logits.softmax(dim=-1)
        cumprobs = torch.cumsum(probs, dim=-1)

        # Normalize left_prob to shape (*leading,) then (*leading, 1) for searchsorted.
        # Avoid `tensor(N,) * ones(N, 1)`, which broadcasts to (N, N) in PyTorch.
        if not torch.is_tensor(left_prob):
            left_prob = torch.full(
                cumprobs.shape[:-1],
                float(left_prob),
                device=logits.device,
                dtype=cumprobs.dtype,
            )
        else:
            left_prob = left_prob.to(device=logits.device, dtype=cumprobs.dtype)
            left_prob = torch.broadcast_to(left_prob, cumprobs.shape[:-1])

        idx = (
            torch.searchsorted(cumprobs, left_prob.unsqueeze(-1))
            .squeeze(-1)
            .clamp(0, cumprobs.shape[-1] - 1)
        )
        cumprobs = torch.cat(
            [torch.zeros(*cumprobs.shape[:-1], 1, device=logits.device, dtype=cumprobs.dtype), cumprobs],
            dim=-1,
        )

        rest_prob = left_prob - cumprobs.gather(-1, idx.unsqueeze(-1)).squeeze(-1)
        left_border = self.borders[idx]
        right_border = self.borders[idx + 1]
        return left_border + (right_border - left_border) * rest_prob / probs.gather(
            -1,
            idx.unsqueeze(-1),
        ).squeeze(-1)

    def mode(self, logits: torch.Tensor) -> torch.Tensor:
        """Mode of the distribution (bin with highest probability density).

        Args:
            logits: Unnormalized log-probabilities of shape `(*leading, num_bars)`,
                where `*leading` indexes independent distributions (e.g. test
                points, or eval positions with optional model batching).

        Returns:
            Midpoint of the highest-density bin for each distribution,
            shape `(*leading,)`.
        """
        density = logits.softmax(dim=-1) / self.bucket_widths
        # argmax returns the lowest-index bin when several share the max density.
        mode_inds = density.argmax(dim=-1)
        bucket_means = self.borders[:-1] + self.bucket_widths / 2
        return bucket_means[mode_inds]

    def sample(self, logits: torch.Tensor) -> torch.Tensor:
        """Draw one sample from each indexed distribution.

        For every leading dimension (all axes except the final `num_bars`
        axis), draws a single continuous target value from the corresponding
        predictive distribution. At inference this is often `(n_test, num_bars)`
        (one distribution per test point), but any `*leading` layout is
        supported.

        Args:
            logits: Unnormalized log-probabilities of shape `(*leading, num_bars)`,
                where `*leading` indexes independent distributions (e.g. test
                points, or eval positions with optional model batching).

        Returns:
            Sampled target values of shape `(*leading,)`.
        """
        # One uniform draw per distribution; shape[:-1] covers all leading axes.
        # Only the last axis holds per-bin logits.
        u = torch.rand(*logits.shape[:-1], device=logits.device)
        return self.icdf(logits, u)


def _to_numpy(values: torch.Tensor) -> np.ndarray:
    return values.detach().cpu().numpy()


def distribution_mean(logits: torch.Tensor, criterion: BarDistribution) -> np.ndarray:
    """Expected value of each predictive distribution, as a NumPy array."""
    return _to_numpy(criterion.mean(logits))


def distribution_median(logits: torch.Tensor, criterion: BarDistribution) -> np.ndarray:
    """Median (50th percentile) of each predictive distribution, as a NumPy array."""
    return _to_numpy(criterion.median(logits))


def distribution_mode(logits: torch.Tensor, criterion: BarDistribution) -> np.ndarray:
    """Mode of each predictive distribution, as a NumPy array."""
    return _to_numpy(criterion.mode(logits))


def distribution_quantiles(
    logits: torch.Tensor,
    criterion: BarDistribution,
    quantiles: list[float] | None = None,
) -> list[np.ndarray]:
    """Quantile values for each predictive distribution.

    Args:
        logits: Unnormalized log-probabilities of shape `(*leading, num_bars)`.
        criterion: Bar distribution whose borders define the support.
        quantiles: Probability levels in `[0, 1]`. Defaults to
            `[0.1, 0.2, ..., 0.9]`.

    Returns:
        One NumPy array per quantile level, each of shape `(*leading,)`.
    """
    quantiles = _DEFAULT_QUANTILES if quantiles is None else quantiles
    if not all((0 <= q <= 1) and isinstance(q, float) for q in quantiles):
        raise ValueError("All quantiles must be between 0 and 1 and floats.")
    return [_to_numpy(criterion.icdf(logits, q)) for q in quantiles]
