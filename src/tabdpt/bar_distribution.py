"""Distribution over binned (bucketed) continuous targets."""

from __future__ import annotations

import torch
from torch import nn


class BarDistribution(nn.Module):
    """Piecewise-uniform distribution over fixed-width bins.

    Each bin is defined by consecutive entries in ``borders``. Model logits over
    bins are turned into a categorical distribution via softmax; within each bin
    the density is uniform, so the overall distribution is a histogram with
    continuous support on ``[borders[0], borders[-1]]``.
    """

    def __init__(self, borders: torch.Tensor):
        """Build a bar distribution from bin edges.

        Args:
            borders: Strictly increasing bin edges of shape ``(num_bars + 1,)``.
                Defines ``num_bars`` contiguous intervals
                ``[borders[i], borders[i + 1])``.
        """
        super().__init__()
        borders = borders.contiguous()
        self.register_buffer("borders", borders)
        assert (self.bucket_widths > 0.0).all(), "borders must be sorted"

    @property
    def bucket_widths(self) -> torch.Tensor:
        """Width of each bin, shape ``(num_bars,)``."""
        return self.borders[1:] - self.borders[:-1]

    @property
    def num_bars(self) -> int:
        """Number of bins."""
        return len(self.borders) - 1

    def mean(self, logits: torch.Tensor) -> torch.Tensor:
        """Expected value of the distribution.

        Args:
            logits: Unnormalized log-probabilities of shape ``(*batch, num_bars)``.

        Returns:
            Expected target values of shape ``(*batch,)``.
        """
        bucket_means = self.borders[:-1] + self.bucket_widths / 2
        probs = torch.softmax(logits, dim=-1)
        return probs @ bucket_means

    def median(self, logits: torch.Tensor) -> torch.Tensor:
        """Median of the distribution (50th percentile).

        Args:
            logits: Unnormalized log-probabilities of shape ``(*batch, num_bars)``.

        Returns:
            Median target values of shape ``(*batch,)``.
        """
        return self.icdf(logits, 0.5)

    def icdf(
        self,
        logits: torch.Tensor,
        left_prob: float | torch.Tensor,
    ) -> torch.Tensor:
        """Inverse CDF (quantile function).

        Args:
            logits: Unnormalized log-probabilities of shape ``(*batch, num_bars)``.
            left_prob: Probability level(s) in ``[0, 1]``. A scalar applies the
                same level to every batch element. A tensor must be broadcastable
                to ``logits.shape[:-1]`` (for example shape ``(*batch,)``) so
                each distribution can use its own level.

        Returns:
            Target values at the requested quantile(s), shape ``(*batch,)``.
        """
        probs = logits.softmax(dim=-1)
        cumprobs = torch.cumsum(probs, dim=-1)
        idx = (
            torch.searchsorted(
                cumprobs,
                left_prob * torch.ones(*cumprobs.shape[:-1], 1, device=logits.device),
            )
            .squeeze(-1)
            .clamp(0, cumprobs.shape[-1] - 1)
        )
        cumprobs = torch.cat(
            [torch.zeros(*cumprobs.shape[:-1], 1, device=logits.device), cumprobs],
            dim=-1,
        )

        rest_prob = left_prob - cumprobs.gather(-1, idx[..., None]).squeeze(-1)
        left_border = self.borders[idx]
        right_border = self.borders[idx + 1]
        return left_border + (right_border - left_border) * rest_prob / probs.gather(
            -1,
            idx[..., None],
        ).squeeze(-1)

    def mode(self, logits: torch.Tensor) -> torch.Tensor:
        """Mode of the distribution (bin with highest probability density).

        Args:
            logits: Unnormalized log-probabilities of shape ``(*batch, num_bars)``.

        Returns:
            Midpoint of the highest-density bin for each batch element,
            shape ``(*batch,)``.
        """
        density = logits.softmax(dim=-1) / self.bucket_widths
        # argmax returns the lowest-index bin when several share the max density.
        mode_inds = density.argmax(dim=-1)
        bucket_means = self.borders[:-1] + self.bucket_widths / 2
        return bucket_means[mode_inds]

    def sample(self, logits: torch.Tensor) -> torch.Tensor:
        """Draw one sample from each batched distribution.

        For every leading batch dimension (all axes except the final ``num_bars``
        axis), draws a single continuous target value from the corresponding
        predictive distribution. Logits are typically
        ``(n_test, num_bars)``, so this returns one sample per test point with
        shape ``(n_test,)``.

        Args:
            logits: Unnormalized log-probabilities of shape ``(*batch, num_bars)``.

        Returns:
            Sampled target values of shape ``(*batch,)``.
        """
        # One uniform draw per distribution of shape (*batch,)
        # Use shape[:-1] to batch over all leadingaxes; 
        # only the last axis holds the per-bin logits.
        u = torch.rand(*logits.shape[:-1], device=logits.device)
        return self.icdf(logits, u)
