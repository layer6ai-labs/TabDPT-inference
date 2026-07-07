"""Distribution over binned (bucketed) continuous targets."""

from __future__ import annotations

import torch
from torch import nn


class BarDistribution(nn.Module):
    """Distribution over fixed-width bins defined by border edges."""

    def __init__(self, borders: torch.Tensor):
        super().__init__()
        borders = borders.contiguous()
        self.register_buffer("borders", borders)
        assert (self.bucket_widths >= 0.0).all(), "borders must be sorted"

    @property
    def bucket_widths(self) -> torch.Tensor:
        return self.borders[1:] - self.borders[:-1]

    @property
    def num_bars(self) -> int:
        return len(self.borders) - 1

    def mean(self, logits: torch.Tensor) -> torch.Tensor:
        bucket_means = self.borders[:-1] + self.bucket_widths / 2
        probs = torch.softmax(logits, dim=-1)
        return probs @ bucket_means

    def median(self, logits: torch.Tensor) -> torch.Tensor:
        return self.icdf(logits, 0.5)

    def icdf(self, logits: torch.Tensor, left_prob: float) -> torch.Tensor:
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
        density = logits.softmax(dim=-1) / self.bucket_widths
        mode_inds = density.argmax(dim=-1)
        bucket_means = self.borders[:-1] + self.bucket_widths / 2
        return bucket_means[mode_inds]

    def sample(self, logits: torch.Tensor) -> torch.Tensor:
        u = torch.rand(*logits.shape[:-1], device=logits.device)
        return self.icdf(logits, u)
