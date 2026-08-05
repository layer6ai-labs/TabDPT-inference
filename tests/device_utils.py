"""Shared test helpers: device selection and RNG seeding.

Tests prioritize GPU; CPU is used only when no GPU has enough free memory available.
"""


def seed_everything(seed: int = 42) -> None:
    """Seed RNGs for reproducible test predictions."""
    import random

    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def pick_device(min_free_gb: float = 2.0) -> str:
    """Automatically select GPU when available"""
    import torch

    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            try:
                free, _ = torch.cuda.mem_get_info(i)
            except RuntimeError:
                continue
            if free / 1e9 >= min_free_gb:
                return f"cuda:{i}"
    return "cpu"
