"""Utility functions for the SingleTask/Classification module."""

import random

import numpy as np
import torch


def make_model_deterministic(seed: int) -> None:
    """Set random seeds for reproducibility.

    Args:
        seed: Seed for the random number generators.

    """
    random.seed(seed)
    np.random.default_rng(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """Get the device to run the model on.

    Returns:
        device: Device to run the model on.

    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
