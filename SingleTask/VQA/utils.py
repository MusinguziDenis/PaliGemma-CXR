"""Utility functions for the VQA task."""

import random
import re

import numpy as np
import torch


def make_model_deterministic(seed: int) -> None:
    """Set random seeds for reproducibility.

    Args:
        seed (int): Random seed to set.

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
    """Get the device to use for training.

    Returns:
        torch.device: The device to use for training.

    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def clean_predictions(batch: dict[str, str]) -> dict[str, str]:
    """Remove new line characters from the model predictions.

    Args:
        batch: model predictions
    Returns:
        batch: cleaned model predictions

    """
    batch["Model Prediction"] = re.sub("[\n]", "", batch["Model Prediction"])
    return batch

def clean_questions(batch: dict[str, str]) -> dict[str, str]:
    """Remove : from the questions.

    Args:
        batch: questions from the dataset
    Returns:
        batch: cleaned questions

    """
    batch["Question"] = re.sub("[:]", "", batch["question"])
    return batch
