import torch
import random
import numpy as np
import re

def make_model_deterministic(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def get_device():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return device

def clean_predictions(batch):
    """
    The model makes predictions with \n which prevents the regex pattern
    detecting the actual response from working
    args:
        batch: model predictions
    """
    batch['Model Prediction'] = re.sub('[\n]', '', batch["Model Prediction"])
    return batch

def clean_questions(batch):
    """
    The questions have : in them
    args:
        batch: questions from the dataset
    """
    batch['Question'] = re.sub('[:]', '', batch["question"])
    return batch

