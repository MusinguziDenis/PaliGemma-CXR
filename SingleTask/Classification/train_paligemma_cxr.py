"""Train a model model on the PaliGemma CXR dataset."""

import random
from functools import partial

import pandas as pd
import torch
import wandb
from dataset import ClassificationDataset, classification_collate_fn
from model import get_model, get_processor
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from train import train
from utils import get_device

device = get_device()

config = {
    "model_id": "google/paligemma-3b-pt-224",
    "num_epochs": 50,
    "num_log_samples": 10,
    "Optimzer": "AdamW",
    "scheduler" : "CosineAnnealingLR",
}

classification_dataset = pd.read_csv("data/image_classification_dataset.csv")

train_df, valid_test_df = train_test_split(
                                            classification_dataset,
                                            test_size=0.2)
valid_df, test_df = train_test_split(valid_test_df, test_size=0.5)

train_dataset = ClassificationDataset(train_df)
valid_dataset = ClassificationDataset(valid_df)
test_dataset = ClassificationDataset(test_df)

train_dataloader = DataLoader(
    dataset = train_dataset,
    batch_size = 8,
    shuffle =True,
    collate_fn= partial(
        classification_collate_fn,
        train = True),
)

valid_dataloader = DataLoader(
    dataset    = valid_dataset,
    batch_size = 1,
    shuffle    = False,
    collate_fn = partial(
        classification_collate_fn,
        train  = True),
)

test_dataloader = DataLoader(
    dataset    = test_dataset,
    batch_size = 1,
    shuffle    = False,
    collate_fn = partial(
        classification_collate_fn,
        train  = False),
)

log_indices = random.sample(range(len(valid_dataset)),
                            config["num_log_samples"])

model = get_model(config)
processor = get_processor(config)

wandb.login()

# Initialize Weights & Biases run
run = wandb.init(
    name= "PaliGemma Classification FT",
    project= "PaliGemma-CXR",
    reinit=True,
    config=config,
)

optimizer = torch.optim.AdamW(model.parameters(),
                            lr= config["lr"])
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                        T_max=len(train_dataloader) * config["num_epochs"])

train_loss, valid_loss = train(config,
                                model,
                                config["num_epochs"],
                                train_dataloader,
                                valid_dataloader,
                                scheduler,
                                device,
                                optimizer,
                               processor,
                               log_indices)
