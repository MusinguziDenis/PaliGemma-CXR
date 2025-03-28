from utils import get_device, make_model_deterministic
import torch
import pandas as pd
from utils import get_device, make_model_deterministic
from model import get_model, get_processor
from train import train
from dataset import VQADataset, vqa_collate_fn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from functools import partial
import wandb
import random


device = get_device()

config = {
    'model_id': "google/paligemma-3b-pt-224",
    "num_epochs": 50,
    "num_log_samples": 10,
    "Optimzer": "AdamW",
    "scheduler" : "CosineAnnealingLR"
}

vqa_dataset = pd.read_csv('../data/kawooya_vqa_datast.csv')

train_df, valid_test_df = train_test_split(vqa_dataset, test_size=0.2)
valid_df, test_df = train_test_split(valid_test_df, test_size=0.5)

train_dataset = VQADataset(train_df)
valid_dataset = VQADataset(valid_df)
test_dataset = VQADataset(test_df)

train_dataloader = DataLoader(
    dataset = train_dataset,
    batch_size = 8,
    shuffle =True,
    collate_fn= partial(
        vqa_collate_fn,
        train = True)
)


valid_dataloader = DataLoader(
    dataset    = valid_dataset,
    batch_size = 1,
    shuffle    = False,
    collate_fn = partial(
        vqa_collate_fn,
        train  = True)
)

test_dataloader = DataLoader(
    dataset    = test_dataset,
    batch_size = 1,
    shuffle    = False,
    collate_fn = partial(
        vqa_collate_fn,
        train  = False)
)

log_indices = random.sample(range(len(valid_dataset)), config["num_log_samples"])

model = get_model(config)
processor = get_processor(config)

wandb.login()

# Initialize Weights & Biases run
run = wandb.init(
    name= "PaliGemma Classification FT",
    project= "PaliGemma-CXR",
    reinit=True,
    config=config
)

optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_dataloader) * config['num_epochs'])

train_loss, valid_loss = train(config, model, config['num_epochs'], train_dataloader, valid_dataloader, scheduler, device, optimizer,
                               processor, log_indices)