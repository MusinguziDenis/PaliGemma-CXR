import random
import wandb

import numpy as np

import torch
from datasets import load_dataset

from pg_datasets import coco_to_xyxy, show_random_elements, string2list, resize_bbox, coco_to_xyxy, convert_to_detection_string
from pg_datasets import ObjectDetectionDataset, get_dataloader

from model import processor, get_model
from train import train, evaluate, extract_objects
from utils import show_random_elements, get_device, post_process, draw_adjusted_bbox, draw_bbox
from metrics import create_eval_object, average_precisions_per_class

# Log into huggingface through the CLI
# !huggingface-cli login

train_config = dict(
    num_epochs = 1,
    eval_interval = 45,  # Evaluate every 'eval_interval' steps
    loss_scaling_factor = 1000.0,  # Variable to scale the loss by a certain amount
    save_dir = '../models',
    accumulation_steps = 8,  # Accumulate gradients over this many steps
    optimizer = "AdamW",
    num_log_samples = 10,
    learning_rate = 3e-5,
    model_id = 'google/paligemma-3b-pt-224',
    model_dtype = torch.bfloat16,
    model_revision = "bfloat16"
)

# Load the dataset
dataset_id = 'dmusingu/lacuna-object-detection-chest-X-ray-dataset' # Dataset is private, will make public soon
dataset = load_dataset(dataset_id, split='train')

# Split the dataset into training and testing
splits = dataset.train_test_split(test_size=0.2)
train_dataset = splits['train']
test_dataset = splits['test']

train_data = ObjectDetectionDataset(train_dataset)
test_data = ObjectDetectionDataset(test_dataset)

# Create the dataloaders
train_dataloader = get_dataloader(train_data, batch_size=4)
test_dataloader = get_dataloader(test_data, batch_size=1)

# Get the indices to log
log_indices = random.sample(range(len(test_dataloader)), 5)


# Create the model and tokenizer
model = get_model(train_config['model_id'])
tokenizer = processor.tokenizer

# Create the optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=train_config['learning_rate'])

# set the device
device = get_device()

# Initialize the wandb run
wandb.login()

run = wandb.init(project="Multimodal TB Project", name="PaliGemma Object Detection 1 Epochs", config=train_config, reinit=True)

# Train the model 
train(model, train_dataloader, test_dataloader,train_config['num_epochs'], device,train_config,optimizer,log_indices,tokenizer)

# Evaluate the model
stats = []
for batch in test_dataloader:
    stats.append(create_eval_object(model, batch))
 
concatenated_stats = [np.concatenate(items, 0) for items in zip(*stats)]
average_precisions, unique_classes = average_precisions_per_class(
    *concatenated_stats
)
mAP_scores = np.mean(average_precisions, axis=0)

print(f"mAP@50", mAP_scores[0])
print(f"mAP@75", mAP_scores[1])

