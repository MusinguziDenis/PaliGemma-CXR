import re
import functools

from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import flax.linen as nn
import jax
import jax.numpy as jnp

from model import processor
from utils import get_device

device = get_device()

class Segmentation(Dataset):
    def __init__(self, dataset):
        super(Segmentation).__init__()
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        image = self.dataset.iloc[idx]['image_path'][0]
        suffix = self.dataset.iloc[idx]['suffix'].lower()
        prefix = self.dataset.iloc[idx]['prefix'][0].lower()
        return {"image": image, "suffix":suffix, "prefix": prefix}
    
    def collate_fn(self, batch):
        images = [Image.open(example["image"]).resize((224, 224)) for example in batch]

        prefix = ["<image><bos> " + example['prefix'] for example in batch]
        
        suffix = [example["suffix"] + "<eos>" for example in batch]
        
        tokens = processor(images = images, text = prefix, suffix = suffix, padding = "longest", return_tensors ="pt" )

        tokens = tokens.to(torch.bfloat16).to(device)
        
        return tokens
    

def get_dataloader(dataset: Segmentation, batch_size:int = 32, shuffle:bool = True):
    dataloader = DataLoader(
        dataset, 
        batch_size = batch_size, 
        shuffle = shuffle, 
        collate_fn = dataset.collate_fn)
    
    return dataloader
