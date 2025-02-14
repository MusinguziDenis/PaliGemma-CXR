from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import torch
import os

from typing import Dict, Union

from PIL import Image
from torchvision.transforms import functional as F

from transformers import AutoProcessor

from functools import partial

class RRGDataset(Dataset):
    def __init__(
            self, 
            dataframe: pd.DataFrame, 
            processor: AutoProcessor,
            device: torch.device,
            max_length: int, 
            train: bool=True,
        ) -> None:
        self.dataframe = dataframe
        self.max_length = max_length

        self.processor = processor
        self.device = device
        self.train = train

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx: int):
        row    = self.dataframe.iloc[idx]
        image  = Image.open(row['image_path']).convert('RGB')
        prompt = "Generate a medical report for the X-ray image provided"
        text   = row['shorter_report']
        
        sample = {"image": image, "prompt": prompt, "report": text}
        return sample
    
    def collate_fn(
            self,
            examples: list[Dict[str, Union[str, Image.Image]]],
            train:bool=True
        )-> torch.Tensor:
    

        image_string = "<image>"
        bos_string = "<bos>"
        eos_string = "<eos>"

        texts  = [image_string + bos_string + example["prompt"] for example in examples]
        if train:
            labels = [example['report'] + eos_string for example in examples]
        else:
            labels = None
        images = [example["image"] for example in examples]
        tokens = self.processor(text=texts, images=images, suffix=labels, return_tensors="pt", padding=True)
            
        tokens = tokens.to(torch.bfloat16).to(self.device)
        return tokens


def get_dataloader(
        dataset: RRGDataset,
        batch_size: int,
        train: bool=True,
        num_workers: int=4,
    ):
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train,
        num_workers=num_workers,
        collate_fn=partial(
            dataset.collate_fn,
            train=train,
        )
    )

    return dataloader
