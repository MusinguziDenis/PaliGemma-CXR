"""Pytorch Dataset for Segmentation task."""

import pandas as pd
import torch
from model import processor
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from utils import get_device

device = get_device()

class Segmentation(Dataset):
    """Segmentation Dataset."""

    def __init__(self, dataset: pd.DataFrame) -> None:
        """Initialize the dataset.

        Args:
            dataset (pd.DataFrame): Dataset containing image paths and text.

        """
        super(Segmentation).__init__()
        self.dataset = dataset

    def __len__(self) -> int:
        """Get the number of samples in the dataset.

        Returns:
            int: Number of samples in the dataset.

        """
        return len(self.dataset)

    def __getitem__(self, idx: int) -> dict[str, str]:
        """Get a sample from the dataset.

        Args:
            idx (int): Index of the sample.

        Returns:
            dict[str, str]: Dictionary containing the image path and text.

        """
        image = self.dataset.iloc[idx]["image_path"][0]
        suffix = self.dataset.iloc[idx]["suffix"].lower()
        prefix = self.dataset.iloc[idx]["prefix"][0].lower()
        return {"image": image, "suffix":suffix, "prefix": prefix}

    def collate_fn(self,
                   batch:list[dict[str, str]],
                ) -> dict[str, torch.Tensor]:
        """Collate function for the dataset.

        Args:
            batch (list[dict[str, str]]): Batch of samples.

        Returns:
            dict[str, torch.Tensor]: Dictionary containing the images and text.

        """
        images = [Image.open(example["image"]).resize((224, 224))\
                                                        for example in batch]

        prefix = ["<image><bos> " + example["prefix"] for example in batch]

        suffix = [example["suffix"] + "<eos>" for example in batch]

        tokens = processor(
            images = images,
            text = prefix,
            suffix = suffix,
            padding = "longest",
            return_tensors ="pt")

        return tokens.to(torch.bfloat16).to(device)


def get_dataloader(
        dataset: Segmentation,
        *,
        batch_size:int = 32,
        shuffle:bool = True,
        ) -> DataLoader:
    """Get the dataloader for the dataset.

    Args:
        dataset (Segmentation): Dataset.
        batch_size (int, optional): Batch size. Defaults to 32.
        shuffle (bool, optional): Shuffle the dataset. Defaults to True.

    Returns:
        DataLoader: Dataloader for the dataset.

    """
    return DataLoader(dataset,
                      batch_size = batch_size,
                      shuffle = shuffle,
                      collate_fn = dataset.collate_fn)
