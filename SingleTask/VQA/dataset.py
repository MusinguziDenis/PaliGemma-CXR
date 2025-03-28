"""Pytorch Dataset for VQA task."""


import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from transformers import AutoProcessor
from utils import get_device

model_id = "google/paligemma-3b-pt-224"
processor = AutoProcessor.from_pretrained(model_id)

device = get_device()

class VQADataset(Dataset):
    """Pytorch Dataset for VQA task."""

    def __init__(self, dataframe: pd.DataFrame) -> None:
        """Initialize the dataset.

        This dataset is used to load the data for the VQA task.

        Args:
            dataframe (pd.DataFrame): Dataframe containing the data.

        """
        super().__init__()
        self.dataframe = dataframe

    def __len__(self) -> int:
        """Get the size of the dataset."""
        return len(self.dataframe)

    def __getitem__(self, idx: int) -> dict[str, str]:
        """Get the item at the given index.

        Args:
            idx (int): Index of the item to get.

        Returns:
            dict[str, str]: image path, question and answer.

        """
        example = self.dataframe.iloc[idx]
        question = example["question"]
        answer  = example["classification_diagnosis_answer"]
        image   = example["image_path"]
        return {"image_path": image, "answer":answer, "question": question}


def vqa_collate_fn(
        examples: list[dict[str, str]],
        *,
        train: bool,
        )-> torch.Tensor:
    """Batching samples.

    Args:
        examples (list[dict[str, str]]): List of examples to batch.
        train (bool): Whether the dataset is for training or not.

    Returns:
        dict: Batching samples.

    """
    image_token = "<image>"
    bos_token = "<bos>"
    eos_token = "<eos>"
    prompt = [image_token + bos_token + example["question"]\
                   for example in examples]
    if train:
        labels = [example["answer"]+ eos_token for example in examples]
    else:
        labels = None
    images     = [Image.open(example["image_path"]).convert("RGB")\
                   for example in examples]
    tokens     = processor(text=prompt, images=images, suffix=labels,
                    return_tensors="pt", padding="longest",
                    )
    return tokens.to(torch.bfloat16).to(device)
