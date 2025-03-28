"""Dataset and collate function for the classification task."""


import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from transformers import AutoProcessor
from utils import get_device

model_id = "google/paligemma-3b-pt-224"
processor = AutoProcessor.from_pretrained(model_id)

device = get_device()

class ClassificationDataset(Dataset):
    """PyTorch Dataset for the classification task."""

    def __init__(self, dataframe: pd.DataFrame) -> None:
        """Initialize the dataset.

        Args:
            dataframe: Dataframe containing the data.

        """
        super().__init__()
        self.dataframe = dataframe

    def __len__(self) -> int:
        """Return the number of examples in the dataset."""
        return len(self.dataframe)

    def __getitem__(self, idx: int) -> dict[str, str]:
        """Return the example at the given index.

        Args:
            idx: Index of the example.

        Returns:
            example: Example at the given index.

        """
        example = self.dataframe.iloc[idx]
        answer  = example["classification_diagnosis_answer"]
        image   = example["image_path"]
        return {"image_path": image, "answer": answer}


def classification_collate_fn(
        examples: list[dict[str, str]],
        *,
        train: bool,
    ) -> dict[str, torch.Tensor]:
    """Collate function used by the dataloader to batch the data.

    Args:
        examples: List of examples.
        train: Whether the data is used for training or not.

    Returns:
        tokens: Batched tokens.

    """
    question     = "What is the diagnosis of the X-ray image?"
    image_token  = "<image>"
    bos_token  = "<bos>"
    eos_token = "<eos>"
    prompt     = [image_token + bos_token + question for _ in examples]
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
