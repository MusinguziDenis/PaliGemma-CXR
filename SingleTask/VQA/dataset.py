from torch.utils.data import Dataset, DataLoader
from typing import List, Dict
from PIL import Image
from functools import partial
import pandas as pd
from transformers import AutoProcessor
import torch
from utils import get_device

model_id = "google/paligemma-3b-pt-224"
processor = AutoProcessor.from_pretrained(model_id)

device = get_device()

class VQADataset(Dataset):
    def __init__(self, dataframe: pd.DataFrame):
        super().__init__()
        self.dataframe = dataframe

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx: int):
        example = self.dataframe.iloc[idx]
        question = example['question']
        answer  = example['classification_diagnosis_answer']
        image   = example['image_path']
        return {'image_path': image, "answer":answer, 'question': question}
    

def vqa_collate_fn(examples: List[Dict[str, str]], train: bool):
    """Collate function used by the dataloader to batch the data"""
    image_token  = "<image>"
    bos_token  = "<bos>"
    eos_token = "<eos>"
    prompt     = [image_token + bos_token + example['question'] for example in examples]
    if train:
        labels = [example['answer']+ eos_token for example in examples]
    else:
        labels = None
    images     = [Image.open(example['image_path']).convert("RGB") for example in examples]
    tokens     = processor(text=prompt, images=images, suffix=labels,
                    return_tensors="pt", padding="longest",
                    )

    tokens = tokens.to(torch.bfloat16).to(device)
    return tokens
