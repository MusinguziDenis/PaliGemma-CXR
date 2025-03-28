import evaluate
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn

from typing import Dict

from transformers import AutoProcessor

bleu_score = evaluate.load('bleu')
meteor_score = evaluate.load('meteor')
rouge_score = evaluate.load('rouge')

def get_model_preds(
        example: pd.Series, 
        model: nn.Module, 
        processor: AutoProcessor, 
        device: torch.device
        ) -> str:
    """Generate predictions from the model for the given example"""
    question = "Generate a medical report for the X-ray image provided"
    image    = Image.open(example['image_path']).convert("RGB")
    question = "<image>" + "<bos>" + question
    inputs   = processor(text= question, images = image, return_tensors = "pt", padding="longest").to(torch.bfloat16).to(device)
    outputs  = model.generate(**inputs, max_new_tokens= 50)[0]
    outputs  = processor.decode(outputs[inputs["input_ids"].shape[1]:], skip_special_tokens = True)
    return outputs

def compute_metrics(examples: pd.DataFrame) -> Dict[str, float]:
    """Compute the BLEU, METEOR and ROUGE scores for the model predictions"""
    bleu    = bleu_score.compute(references = [examples["report"].lower()], predictions=[examples["Model Predictions"].lower()])
    meteor  = meteor_score.compute(references = [examples["report"].lower()], predictions=[examples["Model Predictions"].lower()])
    rouge   = rouge_score.compute(references = [examples["report"].lower()], predictions=[examples["Model Predictions"].lower()])
    return {"BLEU": bleu["bleu"], "METEOR": meteor["meteor"], "ROUGE": rouge["rouge1"]}