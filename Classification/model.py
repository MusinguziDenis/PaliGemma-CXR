import torch 
from transformers import PaliGemmaForConditionalGeneration, AutoProcessor
from typing import Dict
from utils import get_device

device = get_device()

def get_processor(config: Dict):
    """Defines the model process"""
    processor = AutoProcessor.from_pretrained(config['model_id'])
    return processor


def get_model(config: Dict):
    """Defines the model and sets trainable parameters"""
    model = PaliGemmaForConditionalGeneration.from_pretrained(config['model_id'], torch_dtype=config['model_dtype']).to(device)

    for param in model.vision_tower.parameters():
        param.requires_grad = False

    for param in model.multi_modal_projector.parameters():
        param.requires_grad = False

    for param in model.language_model.parameters():
        param.requires_grad = True

    return model


