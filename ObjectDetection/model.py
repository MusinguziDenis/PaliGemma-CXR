import torch.nn as nn
from transformers import PaliGemmaProcessor, PaliGemmaForConditionalGeneration

model_id = "google/paligemma-3b-pt-224"
processor = PaliGemmaProcessor.from_pretrained(model_id)

def get_model(model_id:str = model_id)->nn.Module:
    """
    Initialize the model and freeze the vision tower and multi-modal projector
    Args:
        model_id (str): The model id to initialize the model
    Returns:
        model (nn.Module): The model with the vision tower and multi-modal projector frozen
    """
    model = PaliGemmaForConditionalGeneration.from_pretrained(model_id)
    for param in model.vision_tower.parameters():
        param.requires_grad = False

    for param in model.multi_modal_projector.parameters():
        param.requires_grad = False

    for param in model.language_model.parameters():
        param.requires_grad = True

    return model