from transformers import PaliGemmaForConditionalGeneration, PaliGemmaProcessor
import torch

model_id = "google/paligemma-3b-pt-224"
processor = PaliGemmaProcessor.from_pretrained(model_id)


def get_model(model_id: str, torch_dtype: str, device: torch.device):
    """Prepare the model for training"""
    model = PaliGemmaForConditionalGeneration.from_pretrained(model_id, torch_dtype=torch.float16).to(device) 

    for param in model.vision_tower.parameters():
        param.requires_grad = False

    for param in model.multi_modal_projector.parameters():
        param.requires_grad = False

    for param in model.language_model.parameters():
        param.requires_grad = True

    return model