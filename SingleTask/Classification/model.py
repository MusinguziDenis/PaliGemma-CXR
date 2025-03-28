"""Defines the model and processor for the classification task."""


from transformers import AutoProcessor, PaliGemmaForConditionalGeneration
from utils import get_device

device = get_device()

def get_processor(config: dict) -> AutoProcessor:
    """Configure the processor.

    Args:
        config: Configuration parameters.

    Returns:
        processor: Processor for the model.

    """
    return AutoProcessor.from_pretrained(config["model_id"])


def get_model(config: dict) -> PaliGemmaForConditionalGeneration:
    """Configure the model.

    Args:
        config: Configuration parameters.

    Returns:
        model: Model for the classification task.

    """
    model = PaliGemmaForConditionalGeneration.from_pretrained(
        config["model_id"], torch_dtype=config["model_dtype"]).to(device)

    for param in model.vision_tower.parameters():
        param.requires_grad = False

    for param in model.multi_modal_projector.parameters():
        param.requires_grad = False

    for param in model.language_model.parameters():
        param.requires_grad = True

    return model
