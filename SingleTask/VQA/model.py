"""Configure the model and processor for the VQA task."""

from transformers import AutoProcessor, PaliGemmaForConditionalGeneration
from utils import get_device

device = get_device()

def get_processor(config: dict) -> AutoProcessor:
    """Load the pre-processor.

    Args:
        config (dict): Configuration dictionary containing the model ID.

    Returns:
        AutoProcessor: The processor for the model.

    """
    return AutoProcessor.from_pretrained(config["model_id"])


def get_model(config: dict) -> PaliGemmaForConditionalGeneration:
    """Load and freeze the lm and the conneector.

    Args:
        config (dict): Configuration containing the model ID and data type.

    Returns:
        PaliGemmaForConditionalGeneration: The model for the VQA task.

    """
    model = PaliGemmaForConditionalGeneration.from_pretrained(
                                            config["model_id"],
                                            torch_dtype=config["model_dtype"],
                                            ).to(device)
    for param in model.vision_tower.parameters():
        param.requires_grad = False

    for param in model.multi_modal_projector.parameters():
        param.requires_grad = False

    for param in model.language_model.parameters():
        param.requires_grad = True

    return model
