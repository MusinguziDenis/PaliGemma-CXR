"""Utils for segmentation tasks."""

import matplotlib.pyplot as plt
import torch
from PIL import Image


def get_device() -> torch.device:
    """Get the device to be used for training.

    Returns:
        torch.device: Device to be used for training.

    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def plot_mask(example:list[dict[str, torch.Tensor]]) -> None:
    """Plot the mask and bounding box for a given example.

    Args:
        example (list[dict[str, torch.Tensor]]): Example containing
        the mask and bounding box.

    """
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    image = example[0]["mask"]
    bbox = example[0]["xyxy"]
    final_image = Image.fromarray(image.astype("uint8")*255).convert("RGB")
    rect = plt.Rectangle(
                        (bbox[0], bbox[1]),
                        bbox[2]- bbox[0],
                        bbox[3]-bbox[1],
                        fill=False,
                        edgecolor="red")
    axes[0].imshow(final_image)
    axes[0].add_patch(rect)

    plt.axis("off")
    plt.show()
