import matplotlib.pyplot as plt
from PIL import Image

import torch

def get_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return device


def plot_mask(example):
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    image = example[0]['mask']
    bbox = example[0]['xyxy']
    final_image = Image.fromarray(image.astype('uint8')*255).convert('RGB')
    rect = plt.Rectangle((bbox[0], bbox[1]), bbox[2]- bbox[0], bbox[3]-bbox[1], fill=False, edgecolor='red')
    axes[0].imshow(image)
    axes[0].add_patch(rect)

    plt.axis('off')
    plt.show()