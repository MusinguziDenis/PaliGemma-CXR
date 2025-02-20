import re
import torch
from matplotlib import pyplot as plt, patches
from PIL import Image
from datasets import Dataset

import random
import pandas as pd
from IPython.display import display, HTML

DETECT_RE = re.compile(
    r"(.*?)" + r"((?:<loc\d{4}>){4})\s*" + r"([^;<>]+) ?(?:; )?",
)

def get_device()->torch.device:
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def post_process(text:str)->str:
    """Remove new line characters from the text beca"""
    text = re.sub('[\n]', ' ', text)
    return text


def draw_adjusted_bbox(sample:dict)->None:
    """
    Draw the bounding box on the image
    Args:
        sample (dict): The sample containing the image and bounding box
    """
    fig, ax = plt.subplots(1)
    ax.imshow(sample['image'].resize((224, 224), resample=Image.BICUBIC))
    for name, bbox in zip(sample['mask_name'], sample['resized_bbox']):
        rect = patches.Rectangle(  # Convert (xmin, ymin, w, h)
            (bbox[0], bbox[1]),
            bbox[2] - bbox[0],
            bbox[3] - bbox[1],
            linewidth=1,
            edgecolor="r",
            facecolor="none",
        )
        ax.add_patch(rect)
        ax.axis('off')
        plt.text(
            bbox[0], bbox[1] - 10, name, color="red", fontsize=12, weight="light"
        )
    plt.tight_layout()
    plt.show()


def show_random_elements(dataset: Dataset, num_examples:int= 10):
    """Show samples from the dataset in a tabular format"""
    assert num_examples <= len(dataset), "Can't pick more elements than there are in the dataset."
    picks = []
    for _ in range(num_examples):
        pick = (random.randint(0, len(dataset)-1))
        while pick in picks:
            pick = random.randint(0, len(dataset)-1)
        picks.append(pick)

    df = pd.DataFrame(dataset[picks])
    display(HTML(df.to_html()))


def draw_wandb_inference_bbox(image:Image, objects:list)->plt.Figure:
    fig, ax = plt.subplots(1)
    ax.imshow(image)
    for obj in objects:
        bbox = obj["xyxy"]
        name = obj["name"]
        rect = patches.Rectangle(
            (bbox[0], bbox[1]),
            bbox[2] - bbox[0],
            bbox[3] - bbox[1],
            linewidth=2,
            edgecolor="r",
            facecolor="none",
        )
        ax.add_patch(rect)
        plt.text(
            bbox[0], bbox[1] - 10, name, color="red", fontsize=12, weight="bold"
        )
    
    return fig


def draw_bbox(sample):
    fig, ax = plt.subplots(1)
    ax.imshow(sample['image'])
    for name, bbox in zip(sample['mask_name'], sample['bbox']):
        rect = patches.Rectangle( # Convert the bbox to (xmin, ymin, w, h)
            (bbox[0], bbox[1]),
            bbox[2] - bbox[0],
            bbox[3] - bbox[1],
            linewidth=1,
            edgecolor="r",
            facecolor="none",
        )
        ax.add_patch(rect)
        ax.axis("off")
        plt.text(
            bbox[0], bbox[1] - 10, name, color="red", fontsize=12, weight="light"
        )
    plt.tight_layout()
    plt.show()