# Prepare a HuggingFace dataset for finetuning PaliGemma
# Assumptions:
# - The bounding boxes are in coco format
# - The dataset HuggingFace dataset. Follow this tutorial to create a huggingface dataset: https://huggingface.co/docs/datasets/en/image_dataset

from datasets import ClassLabel
from datasets import Dataset
import random
import pandas as pd
from IPython.display import display, HTML
from ast import literal_eval
from typing import Union, List


def coco_to_xyxy(batch):
    """
    Convert coco bounding box to xyxy format
    Args:
    - batch: a dictionary with the following keys:
        - 'resized_bbox': a list of 4 floats representing the bounding box in coco format
    Returns:
    - batch: a dictionary with the following
        - 'xyxy': a list of 4 floats representing the bounding box in xyxy format
    """
    coco_bbox = batch['resized_bbox']
    x, y, width, height = coco_bbox
    x1, y1 = x, y
    x2, y2 = x + width, y + height
    batch['xyxy'] = [x1, y1, x2, y2]
    return batch



def show_random_elements(
        dataset: Dataset, 
        num_examples:int= 10
        ):
    """
    Display a random sample of the dataset
    Args:
    - dataset: a HuggingFace dataset
    - num_examples: number of examples to display
    """
    assert num_examples <= len(dataset), "Can't pick more elements than there are in the dataset."
    picks = []
    for _ in range(num_examples):
        pick = (random.randint(0, len(dataset)-1))
        while pick in picks:
            pick = random.randint(0, len(dataset)-1)
        picks.append(pick)

    df = pd.DataFrame(dataset[picks])
    display(HTML(df.to_html()))


def string2list(string):
    """
    Convert a string to a list
    When lists are stored in csv files or HuggingFace datasets, they are stored as strings
    This function converts the string to a list
    Args:
    - string: a string
    Returns:
    - a list
    """
    return literal_eval(string)


def resize_bbox(batch):
    """
    Resize the bounding box to the size of the image
    Args:
    - batch: a dictionary with the following keys:
        - 'width': an int representing the width of the image
        - 'height': an int representing the height of the image
        - 'bbox': a list of 4 floats representing the bounding box in xyxy format
    Returns:
    - batch: a dictionary with the following
        - 'resized_bbox': a list of 4 floats representing the bounding box in xyxy format
    """
    width, height = batch['width'], batch['height']
    x, y, w, h = batch['bbox']
    x, y, w, h = int(x / width * 224), int(y / height * 224), int(w / width * 224), int(h / height * 224)
    batch['resized_bbox'] = [x, y, w, h]
    return batch


def create_list(x: Union[int, List[int]]):
    """
    Converts objects to lists for the convert to detection string function
    Args:
    - x: an object
    Returns:
    - a list
    """
    return [x]


def convert_to_detection_string(batch):
    """
    Convert the dataset to a suffix string expected by PaliGemma
    Args:
    - batch: a dictionary with the following
        - 'xyxy': a list of 4 floats representing the bounding box in xyxy format
        - 'category_id': a string representing the class of the object
        - 'width': an int representing the width of the image
        - 'height': an int representing the height of the image
    Returns:
    - batch: a dictionary with the following
        - 'suffix': a string representing the bounding box in the format expected by PaliGemma
    """
    bboxs = create_list(batch['xyxy'])
    mask_names = create_list(batch['category_id'])
    image_width = create_list(batch['width'])
    image_height = create_list(batch['height']) 
    def format_location(value, max_value):
        return f"<loc{int(round(value * 1024 / max_value)):04}>"

    detection_strings = []
    for name, bbox, w, h in zip(mask_names, bboxs, image_width, image_height):
        x1, y1, x2, y2 = bbox
        locs = [
            format_location(y1, h),
            format_location(x1, w),
            format_location(y2, h),
            format_location(x2, w),
        ]
        detection_string = "".join(locs) + f" {name}"
        detection_strings.append(detection_string)

    suffix = " ; ".join(detection_strings)
    batch['suffix'] = suffix

    return batch