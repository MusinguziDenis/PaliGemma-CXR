import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


def coco_to_xyxy(coco_bbox):
    x, y, width, height = coco_bbox
    x1, y1 = x, y
    x2, y2 = x + width, y + height
    return [x1, y1, x2, y2]

# Resize the bouding boxes to 224 x 224
def resize_bounding_boxes(bboxes, width, height):
    original_width, original_height = width[0], height[0]
    new_width, new_height = 224, 224

    scale_x = new_width / original_width
    scale_y = new_height / original_height

    resized_bboxes = []
    for bbox in bboxes:
        x_min, y_min, x_max, y_max = bbox
        new_bbox = [
            x_min * scale_x,
            y_min * scale_y,
            x_max * scale_x,
            y_max * scale_y
        ]
        resized_bboxes.append(new_bbox)

    return np.array(resized_bboxes)


def convert_to_detection_string(bboxs, mask_names, image_width, image_height):    
    def format_location(value, max_value):
        return f"<loc{int(round(value * 1024 / max_value)):04}>"

    detection_strings = []
    for name, bbox, w, h in zip(mask_names, bboxs, image_width, image_height):
        # print(bbox)
        x1, y1, x2, y2 = coco_to_xyxy(bbox)
        locs = [
            format_location(y1, h),
            format_location(x1, w),
            format_location(y2, h),
            format_location(x2, w),
        ]
        detection_string = "".join(locs) + f" {name}"
        detection_strings.append(detection_string)

    return " ; ".join(detection_strings)

def plot_reversed_image(examples):
    fig, ax = plt.subplots(2, len(examples)//2, figsize=(20, 10))
    ax = ax.flatten()
    for i, example in enumerate(examples):
        image = Image.open(f"../data/hf-dataset/{example['file_name']}").resize((224, 224))
        ax[i].imshow(image)
        ax[i].axis('off')
        for bbox in example['reversed_bbox']:
            x1, y1, x2, y2 = bbox['xyxy']
            name = bbox['name']
            rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor='red', linewidth=2)
            ax[i].add_patch(rect)
            ax[i].text(x1, y1, name, fontsize=12, color='red')
    plt.axis('off')
    plt.show()