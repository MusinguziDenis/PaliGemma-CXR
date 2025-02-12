## Finetune PaliGemma for Object Detection an X-ray Dataset

Dataset
* [dmusingu/object-detection-chest-x-ray](https://huggingface.co/datasets/dmusingu/object-detection-chest-x-ray)

PaliGemma expects object detection data with the followinng prefix (prompt) format

```
detect {object}; ..{object};
```
The objects include all the object encountered in the training dataset.

PaliGemma expects the suffix (label) to be formatted as follows.
The suffix should have 4 location tokens followed by the name of the object. In cases where there are multiple objects, their labels are separated by ;
```
<loc0001><loc0002><loc0003><loc0004> {object}; <loc0001><loc0002><loc0003><loc0004> {object}; 
```
In the notebook, we illustrate how to convert a HuggingFace object detectiion dataset into PaliGemma object detection format.    
**Procedure**
* Load the dataset from HuggingFace
* Resize the bounding boxes to size that matches the expected image input shape of PaliGemma which is 224 x 224.
* Convert the bounding box format to `x_min, y_min, x_max, y_max` from the coco format `x_min, y_min, width, height`.
* Convert the bounding boxes and labels to the suffix format expected by PaliGemma
* Create the prefix token expect by PaliGemma.

## TODO
* Create Pytorch Dataset and DataLoader
* Write Training Code