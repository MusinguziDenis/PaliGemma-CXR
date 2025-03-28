import numpy as np  
from typing import Tuple
from train import extract_objects

from model import processor


def mask_iou_batch_split(
    masks_true: np.ndarray, masks_detection: np.ndarray
) -> np.ndarray:
    """
    Internal function.
    Compute Intersection over Union (IoU) of two sets of masks -
        `masks_true` and `masks_detection`.

    Args:
        masks_true (np.ndarray): 3D `np.ndarray` representing ground-truth masks.
        masks_detection (np.ndarray): 3D `np.ndarray` representing detection masks.

    Returns:
        np.ndarray: Pairwise IoU of masks from `masks_true` and `masks_detection`.
    """
    intersection_area = np.logical_and(masks_true[:, None], masks_detection).sum(
        axis=(2, 3)
    )

    masks_true_area = masks_true.sum(axis=(1, 2))
    masks_detection_area = masks_detection.sum(axis=(1, 2))
    union_area = masks_true_area[:, None] + masks_detection_area - intersection_area

    return np.divide(
        intersection_area,
        union_area,
        out=np.zeros_like(intersection_area, dtype=float),
        where=union_area != 0,
    )


def mask_iou_batch(
    masks_true: np.ndarray,
    masks_detection: np.ndarray,
    memory_limit: int = 1024 * 5,
) -> np.ndarray:
    """
    Compute Intersection over Union (IoU) of two sets of masks -
        `masks_true` and `masks_detection`.

    Args:
        masks_true (np.ndarray): 3D `np.ndarray` representing ground-truth masks.
        masks_detection (np.ndarray): 3D `np.ndarray` representing detection masks.
        memory_limit (int): memory limit in MB, default is 1024 * 5 MB (5GB).

    Returns:
        np.ndarray: Pairwise IoU of masks from `masks_true` and `masks_detection`.
    """
    memory = (
        masks_true.shape[0]
        * masks_true.shape[1]
        * masks_true.shape[2]
        * masks_detection.shape[0]
        / 1024
        / 1024
    )
    if memory <= memory_limit:
        return mask_iou_batch_split(masks_true, masks_detection)

    ious = []
    step = max(
        memory_limit
        * 1024
        * 1024
        // (
            masks_detection.shape[0]
            * masks_detection.shape[1]
            * masks_detection.shape[2]
        ),
        1,
    )
    for i in range(0, masks_true.shape[0], step):
        ious.append(mask_iou_batch_split(masks_true[i : i + step], masks_detection))

    return np.vstack(ious)

def match_detection_batch(
        predictions_classes: np.ndarray, # shape = (num_predictions,)
        target_classes: np.ndarray, # shape = (num_targets,)
        iou: np.ndarray, # shape = (num_predictions, num_targets)
        iou_thresholds: np.ndarray, # shape = (num_iou_levels,)
    ) -> np.ndarray:
        num_predictions, num_iou_levels = (
            predictions_classes.shape[0],
            iou_thresholds.shape[0],
        )
        correct = np.zeros((num_predictions, num_iou_levels), dtype=bool)
        correct_class = target_classes[:, None] == predictions_classes

        for i, iou_level in enumerate(iou_thresholds):
            matched_indices = np.where((iou >= iou_level) & correct_class)

            if matched_indices[0].shape[0]:
                combined_indices = np.stack(matched_indices, axis=1)
                iou_values = iou[matched_indices][:, None]
                matches = np.hstack([combined_indices, iou_values])

                if matched_indices[0].shape[0] > 1:
                    matches = matches[matches[:, 2].argsort()[::-1]]
                    matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                    matches = matches[np.unique(matches[:, 0], return_index=True)[1]]

                correct[matches[:, 1].astype(int), i] = True

        return correct


def average_precisions_per_class(
        matches: np.ndarray,
        prediction_confidence: np.ndarray,
        prediction_class_ids: np.ndarray,
        true_class_ids: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute the average precision, given the recall and precision curves.
        Source: https://github.com/rafaelpadilla/Object-Detection-Metrics.

        Args:
            matches (np.ndarray): True positives.
            prediction_confidence (np.ndarray): Objectness value from 0-1.
            prediction_class_ids (np.ndarray): Predicted object classes.
            true_class_ids (np.ndarray): True object classes.
            eps (float, optional): Small value to prevent division by zero.

        Returns:
            (Tuple[np.ndarray, np.ndarray]): Average precision for different
                IoU levels, and an array of class IDs that were matched.
        """
        eps = 1e-16

        sorted_indices = np.argsort(-prediction_confidence)
        matches = matches[sorted_indices]
        prediction_class_ids = prediction_class_ids[sorted_indices]

        unique_classes, class_counts = np.unique(true_class_ids, return_counts=True)
        num_classes = unique_classes.shape[0]

        average_precisions = np.zeros((num_classes, matches.shape[1]))

        for class_idx, class_id in enumerate(unique_classes):
            is_class = prediction_class_ids == class_id
            total_true = class_counts[class_idx]
            total_prediction = is_class.sum()

            if total_prediction == 0 or total_true == 0:
                continue

            false_positives = (1 - matches[is_class]).cumsum(0)
            true_positives = matches[is_class].cumsum(0)
            false_negatives = total_true - true_positives

            recall = true_positives / (true_positives + false_negatives + eps)
            precision = true_positives / (true_positives + false_positives)

            for iou_level_idx in range(matches.shape[1]):
                average_precisions[class_idx, iou_level_idx] = (
                    compute_average_precision(
                        recall[:, iou_level_idx], precision[:, iou_level_idx]
                    )
                )

        return average_precisions, unique_classes


def compute_average_precision(recall: np.ndarray, precision: np.ndarray) -> float:
        """
        Compute the average precision using 101-point interpolation (COCO), given
            the recall and precision curves.

        Args:
            recall (np.ndarray): The recall curve.
            precision (np.ndarray): The precision curve.

        Returns:
            (float): Average precision.
        """
        if len(recall) == 0 and len(precision) == 0:
            return 0.0

        recall_levels = np.linspace(0, 1, 101)
        precision_levels = np.zeros_like(recall_levels)
        for r, p in zip(recall[::-1], precision[::-1]):
            precision_levels[recall_levels <= r] = p

        average_precision = (1 / 101 * precision_levels).sum()
        return average_precision


def box_iou_batch(boxes_true: np.ndarray, boxes_detection: np.ndarray) -> np.ndarray:
    """
    Compute Intersection over Union (IoU) of two sets of bounding boxes -
        `boxes_true` and `boxes_detection`. Both sets
        of boxes are expected to be in `(x_min, y_min, x_max, y_max)` format.

    Args:
        boxes_true (np.ndarray): 2D `np.ndarray` representing ground-truth boxes.
            `shape = (N, 4)` where `N` is number of true objects.
        boxes_detection (np.ndarray): 2D `np.ndarray` representing detection boxes.
            `shape = (M, 4)` where `M` is number of detected objects.

    Returns:
        np.ndarray: Pairwise IoU of boxes from `boxes_true` and `boxes_detection`.
            `shape = (N, M)` where `N` is number of true objects and
            `M` is number of detected objects.
    """

    def box_area(box):
        return (box[2] - box[0]) * (box[3] - box[1])

    area_true = box_area(boxes_true.T)
    area_detection = box_area(boxes_detection.T)

    top_left = np.maximum(boxes_true[:, None, :2], boxes_detection[:, :2])
    bottom_right = np.minimum(boxes_true[:, None, 2:], boxes_detection[:, 2:])

    area_inter = np.prod(np.clip(bottom_right - top_left, a_min=0, a_max=None), 2)
    ious = area_inter / (area_true[:, None] + area_detection - area_inter)
    ious = np.nan_to_num(ious)
    return ious

def convert_object_eval(objects: list):
    """
    Convert the object to the format for evaluation
    """
    bbox = []
    classes = []
    for object in objects:
        bbox.append(object['xyxy'])
        classes.append(label2idx.get(object['name'].lower(), 'wrong'))
    return np.array(bbox), np.array(classes)


def create_eval_object(model, sample):
    thresholds = np.array([0.5, 0.75])
    input = sample['input_ids'][sample['token_type_ids']==0][None, :]
    token_type_ids = sample['token_type_ids'][0][sample['token_type_ids'][0]==0][None, :]
    pixel_values = sample['pixel_values']
    attention_mask = sample['attention_mask'][0][sample['token_type_ids'][0]==0][None, :]
    output = model.generate(input_ids = input, attention_mask = attention_mask, token_type_ids = token_type_ids, pixel_values = pixel_values, max_new_tokens = 50)
    model_output = processor.batch_decode(output, skip_special_tokens = True)[0]
    model_objects = extract_objects(model_output)
    gt_string = processor.decode(sample['input_ids'][0][sample['token_type_ids'][0]==1], skip_special_tokens =True)
    gt_objects =  extract_objects(gt_string)
    true_bbox, true_class = convert_object_eval(model_objects)
    pred_bbox, pred_class = convert_object_eval(gt_objects)
    ious = box_iou_batch(true_bbox, pred_bbox)
    matches = match_detection_batch(pred_class, true_class, ious,thresholds)
    pred_confidence = np.ones(pred_class.shape[0])
    results = [matches, pred_confidence, pred_class, true_class]
    return results