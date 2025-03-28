"""Training script for the Segmentation model."""


import matplotlib.pyplot as plt
import numpy as np
import torch
import wandb
from dataset import idx2label, label2idx, string2mask
from matplotlib import patches
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader
from transformers import AutoProcessor
from utils import get_device

device = get_device()

mask_threshold = 0.5


def plot_image_with_mask(
        image: Image.Image,
        mask:list[dict[str, str]],
        ) -> None:
    """Plot the image with the bounding boxes and masks.

    Args:
        image (Image.Image): Image to plot.
        mask (list[dict[str, str]]): List of dictionaries containing
                                        the bounding box and mask data.

    Returns:
        None

    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    image = np.array(image)

    for m in mask:
        bbox = m["xyxy"]
        mask_array = m["mask"]
        name = m["name"]

        mask_temp = image.copy()
        mask_temp[mask_array > mask_threshold]  = [255, 0, 0]
        mask_temp[mask_array <= mask_threshold] = [0, 0, 0]
        ax.imshow(mask_temp, alpha=0.9)
        ax.imshow(image, cmap="gray" , alpha=0.5)

        # Plot the bounding box
        rect = patches.Rectangle((bbox[0],bbox[1]),
                                  bbox[2] - bbox[0],
                                  bbox[3] - bbox[1],
                                  linewidth=2,
                                  edgecolor="r",
                                  facecolor="none")
        ax.add_patch(rect)

        # Add the label
        ax.text(bbox[0],
                bbox[1] - 10,
                name, color="red",
                fontsize=12,
                weight="bold")

    plt.axis("off")
    plt.show()



def log_image_masks_to_wandb(outputs: list[dict[str, str]]) -> wandb.Image:
    """Log the image with the bounding boxes to wandb.

    Args:
        outputs (list[dict[str, str]]): List of dictionaries
                                containing the bounding box and mask data.

    Returns:
        wandb.Image: Wandb image object with the bounding boxes and masks.

    """
    box_data = []
    composed_mask = np.zeros((224, 224), dtype = np.float64)
    for output in outputs:
        bbox = output.get("xyxy", [0,0, 224, 224])
        mask = output.get("mask", np.zeros((224, 224), dtype = np.float64))
        mask = mask if isinstance(mask, np.ndarray)\
              else np.array(mask, dtype=np.float64)
        if mask.shape != composed_mask.shape:
            error_msg = f"Mask shape {mask.shape} does not match\
                  composed mask shape {composed_mask.shape}"
            raise ValueError(error_msg)
        composed_mask += mask
        name = output.get("name", "wrong")
        position = {
            "minX":bbox[0],
            "minY":bbox[1],
            "maxX":bbox[2],
            "maxY":bbox[3],
        }
        box_data.append(
            {
                "position": position,
                "class_id":label2idx.get(name, label2idx["wrong"]),
                "domain": "pixel",
            },
        )

    wandb_bbox_dict = {
        "box_data": box_data,
        "class_labels": idx2label,
    }

    wandb_boxes = {}

    wandb_boxes["predictions"] = wandb_bbox_dict

    image = Image.fromarray(composed_mask.astype("uint8")*255).convert("RGB")\
    if isinstance(mask, np.ndarray) else np.zeros((224, 224))

        # Log the image with the bounding boxes
    return wandb.Image(image, boxes=wandb_boxes)


def evaluate(
        model: nn.Module,
        val_loader: DataLoader,
        processor: AutoProcessor,
        device: torch.device,
        step: int,
        epoch:int,
        log_indices:
        list[int],
        max_samples: int = 10,
        ) -> float:
    """Evaluate the model.

    Args:
        model: model to evaluate
        val_loader: dataloader for validation
        processor: processor for the model
        device: device to evaluate the model on
        step: step number
        epoch: epoch number
        log_indices: indices of the batches to log
        max_samples: maximum number of samples to evaluate
    Returns:
        float: average loss for the validation set

    """
    model.eval()

    total_loss = 0
    log_gt_bbox = []
    log_pred_bbox = []
    # Initialize the table for logging to Weights & Biases
    table = wandb.Table(columns = ["GT Mask ", "Predicted Mask"])

    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            if max_samples and i >=max_samples:
                break

            if batch is None:
                continue

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            pixel_values = batch["pixel_values"].to(device)
            token_type_ids = batch["token_type_ids"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                labels=labels,
                token_type_ids=token_type_ids,
            )

            loss = outputs.loss

            total_loss += loss.item()

            if i in log_indices:
                model_output = model.generate(
                input_ids= input_ids[0][token_type_ids[0]==0].unsqueeze(0),
                attention_mask = attention_mask[0][token_type_ids[0] == 0]\
                    .unsqueeze(0),
                token_type_ids = token_type_ids[0][token_type_ids[0] == 0]\
                    .unsqueeze(0),
                pixel_values = pixel_values,
                max_new_tokens =50,
                )
                model_preds = processor.batch_decode(model_output,
                                    skip_special_tokens=True,
                                    clean_up_tokenization_spaces=False)[0]

                # Convert the model predictions to segmentation strings
                objects = string2mask({"suffix": model_preds})

                # pass image to draw the segmentation mask and bbox
                ann_image = log_image_masks_to_wandb(objects)

                log_pred_bbox.append(ann_image)

                gt_string  = processor.batch_decode(
                                input_ids[0][token_type_ids[0]==1][None, :],
                                skip_special_tokens=True)[0]

                gt_objects = string2mask({"suffix": gt_string})

                gt_image = log_image_masks_to_wandb(gt_objects)

                log_gt_bbox.append(gt_image)

                # Add data to the table
                table.add_data(gt_image, ann_image)

            # Log the images and bboxes to wandb
    wandb.log({f"Evaluation Results epoch {epoch} step {step}":\
                table, "Epoch": epoch, "Step": step})

    avg_loss = total_loss / len(val_loader)
    model.train()
    return avg_loss

def train(
        model:nn.Module,
        train_dataloader: torch.utils.data.DataLoader,
        valid_dataloader: torch.utils.data.DataLoader,
        tokenizer: AutoProcessor,
        log_indices: list[int],
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        config:dict,
        num_epochs:int,
        ) -> None:
    """Train the model.

    Args:
        model: model to train
        train_dataloader: dataloader for training
        valid_dataloader: dataloader for validation
        tokenizer: tokenizer for the model
        log_indices: indices of the batches to log
        optimizer: optimizer for the model
        device: device to train the model on
        config: configuration dictionary
        num_epochs: number of epochs for which to train the model
    Returns:
        None

    """
    model.train()
    for epoch in range(num_epochs):  # Number of epochs
        total_train_loss = 0
        batch_count = 0

        step = 0
        for batch in train_dataloader:
            step += 1

            if batch is None:  # Skip if the batch is None
                continue

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            pixel_values = batch["pixel_values"].to(device)
            token_type_ids = batch["token_type_ids"].to(device)
            labels = batch["labels"].to(device)


            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                labels=labels,
                token_type_ids=token_type_ids,
            )
            loss = outputs.loss
            loss.backward()


            if (step % config["accumulation_steps"]) == 0:
                for param in model.parameters():
                    if param.grad is not None:
                        param.grad /= config["accumulation_steps"]

                optimizer.step()
                optimizer.zero_grad()

            total_train_loss += loss.item()
            batch_count += 1

            # Log batch loss to wandb
            wandb.log({"Batch Loss": loss.item(), "Step": step})

            print(f"Epoch: {epoch}, Step: {step}, Batch Loss: {loss.item()}")

            if step % config["eval_interval"] == 0:
                val_loss = evaluate(model,
                                    valid_dataloader,
                                    device,
                                    tokenizer = tokenizer,
                                    step=step,
                                    epoch = epoch,
                                    log_indices = log_indices)
                wandb.log({
                    "Validation Loss": val_loss,
                    "Step": step,
                })
                print(f"Step: {step}, Validation Loss: {val_loss}")

                avg_train_loss = total_train_loss / batch_count
                wandb.log({
                    "Epoch": epoch,
                    "Average Training Loss": avg_train_loss,
                })


        print(f"Epoch: {epoch}, Average Training Loss: {avg_train_loss}")

    wandb.finish()
