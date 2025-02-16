import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from utils import get_device

from PIL import Image
import numpy as np

from transformers import AutoProcessor

import wandb
from typing import List

from dataset import label2idx, idx2label, string2mask

import matplotlib.pyplot as plt

device = get_device()


import matplotlib.patches as patches
import numpy.ma as ma

def plot_image_with_mask(image, mask):
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    image = np.array(image)
    
    for m in mask:
        bbox = m['xyxy']
        mask_array = m['mask']
        name = m['name']

        mask_temp = image.copy()
        mask_temp[mask_array > 0.5]  = [255, 0, 0]
        mask_temp[mask_array <= 0.5] = [0, 0, 0]
        ax.imshow(mask_temp, alpha=0.9)
        ax.imshow(image, cmap='gray' , alpha=0.5)
        
        # Plot the bounding box
        rect = patches.Rectangle((bbox[0], bbox[1]), bbox[2] - bbox[0], bbox[3] - bbox[1], linewidth=2, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        
        # Add the label
        ax.text(bbox[0], bbox[1] - 10, name, color='red', fontsize=12, weight='bold')
    
    plt.axis('off')
    plt.show()



def log_image_masks_to_wandb(outputs):
    box_data = []
    composed_mask = np.zeros((224, 224), dtype = np.float64)
    for output in outputs:
        bbox = output.get('xyxy', [0,0, 224, 224])
        mask = output.get('mask', np.zeros((224, 224), dtype = np.float64))
        mask = mask if isinstance(mask, np.ndarray) else np.array(mask, dtype=np.float64)
        assert mask.dtype == composed_mask.dtype
        composed_mask += mask
        name = output.get('name', 'wrong')
        position = dict(
            minX=bbox[0],
            minY=bbox[1],
            maxX=bbox[2],
            maxY=bbox[3],
        )
        box_data.append(
            dict(
                position=position,
                class_id=label2idx.get(name, label2idx['wrong']),
                domain = 'pixel'
            )
        )

    wandb_bbox_dict = dict(
        box_data    = box_data,
        class_labels= idx2label
    )

    wandb_boxes = {}

    wandb_boxes['predictions'] = wandb_bbox_dict
    
    image = Image.fromarray(composed_mask.astype('uint8')*255).convert('RGB') if isinstance(mask, np.ndarray) else np.zeros((224, 224))

        # Log the image with the bounding boxes
    img = wandb.Image(image, boxes=wandb_boxes)
    return img


def evaluate(
        model: nn.Module, 
        val_loader: DataLoader,
        processor: AutoProcessor,
        device: torch.device, 
        step: int, 
        epoch:int, 
        log_indices: 
        List[int], 
        max_samples: int = None
        ):
    """Evaluate the model on the validation set"""
    model.eval()

    total_loss = 0
    # log_images = []
    log_gt_bbox = []
    log_pred_bbox = []
    # Initialize the table for logging to Weights & Biases
    table = wandb.Table(columns = ['GT Mask ', "Predicted Mask"])
    
    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            if max_samples and i >=max_samples:
                break

            if batch is None:
                continue

            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            pixel_values = batch['pixel_values'].to(device)
            token_type_ids = batch['token_type_ids'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(
                input_ids=input_ids, 
                attention_mask=attention_mask, 
                pixel_values=pixel_values, 
                labels=labels,
                token_type_ids=token_type_ids
            )

            loss = outputs.loss

            total_loss += loss.item()

            if i in log_indices:
                image = pixel_values.cpu().squeeze().float().numpy()
                model_output = model.generate(
                    input_ids= input_ids[0][token_type_ids[0] == 0].unsqueeze(0),
                    attention_mask = attention_mask[0][token_type_ids[0] == 0].unsqueeze(0),
                    token_type_ids = token_type_ids[0][token_type_ids[0] == 0].unsqueeze(0),
                    pixel_values = pixel_values,
                    max_new_tokens =50
                )
                model_preds = processor.batch_decode(model_output, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
                
                # Convert the model predictions to segmentation strings
                objects = string2mask({"suffix": model_preds})

                # print(objects)

                # pass it to the function to draw the segmentation mask and bbox
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
    wandb.log({"Evaluation Results epoch {} step {}".format(epoch, step): table, "Epoch": epoch, "Step": step})

    avg_loss = total_loss / len(val_loader)
    model.train()
    return avg_loss

def train(
        model:nn.Module,
        train_dataloader: torch.utils.data.DataLoader,
        valid_dataloader: torch.utils.data.DataLoader,
        tokenizer: AutoProcessor,
        log_indices: List[int],
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        config:dict,
        num_epochs:int
        ):
    """
    Function to train the model
    Parameters:
    num_epochs: number of epochs for which to train the model
    """
    model.train()
    best_val_loss = float('inf')
    for epoch in range(num_epochs):  # Number of epochs
        total_train_loss = 0
        batch_count = 0
    
        step = 0
        for batch in train_dataloader:
            step += 1
    
            if batch is None:  # Skip if the batch is None
                continue
    
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            pixel_values = batch['pixel_values'].to(device)
            token_type_ids = batch['token_type_ids'].to(device)
            labels = batch['labels'].to(device)

            
            outputs = model(
                input_ids=input_ids, 
                attention_mask=attention_mask, 
                pixel_values=pixel_values, 
                labels=labels,
                token_type_ids=token_type_ids
            )
            loss = outputs.loss
                
            predictions = torch.argmax(outputs.logits, dim=-1)                
    
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
                val_loss = evaluate(model, valid_dataloader, device, tokenizer = tokenizer, step=step, epoch = epoch, log_indices = log_indices)
                wandb.log({
                    "Validation Loss": val_loss,
                    "Step": step
                })
                print(f"Step: {step}, Validation Loss: {val_loss}")
    
                avg_train_loss = total_train_loss / batch_count
                wandb.log({
                    "Epoch": epoch,
                    "Average Training Loss": avg_train_loss,
                })

                
        print(f"Epoch: {epoch}, Average Training Loss: {avg_train_loss}")
    
    wandb.finish()