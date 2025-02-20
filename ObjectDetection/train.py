import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms.functional import resize, to_pil_image

import re
import wandb

from transformers import AutoProcessor
from typing import List

from utils import get_device, draw_wandb_inference_bbox

from pg_datasets import label2idx, idx2label
from model import processor


DETECT_RE = re.compile(
    r"(.*?)" + r"((?:<loc\d{4}>){4})\s*" + r"([^;<>]+) ?(?:; )?",
)

device = get_device()


def post_process(text):
    """
    Remove new line characters from the text. The new line character affects the regex matching
    """
    text = re.sub('[\n]', ' ', text)
    return text


def extract_objects(
        detection_string: str, 
        image_width:int = 224, 
        image_height:int=224, 
        unique_labels:bool=False
    ):
    """
    Exract the predicted objects and their bounding boxes
    Args:
        detection_string: str: The string containing the predicted objects
        image_width: int: The width of the image
        image_height: int: The height of the image
        unique_labels: bool: If True, add a prime to the label if it already exists in the list of labels
    Returns:
        objects: list: List of dictionaries containing the bounding box and the label of the object
    """
    objects = []
    seen_labels = set()

    detection_string = post_process(detection_string)

    while detection_string:
        match = DETECT_RE.match(detection_string)
        if not match:
            break

        prefix, locations, label = match.groups()
        location_values = [int(loc) for loc in re.findall(r"\d{4}", locations)]
        y1, x1, y2, x2 = [value / 1024 for value in location_values]
        y1, x1, y2, x2 = map(
            round,
            (y1 * image_height, x1 * image_width, y2 * image_height, x2 * image_width),
        )

        label = label.strip()  # Remove trailing spaces from label

        if unique_labels and label in seen_labels:
            label = (label or "") + "'"
        seen_labels.add(label)

        objects.append(dict(xyxy=(x1, y1, x2, y2), name=label))

        detection_string = detection_string[len(match.group()) :]

    return objects


def log_image_bboxes_to_wandb(
        image: torch.Tensor, 
        outputs: List[dict]
        ):
    """
    Function to log the image with the bounding boxes to Weights & Biases
    Args:
        image: torch.Tensor: The image tensor
        outputs: List[dict]: List of dictionaries containing the bounding box and the label of the object
    Returns:
        img: wandb.Image: The image with the bounding boxes
    """
    box_data = []
    for output in outputs:
        bbox = output['xyxy']
        name = output['name']
        position = dict(
            minX=bbox[0],
            minY=bbox[1],
            maxX=bbox[2],
            maxY=bbox[3],
        )
        box_data.append(
            dict(
                position=position,
                class_id=label2idx.get(name, 'Wrong'),
                domain = 'pixel'
            )
        )

    wandb_bbox_dict = dict(
        box_data    = box_data,
        class_labels= idx2label
    )

    wandb_boxes = {}

    wandb_boxes['predictions'] = wandb_bbox_dict

        # Log the image with the bounding boxes
    img = wandb.Image(image.transpose(1, 2, 0), boxes=wandb_boxes)
    return img



def evaluate(model: nn.Module, 
             val_loader: DataLoader, 
             device: torch.device, 
             tokenizer: AutoProcessor, 
             step: int, 
             epoch:int, 
             log_indices: List[int], 
             max_samples: int = None,
             processor: AutoProcessor = None
             ):
    """
    Function to evaluate the model
    Args:
        model: nn.Module: The model to evaluate
        val_loader: DataLoader: The validation dataloader
        device: torch.device: The device to run the evaluation on
        tokenizer: AutoProcessor: The tokenizer to use for decoding the model output
        step: int: The step number
        epoch: int: The epoch number
        log_indices: List[int]: The indices of the samples to log
        max_samples: int: The maximum number of samples to evaluate
        processor: AutoProcessor: The tokenizer to use for decoding the model output
    """
    model.eval()

    total_loss = 0
    log_gt_bbox = []
    log_pred_bbox = []
    # Initialize the table for logging to Weights & Biases
    table = wandb.Table(columns = ['Ground Truth BBOX', "Predicted BBOX"])
    
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
                
                # Convert the model predictions to bbox
                objects = extract_objects(model_preds, 224, 224, unique_labels=False)

                # pass it to the function to draw the bbox
                ann_image = log_image_bboxes_to_wandb(image, objects)
                log_pred_bbox.append(ann_image)

                gt_string  = processor.batch_decode(
                                                input_ids[0][token_type_ids[0]==1][None, :],
                                                skip_special_tokens=True)[0]
                
                gt_objects = extract_objects(gt_string, 224, 224, unique_labels=False)
                
                gt_image = log_image_bboxes_to_wandb(image, gt_objects)
                
                log_gt_bbox.append(gt_image)

                # Add data to the table
                table.add_data(gt_image, ann_image)

            # Log the images and bboxes to wandb
    wandb.log({"Evaluation Results epoch {} step {}".format(epoch, step): table, "Epoch": epoch, "Step": step})

    avg_loss = total_loss / len(val_loader)
    model.train()
    return avg_loss


def train(
        model: nn.Module, 
        train_dataloader: DataLoader,
        valid_dataloader: DataLoader,
        num_epochs: int, 
        device: torch.device,
        train_config: dict,
        optimizer: torch.optim.Optimizer,
        log_indices: list = None,
        tokenizer: AutoProcessor = None
    ):
    """
    Function to train the model
    Args:
        model: nn.Module: The model to train
        train_dataloader: DataLoader: The training dataloader
        valid_dataloader: DataLoader: The validation dataloader
        num_epochs: int: The number of epochs to train the model
        device: torch.device: The device to train the model on
        train_config: dict: The training configuration
        optimizer: torch.optim.Optimizer: The optimizer to use for training the model
        log_indices: list: The indices of the samples to log
        tokenizer: AutoProcessor: The tokenizer to use for decoding the model output
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
            
    
            if (step % train_config["accumulation_steps"]) == 0:
                for param in model.parameters():
                    if param.grad is not None:
                        param.grad /= train_config["accumulation_steps"]

                optimizer.step()

                optimizer.zero_grad()
    
    
            total_train_loss += loss.item()
            batch_count += 1
    
            # Log batch loss to wandb
            wandb.log({"Batch Loss": loss.item(), "Step": step})
    
            print(f"Epoch: {epoch}, Step: {step}, Batch Loss: {loss.item()}")
    
            if step % train_config["eval_interval"] == 0:
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


def test(
        model: nn.Module, 
        test_loader: DataLoader, 
        device: torch.device, 
        tokenizer: AutoProcessor, 
        step: int, 
        epoch:int, 
        log_indices: List[int], 
        max_samples: int = None
    )->None:
    model.eval()

    # Intialize the table for logging to Weights & Biases
    table = wandb.Table(columns = ['Image','Ground Truth BBOX', "Predicted BBOX"])

    log_images = []
    log_gt_bbox = []
    log_pred_bbox = []

    
    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            if max_samples and i >=max_samples:
                break

            if batch is None:
                continue

            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            pixel_values = batch['pixel_values'].to(device)
            labels = input_ids.clone().detach()

            if i in log_indices:
                generate_ids = model.generate(**batch, max_new_tokens=50)
                generated_outputs = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
                detection_string = generated_outputs.split("\n")[1]
                objects = extract_objects(detection_string, 224, 224, unique_labels=False)
                
                
                # log_pred_texts.append()
                log_images.append(pixel_values.cpu().squeeze().float().numpy())
                
            
                # Convert image to PIL format
                pil_img = to_pil_image(resize(torch.from_numpy(log_images[-1]).squeeze(), (224, 224))).convert("RGB")

                gt_string  = tokenizer.decode(labels[0], skip_special_tokens=True).split("\n")[1]
                gt_objects = extract_objects(gt_string, 224, 224, unique_labels=False)
                gt_bbox    = draw_wandb_inference_bbox(pil_img, gt_objects)
                

                bbox_image = draw_wandb_inference_bbox(pil_img, objects)
                log_pred_bbox.append(bbox_image)
                log_gt_bbox.append(gt_bbox)
                

                # Add data to the table
                table.add_data(wandb.Image(pil_img), wandb.Image(gt_bbox), wandb.Image(bbox_image))
            
    wandb.log({"Evaluation Results epoch {} step {}".format(epoch, step): table, "Epoch": epoch, "Step": step})