import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os

import wandb

from transformers import AutoProcessor

from typing import List
from torchvision.transforms.functional import resize, to_pil_image


def evaluate(
        model: nn.Module, 
        val_loader: DataLoader, 
        device: torch.device, 
        tokenizer: AutoProcessor, 
        step: int, 
        epoch:int, 
        task: str, 
        log_indices: List,
        processor: AutoProcessor,
        max_samples: int = None
    )->torch.Tensor:
    """
    Evaluate the model on the validation set and log some of the outputs to Weights & Biases
    args:
        model: The model to evaluate
        val_loader: DataLoader object for the validation set
        device: torch.device object
        tokenizer: AutoProcessor object from HuggingFace
        step: int, the current training step number
        epoch: int, the current epoch number
        task: str, the task for which the model is being evaluated
        log_indices: List of indices for which to log the outputs to Weights & Biases
        processor: AutoProcessor object from HuggingFace
        max_samples: int, default is None. The maximum number of samples to evaluate
    """
    model.eval()

    total_loss = 0
    log_images = []
    log_gt_texts = []
    log_pred_texts = []
    # Intialize the table for logging to Weights & Biases
    table = wandb.Table(columns = ['Image', "Ground Truth Report", "Predicted Report"])

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
                input_ids = input_ids,
                attention_mask = attention_mask,
                pixel_values = pixel_values,
                token_type_ids = token_type_ids,
                labels = labels
            )

            loss = outputs.loss

            total_loss += loss.item()

            predictions = torch.argmax(outputs.logits, dim=-1)
            predicted_text = tokenizer.decode(predictions[0], skip_special_tokens=True)

            if i in log_indices:
                log_images.append(pixel_values.cpu().squeeze().float().numpy())
                log_gt_texts.append(processor.batch_decode(
                                                input_ids[0][token_type_ids[0]==1][None, :],
                                                skip_special_tokens=True)[0]
                                               )
                
                log_pred_texts.append(processor.batch_decode(
                                                model.generate(
                                                input_ids = input_ids[0][token_type_ids[0]==0][None, :], 
                                                max_new_tokens= 50),
                                                skip_special_tokens =True, 
                                                clean_up_tokenization_spaces=False)[0]
                                     )

                # Convert image to PIL format
                pil_img = to_pil_image(resize(torch.from_numpy(log_images[-1]).squeeze(), (224, 224))).convert("RGB")

                # Add data to the table
                table.add_data(wandb.Image(pil_img), log_gt_texts[-1], log_pred_texts[-1])

    wandb.log({"{} Evaluation Results epoch {} step {}".format(task, epoch, step): table, "Epoch": epoch, "Step": step})

    avg_loss = total_loss / (i + 1)  # i+1 to account for the loop index
    model.train()

    return avg_loss


def train(
        model: nn.Module,
        train_dataloader: DataLoader,
        valid_dataloader: DataLoader,
        device: torch.device,
        train_config: dict,
        optimizer: torch.optim.Optimizer,
        processor: AutoProcessor,
        tokenizer: AutoProcessor,
        log_indices: List,
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
            token_type_ids = batch["token_type_ids"].to(device)
            labels = batch["labels"].to(device)
    
    
            outputs = model(
                input_ids=input_ids, 
                attention_mask=attention_mask, 
                pixel_values=pixel_values, 
                labels=labels,
                token_type_ids = token_type_ids
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
    
            if step % train_config["accumulation_steps"] == 0:
                val_loss = evaluate(model, valid_dataloader, device, tokenizer=tokenizer, log_indices=log_indices, step=step, epoch=epoch, processor=processor,  task ="RRG")
                wandb.log({
                    "Validation Loss": val_loss,
                    "Step": step
                })
                print(f"Step: {step}, Validation Loss: {val_loss}")
    
                if train_config['save_best_model']:
                    # Save the best model on disk
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        best_model_path = os.path.join(train_config["save_dir"], f"best_model")
                        model.save_pretrained(best_model_path, safe_serialization=False)
                        tokenizer.save_pretrained(best_model_path)
    
                avg_train_loss = total_train_loss / batch_count
                wandb.log({
                    "Epoch": epoch,
                    "Average Training Loss": avg_train_loss,
                })
                
        print(f"Epoch: {epoch}, Average Training Loss: {avg_train_loss}")