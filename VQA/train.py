from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
import torch.nn as nn
import torch
from typing import Dict, List
import wandb
from transformers import AutoProcessor
from torchvision.transforms.functional import resize, to_pil_image
from model import get_processor 


def train(config: Dict, 
          model: nn.Module, 
          num_epochs: int, 
          train_dataloader: DataLoader,
          valid_dataloader: DataLoader,
          scheduler: lr_scheduler, 
          device: torch.device, 
          optimizer: torch.optim.Optimizer,
          processor: AutoProcessor,
          log_indices: List[int]
          )-> float:
    """
    Function to train the model
    Parameters:
    num_epochs: number of epochs for which to train the model
    """
    model.train()
    best_val_loss = float('inf')
    for epoch in range(num_epochs):
        total_train_loss = 0
        batch_count = 0

        step = 0
        for batch in  train_dataloader: 
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
            
        # total_loss = loss
        predictions = torch.argmax(outputs.logits, dim=-1)                

        loss.backward()

        if (step % config["accumulation_steps"]) == 0:
            for param in model.parameters():
                if param.grad is not None:
                    param.grad /= config["accumulation_steps"]

            optimizer.step()
            scheduler.step()

            optimizer.zero_grad()


        total_train_loss += loss.item()
        batch_count += 1

        # Log batch loss to wandb
        wandb.log({"Batch Loss": loss.item(), "Step": step})

        print(f"Epoch: {epoch}, Step: {step}, Batch Loss: {loss.item()}")

        if step % config["eval_interval"] == 0:
            val_loss = evaluate(model, valid_dataloader, device, tokenizer=processor.tokenizer, log_indices=log_indices, step=step, epoch=epoch, processor=processor)
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

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {"model_state_dict": model.state_dict(),
                    "epoch": epoch + 1, 
                    }, "model.pth"
            )

                
        print(f"Epoch: {epoch}, Average Training Loss: {avg_train_loss}")

    return avg_train_loss, val_loss


def evaluate(
        model: nn.Module, 
        val_loader: DataLoader, 
        device: torch.device, 
        tokenizer: AutoProcessor, 
        step: int, epoch:int, 
        log_indices: List[int],
        processor: AutoProcessor,
        max_samples: int = None,
        )->float:
    
    model.eval()

    total_loss = 0
    log_images = []
    log_gt_texts = []
    log_pred_texts = []
    # Intialize the table for logging to Weights & Biases
    table = wandb.Table(columns = ['Image', "Ground Truth", "Prediction"])


    for i, batch in enumerate(val_loader):
        if max_samples and i >=max_samples:
            break

        if batch is None:
            continue

        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        pixel_values = batch['pixel_values'].to(device)
        labels = batch['labels'].to(device)
        token_type_ids = batch["token_type_ids"].to(device)

        with torch.no_grad():
            outputs = model(
                input_ids = input_ids,
                attention_mask = attention_mask,
                pixel_values = pixel_values,
                labels = labels
            )

        loss = outputs.loss
        total_loss  += loss.item()

        predictions = torch.argmax(outputs.logits, dim=-1)
        predicted_text = tokenizer.decode(predictions[0], skip_special_tokens=True)

        if i in log_indices:
            log_images.append(pixel_values.cpu().squeeze().float().numpy())
            log_gt_texts.append(tokenizer.decode(input_ids[0], skip_special_tokens=True))
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

            wandb.log({"Classification Evaluation Results epoch {} step {}".format(epoch, step): table, "Epoch": epoch, "Step": step})

    avg_loss = total_loss / (i + 1)  # i+1 to account for the loop index
    model.train()

    return avg_loss
