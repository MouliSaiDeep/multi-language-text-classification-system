import torch
import mlflow
from tqdm import tqdm
from sklearn.metrics import f1_score
import numpy as np

def train_epoch(model, dataloader, optimizer, scheduler, device, epoch, scaler):
    model.train()
    total_loss = 0.0
    
    # Progress bar
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch} [Train]")
    
    for step, batch in enumerate(progress_bar):
        optimizer.zero_grad()
        
        # Move inputs to device
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)
        
        # AMP autocast
        with torch.cuda.amp.autocast(enabled=device.type == 'cuda'):
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            
        # Scale loss and backward
        scaler.scale(loss).backward()
        
        # Optimizer step and scaler update
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        
        loss_val = loss.item()
        total_loss += loss_val
        
        # Log training loss per step to MLflow
        mlflow.log_metric("train_loss_step", loss_val, step=epoch * len(dataloader) + step)
        
        progress_bar.set_postfix({"loss": f"{loss_val:.4f}"})
        
    avg_loss = total_loss / len(dataloader)
    mlflow.log_metric("train_loss_epoch", avg_loss, step=epoch)
    return avg_loss

def evaluate_model(model, dataloader, device):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label']
            
            with torch.cuda.amp.autocast(enabled=device.type == 'cuda'):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                preds = torch.argmax(outputs.logits, dim=-1)
                
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            
    # Compute macro F1
    macro_f1 = f1_score(all_labels, all_preds, average='macro')
    return macro_f1, all_labels, all_preds
