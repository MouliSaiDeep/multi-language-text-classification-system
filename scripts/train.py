import os
import sys
# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import torch
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
from torch.optim import AdamW
import mlflow
from dotenv import load_dotenv
from datasets import load_dataset, concatenate_datasets

from src.models.model import get_model, get_tokenizer
from src.data.dataset import XNLIDataset
from src.data.sampler import BalancedBatchSampler
from src.training.trainer import train_epoch, evaluate_model

def load_dataset_with_retries(path, name=None, split=None, max_retries=5, delay=5):
    import time
    for attempt in range(max_retries):
        try:
            return load_dataset(path, name, split=split)
        except Exception as e:
            if attempt == max_retries - 1:
                raise e
            print(f"Error loading dataset {path} {name} (attempt {attempt + 1}/{max_retries}): {e}. Retrying in {delay} seconds...")
            time.sleep(delay)
            delay *= 2

def main():
    load_dotenv()
    
    parser = argparse.ArgumentParser(description="Fine-tune multilingual model on XNLI")
    parser.add_argument("--epochs", type=int, default=1, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoint", help="Directory to save model checkpoints")
    parser.add_argument("--smoke_test", action="store_true", help="Run a quick smoke test with tiny data subsets")
    args = parser.parse_args()

    # Read configuration from .env / environment
    model_name = os.getenv("MODEL_NAME", "xlm-roberta-base")
    max_length = int(os.getenv("MODEL_MAX_LENGTH", "128"))
    batch_size = int(os.getenv("BATCH_SIZE", "32"))
    mlflow_tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "./mlruns")

    # Set up MLflow
    # Allow file store backend to avoid maintenance mode exception
    os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    mlflow.set_experiment("xnli_classification")

    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load tokenizer and model
    print(f"Loading tokenizer and model: {model_name}")
    tokenizer = get_tokenizer(model_name)
    model = get_model(model_name, num_labels=3)
    model.to(device)

    # 9 training languages (Swahili 'sw' is strictly held out)
    train_langs = ["en", "fr", "es", "de", "zh", "ru", "hi", "vi", "ar"]
    
    print("Loading datasets...")
    train_datasets = []
    val_datasets = []
    
    for lang in train_langs:
        print(f"  Loading language: {lang}")
        lang_train = load_dataset_with_retries("facebook/xnli", lang, split="train")
        lang_val = load_dataset_with_retries("facebook/xnli", lang, split="validation")
        
        # Add language tag to datasets
        lang_train = lang_train.add_column("language", [lang] * len(lang_train))
        lang_val = lang_val.add_column("language", [lang] * len(lang_val))
        
        if args.smoke_test:
            # Select tiny subsets for fast execution during validation/testing
            lang_train = lang_train.select(range(min(40, len(lang_train))))
            lang_val = lang_val.select(range(min(20, len(lang_val))))
            
        train_datasets.append(lang_train)
        val_datasets.append(lang_val)

    # Combine datasets
    concat_train = concatenate_datasets(train_datasets)
    concat_val = concatenate_datasets(val_datasets)

    # Create dataset objects
    train_dataset = XNLIDataset(concat_train, tokenizer, max_length=max_length)
    val_dataset = XNLIDataset(concat_val, tokenizer, max_length=max_length)

    # Samplers and DataLoaders
    # Use balanced batch sampler to stratify the 9 languages in every training batch
    train_sampler = BalancedBatchSampler(train_dataset, batch_size=batch_size, drop_last=True)
    train_loader = DataLoader(train_dataset, batch_sampler=train_sampler, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    # Optimizer and Scheduler
    optimizer = AdamW(model.parameters(), lr=args.lr, eps=1e-8)
    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps
    )

    # AMP GradScaler
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")

    print("Starting training...")
    with mlflow.start_run() as run:
        # Log hyperparameters
        mlflow.log_params({
            "learning_rate": args.lr,
            "batch_size": batch_size,
            "model_name": model_name,
            "max_length": max_length,
            "num_epochs": args.epochs,
            "optimizer": "AdamW",
            "scheduler": "linear_warmup",
            "smoke_test": args.smoke_test
        })

        best_val_f1 = -1.0
        
        for epoch in range(1, args.epochs + 1):
            train_loss = train_epoch(
                model=model,
                dataloader=train_loader,
                optimizer=optimizer,
                scheduler=scheduler,
                device=device,
                epoch=epoch,
                scaler=scaler
            )
            
            val_f1, _, _ = evaluate_model(model, val_loader, device)
            
            # Log metrics to MLflow
            mlflow.log_metric("val_macro_f1", val_f1, step=epoch)
            print(f"Epoch {epoch} | Train Loss: {train_loss:.4f} | Val Macro F1: {val_f1:.4f}")
            
            # Save checkpoint
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                os.makedirs(args.checkpoint_dir, exist_ok=True)
                model.save_pretrained(args.checkpoint_dir)
                tokenizer.save_pretrained(args.checkpoint_dir)
                print(f"Saved new best model checkpoint to {args.checkpoint_dir}")
                
        # If no checkpoint was saved (e.g. score was 0), save at least the final model
        if not os.path.exists(os.path.join(args.checkpoint_dir, "config.json")):
            os.makedirs(args.checkpoint_dir, exist_ok=True)
            model.save_pretrained(args.checkpoint_dir)
            tokenizer.save_pretrained(args.checkpoint_dir)
            print(f"Saved final model checkpoint to {args.checkpoint_dir}")

    print("Training finished successfully!")

if __name__ == "__main__":
    main()
