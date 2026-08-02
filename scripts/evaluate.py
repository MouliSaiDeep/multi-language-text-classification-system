import os
import sys
import io

# Ensure UTF-8 output encoding to avoid UnicodeEncodeError in Windows terminals
if sys.platform.startswith('win'):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import time
import json
import subprocess
import torch
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import precision_recall_fscore_support, f1_score
import onnxruntime as ort
from dotenv import load_dotenv

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
    
    parser = argparse.ArgumentParser(description="Evaluate fine-tuned model and export metrics")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoint", 
                        help="Path to PyTorch checkpoint directory")
    parser.add_argument("--smoke_test", action="store_true", 
                        help="Run evaluations on a tiny fraction of data")
    args = parser.parse_args()

    model_name = os.getenv("MODEL_NAME", "xlm-roberta-base")
    onnx_model_path = os.getenv("ONNX_MODEL_PATH", "src/api/model.onnx")
    max_length = int(os.getenv("MODEL_MAX_LENGTH", "128"))

    # Ensure ONNX model exists; if not, trigger export script
    if not os.path.exists(onnx_model_path):
        print(f"ONNX model not found at {onnx_model_path}. Exporting...")
        cmd = ["python", "scripts/export_onnx.py", "--checkpoint_dir", args.checkpoint_dir]
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"
        subprocess.run(cmd, check=True, env=env)

    # 1. Load ONNX model for evaluation
    print(f"Loading ONNX model: {onnx_model_path}")
    session = ort.InferenceSession(onnx_model_path)
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Languages list
    train_langs = ["en", "fr", "es", "de", "zh", "ru", "hi", "vi", "ar"]
    heldout_lang = "sw"

    # Evaluation results dictionary
    results = {}
    f1_scores_train = []

    # Helper function to evaluate ONNX model on a list of samples
    def evaluate_onnx(lang_dataset):
        preds = []
        labels = []
        
        # Determine number of samples to evaluate
        num_samples = len(lang_dataset)
        if args.smoke_test:
            num_samples = min(20, num_samples)
            
        for i in range(num_samples):
            item = lang_dataset[i]
            premise = item["premise"]
            hypothesis = item["hypothesis"]
            label = item["label"]
            
            inputs = tokenizer(
                premise,
                hypothesis,
                padding="max_length",
                truncation=True,
                max_length=max_length,
                return_tensors="np"
            )
            
            ort_inputs = {
                "input_ids": inputs["input_ids"].astype(np.int64),
                "attention_mask": inputs["attention_mask"].astype(np.int64)
            }
            
            ort_outs = session.run(None, ort_inputs)
            pred = np.argmax(ort_outs[0], axis=-1)[0]
            
            preds.append(pred)
            labels.append(label)
            
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, preds, average="macro", zero_division=0
        )
        return precision, recall, f1

    # Measure ONNX Latency
    print("Measuring ONNX inference latency...")
    # Prepare dummy data
    inputs = tokenizer(
        "This is a premise.",
        "This is a hypothesis.",
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="np"
    )
    ort_inputs = {
        "input_ids": inputs["input_ids"].astype(np.int64),
        "attention_mask": inputs["attention_mask"].astype(np.int64)
    }
    
    # Warmup
    for _ in range(10):
        _ = session.run(None, ort_inputs)
        
    start_time = time.perf_counter()
    num_runs = 50
    for _ in range(num_runs):
        _ = session.run(None, ort_inputs)
    onnx_latency_ms = (time.perf_counter() - start_time) / num_runs * 1000
    print(f"ONNX latency: {onnx_latency_ms:.2f} ms")

    # Measure PyTorch Latency
    print("Measuring PyTorch inference latency...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    try:
        pt_model = AutoModelForSequenceClassification.from_pretrained(args.checkpoint_dir)
    except Exception:
        pt_model = AutoModelForSequenceClassification.from_pretrained(model_name)
        
    pt_model.to(device)
    pt_model.eval()
    
    pt_inputs = tokenizer(
        "This is a premise.",
        "This is a hypothesis.",
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    )
    pt_input_ids = pt_inputs["input_ids"].to(device)
    pt_attention_mask = pt_inputs["attention_mask"].to(device)
    
    with torch.no_grad():
        for _ in range(10):
            _ = pt_model(pt_input_ids, pt_attention_mask)
            
        start_time = time.perf_counter()
        for _ in range(num_runs):
            _ = pt_model(pt_input_ids, pt_attention_mask)
        pt_latency_ms = (time.perf_counter() - start_time) / num_runs * 1000
    print(f"PyTorch latency: {pt_latency_ms:.2f} ms")

    # Evaluate each training language
    print("Evaluating training languages...")
    for lang in train_langs:
        print(f"  Evaluating {lang}...")
        ds = load_dataset_with_retries("facebook/xnli", lang, split="test")
        p, r, f1 = evaluate_onnx(ds)
        results[lang] = {"precision": p, "recall": r, "f1": f1}
        f1_scores_train.append(f1)
        print(f"    F1 for {lang}: {f1:.4f}")

    # Evaluate Swahili (heldout zero-shot)
    print(f"Evaluating zero-shot language: {heldout_lang}...")
    ds_sw = load_dataset_with_retries("facebook/xnli", heldout_lang, split="test")
    p_sw, r_sw, f1_sw = evaluate_onnx(ds_sw)
    results[heldout_lang] = {"precision": p_sw, "recall": r_sw, "f1": f1_sw}
    print(f"    F1 for {heldout_lang} (zero-shot): {f1_sw:.4f}")

    macro_f1 = np.mean(f1_scores_train)
    print(f"\nFinal Macro F1 across 9 languages: {macro_f1:.4f}")
    print(f"Zero-shot Swahili F1: {f1_sw:.4f}")

    # Write evaluation.json
    eval_json = {
        "macro_f1": float(round(macro_f1, 4)),
        "zero_shot_f1": float(round(f1_sw, 4)),
        "average_latency_ms": float(round(onnx_latency_ms, 2)),
        "smoke_test": bool(args.smoke_test)
    }
    
    with open("evaluation.json", "w") as f:
        json.dump(eval_json, f, indent=2)
    print("Saved evaluation.json")

    # Generate EVALUATION.md
    print("Writing EVALUATION.md...")
    with open("EVALUATION.md", "w", encoding="utf-8") as f:
        if args.smoke_test:
            f.write("> **⚠️ Smoke-test run.** These metrics come from `--smoke_test` (a tiny subset, ~1 epoch) used to validate the pipeline end-to-end. They are not representative of real model performance and are expected to fall short of the target thresholds below. Rerun `python scripts/train.py --epochs <N>` and `python scripts/evaluate.py` (both without `--smoke_test`) to produce a real evaluation before treating these numbers as final.\n\n")
        f.write("# Model Evaluation Report\n\n")
        f.write("This report summarizes the evaluation metrics for the cross-lingual NLI model.\n\n")
        
        f.write("## Test Metrics per Language\n\n")
        f.write("| Language | Code | Precision | Recall | F1-Score |\n")
        f.write("| --- | --- | --- | --- | --- |\n")
        for lang in train_langs:
            metrics = results[lang]
            f.write(f"| {lang.upper()} | {lang} | {metrics['precision']:.4f} | {metrics['recall']:.4f} | {metrics['f1']:.4f} |\n")
        f.write("\n")
        
        f.write("## Zero-Shot Evaluation (Held-out Language)\n\n")
        f.write("| Language | Code | Precision | Recall | F1-Score |\n")
        f.write("| --- | --- | --- | --- | --- |\n")
        f.write(f"| SWAHILI | {heldout_lang} | {p_sw:.4f} | {r_sw:.4f} | {f1_sw:.4f} |\n\n")
        
        f.write("## Performance & Latency Comparison\n\n")
        f.write("| Framework | Average Latency (ms) | Speedup |\n")
        f.write("| --- | --- | --- |\n")
        f.write(f"| PyTorch (FP32) | {pt_latency_ms:.2f} ms | 1.0x |\n")
        f.write(f"| ONNX Runtime | {onnx_latency_ms:.2f} ms | {pt_latency_ms/onnx_latency_ms:.2f}x |\n\n")
        
        f.write("## Summary Metrics\n\n")
        f.write(f"- **Macro F1 (9 training languages)**: {macro_f1:.4f}\n")
        f.write(f"- **Zero-shot F1 (Swahili)**: {f1_sw:.4f}\n")
        f.write(f"- **Target Metrics**: macro_f1 >= 0.75, zero_shot_f1 >= 0.60\n")

    print("EVALUATION.md written successfully!")

if __name__ == "__main__":
    main()
