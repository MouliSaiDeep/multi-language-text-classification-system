import os
import argparse
import torch
import sys
import io
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from dotenv import load_dotenv

# Ensure UTF-8 output encoding to avoid UnicodeEncodeError in Windows terminals
if sys.platform.startswith('win'):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

def main():
    load_dotenv()
    
    parser = argparse.ArgumentParser(description="Export fine-tuned PyTorch model to ONNX")
    parser.add_argument("--checkpoint_dir", type=str, default=None, 
                        help="Path to fine-tuned PyTorch checkpoint. If not provided, base model will be used.")
    parser.add_argument("--output_path", type=str, default=None,
                        help="Path to save the ONNX file. If not provided, reads ONNX_MODEL_PATH from environment.")
    args = parser.parse_args()

    model_name = os.getenv("MODEL_NAME", "xlm-roberta-base")
    max_length = int(os.getenv("MODEL_MAX_LENGTH", "128"))
    
    onnx_model_path = args.output_path or os.getenv("ONNX_MODEL_PATH", "src/api/model.onnx")
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(onnx_model_path), exist_ok=True)
    
    load_path = args.checkpoint_dir if args.checkpoint_dir else model_name
    print(f"Loading model from: {load_path}")
    
    model = AutoModelForSequenceClassification.from_pretrained(load_path)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    model.eval()
    
    # Create dummy inputs (premise and hypothesis pair)
    dummy_text = ("This is a test premise.", "This is a test hypothesis.")
    inputs = tokenizer(
        dummy_text[0],
        dummy_text[1],
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    )
    
    dummy_input_ids = inputs["input_ids"]
    dummy_attention_mask = inputs["attention_mask"]
    
    print(f"Exporting to ONNX: {onnx_model_path}")
    
    # Export model to ONNX
    torch.onnx.export(
        model,
        (dummy_input_ids, dummy_attention_mask),
        onnx_model_path,
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch_size", 1: "sequence_length"},
            "attention_mask": {0: "batch_size", 1: "sequence_length"},
            "logits": {0: "batch_size"}
        },
        opset_version=14
    )
    
    print("Export completed successfully!")

if __name__ == "__main__":
    main()
