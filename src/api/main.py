import os
import numpy as np
from typing import List
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from contextlib import asynccontextmanager
import onnxruntime as ort
from transformers import AutoTokenizer
from dotenv import load_dotenv

load_dotenv()

# Pydantic Schemas exactly as specified in the API contract
class Pair(BaseModel):
    premise: str = Field(..., description="The premise sentence")
    hypothesis: str = Field(..., description="The hypothesis sentence")

class ClassifyRequest(BaseModel):
    pairs: List[Pair] = Field(..., description="List of premise-hypothesis pairs to classify")

class Prediction(BaseModel):
    premise: str
    hypothesis: str
    label: str
    confidence: float

class ClassifyResponse(BaseModel):
    predictions: List[Prediction]

# Lifespan Context Manager
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load ONNX model and tokenizer once at startup
    model_name = os.getenv("MODEL_NAME", "xlm-roberta-base")
    onnx_model_path = os.getenv("ONNX_MODEL_PATH", "src/api/model.onnx")
    max_length = int(os.getenv("MODEL_MAX_LENGTH", "128"))
    
    if not os.path.exists(onnx_model_path):
        raise RuntimeError(f"ONNX model file not found at {onnx_model_path}. Please run export_onnx.py first.")
        
    print(f"Loading ONNX model from {onnx_model_path}...")
    app.state.session = ort.InferenceSession(onnx_model_path)
    print(f"Loading tokenizer {model_name}...")
    app.state.tokenizer = AutoTokenizer.from_pretrained(model_name)
    app.state.max_length = max_length
    
    yield
    # Cleanup on shutdown (if any)

app = FastAPI(
    title="Multilingual NLI API",
    version="1.0.0",
    description="Production-grade API serving XLM-RoBERTa fine-tuned for XNLI, served via ONNX Runtime.",
    lifespan=lifespan
)

# Label mapping
# Standard XNLI mapping: 0 -> entailment, 1 -> neutral, 2 -> contradiction
LABEL_MAP = {
    0: "entailment",
    1: "neutral",
    2: "contradiction"
}

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """
    Handle validation errors, returning a clean 422 HTTP response.
    """
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={"detail": exc.errors()}
    )

@app.post("/classify", response_model=ClassifyResponse)
async def classify(request: ClassifyRequest):
    if not request.pairs:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="The 'pairs' list cannot be empty."
        )
        
    tokenizer = app.state.tokenizer
    session = app.state.session
    max_length = app.state.max_length
    
    # Extract premises and hypotheses
    premises = [p.premise for p in request.pairs]
    hypotheses = [p.hypothesis for p in request.pairs]
    
    try:
        # Tokenize dynamically batched inputs
        inputs = tokenizer(
            premises,
            hypotheses,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="np"
        )
        
        # Prepare inputs for ONNX Runtime with explicit int64 cast
        ort_inputs = {
            "input_ids": inputs["input_ids"].astype(np.int64),
            "attention_mask": inputs["attention_mask"].astype(np.int64)
        }
        
        # Execute forward pass
        ort_outs = session.run(None, ort_inputs)
        logits = ort_outs[0] # Shape: [batch_size, 3]
        
        # Compute Softmax probabilities
        exp_logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
        probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
        
        # Formulate predictions
        predictions = []
        for i in range(len(request.pairs)):
            prob = probs[i]
            pred_idx = int(np.argmax(prob))
            confidence = float(prob[pred_idx])
            pred_label = LABEL_MAP.get(pred_idx, "neutral")
            
            predictions.append(Prediction(
                premise=premises[i],
                hypothesis=hypotheses[i],
                label=pred_label,
                confidence=confidence
            ))
            
        return ClassifyResponse(predictions=predictions)
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference error: {str(e)}"
        )

@app.get("/health")
async def health():
    return {"status": "healthy"}
