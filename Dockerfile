FROM python:3.10-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    MODEL_NAME=xlm-roberta-base \
    MODEL_MAX_LENGTH=128 \
    BATCH_SIZE=32 \
    ONNX_MODEL_PATH=src/api/model.onnx \
    MLFLOW_TRACKING_URI=./mlruns

# Set work directory
WORKDIR /app

# Install system dependencies if required
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code and config
COPY src/ /app/src/

# We will mount or copy the ONNX model artifact
# Ensure target folder exists
RUN mkdir -p /app/src/api

# Expose port
EXPOSE 8000

# Run uvicorn server
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
