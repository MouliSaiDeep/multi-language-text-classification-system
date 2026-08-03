> **⚠️ Smoke-test run.** These metrics come from `--smoke_test` (a tiny subset, ~1 epoch) used to validate the pipeline end-to-end. They are not representative of real model performance and are expected to fall short of the target thresholds below. Rerun `python scripts/train.py --epochs <N>` and `python scripts/evaluate.py` (both without `--smoke_test`) to produce a real evaluation before treating these numbers as final.

# Model Evaluation Report

This report summarizes the evaluation metrics for the cross-lingual NLI model.

## Test Metrics per Language

| Language | Code | Precision | Recall | F1-Score |
| --- | --- | --- | --- | --- |
| EN | en | 1.0000 | 1.0000 | 1.0000 |
| FR | fr | 1.0000 | 1.0000 | 1.0000 |
| ES | es | 1.0000 | 1.0000 | 1.0000 |
| DE | de | 1.0000 | 1.0000 | 1.0000 |
| ZH | zh | 1.0000 | 1.0000 | 1.0000 |
| RU | ru | 1.0000 | 1.0000 | 1.0000 |
| HI | hi | 1.0000 | 1.0000 | 1.0000 |
| VI | vi | 1.0000 | 1.0000 | 1.0000 |
| AR | ar | 1.0000 | 1.0000 | 1.0000 |

## Zero-Shot Evaluation (Held-out Language)

| Language | Code | Precision | Recall | F1-Score |
| --- | --- | --- | --- | --- |
| SWAHILI | sw | 1.0000 | 1.0000 | 1.0000 |

## Performance & Latency Comparison

| Framework | Average Latency (ms) | Speedup |
| --- | --- | --- |
| PyTorch (FP32) | 86.10 ms | 1.0x |
| ONNX Runtime | 73.52 ms | 1.17x |

## Summary Metrics

- **Macro F1 (9 training languages)**: 1.0000
- **Zero-shot F1 (Swahili)**: 1.0000
- **Target Metrics**: macro_f1 >= 0.75, zero_shot_f1 >= 0.60
