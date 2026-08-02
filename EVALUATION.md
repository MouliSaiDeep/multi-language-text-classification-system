> **⚠️ Smoke-test run.** These metrics come from `--smoke_test` (a tiny subset, ~1 epoch) used to validate the pipeline end-to-end. They are not representative of real model performance and are expected to fall short of the target thresholds below. Rerun `python scripts/train.py --epochs <N>` and `python scripts/evaluate.py` (both without `--smoke_test`) to produce a real evaluation before treating these numbers as final.

# Model Evaluation Report

This report summarizes the evaluation metrics for the cross-lingual NLI model.

## Test Metrics per Language

| Language | Code | Precision | Recall | F1-Score |
| --- | --- | --- | --- | --- |
| EN | en | 0.4561 | 0.3810 | 0.2628 |
| FR | fr | 0.1167 | 0.3333 | 0.1728 |
| ES | es | 0.4561 | 0.3810 | 0.2628 |
| DE | de | 0.2778 | 0.3333 | 0.2341 |
| ZH | zh | 0.3595 | 0.4286 | 0.3278 |
| RU | ru | 0.1167 | 0.3333 | 0.1728 |
| HI | hi | 0.1167 | 0.3333 | 0.1728 |
| VI | vi | 0.1053 | 0.2857 | 0.1538 |
| AR | ar | 0.3056 | 0.4286 | 0.3532 |

## Zero-Shot Evaluation (Held-out Language)

| Language | Code | Precision | Recall | F1-Score |
| --- | --- | --- | --- | --- |
| SWAHILI | sw | 0.2917 | 0.3810 | 0.2951 |

## Performance & Latency Comparison

| Framework | Average Latency (ms) | Speedup |
| --- | --- | --- |
| PyTorch (FP32) | 148.81 ms | 1.0x |
| ONNX Runtime | 130.37 ms | 1.14x |

## Summary Metrics

- **Macro F1 (9 training languages)**: 0.2348
- **Zero-shot F1 (Swahili)**: 0.2951
- **Target Metrics**: macro_f1 >= 0.75, zero_shot_f1 >= 0.60
