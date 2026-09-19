# Experiment 1 — Temporal Model Comparison

## Objective
Evaluate whether the causal Gated Recurrent Unit (GRU) temporal mechanism provides useful temporal modeling compared with alternative lightweight temporal architectures (Causal LSTM and Lightweight Causal Dilated TCN) when the spatial CNN+GNN representation (128-dimensional embedding, 8-window history = 22.5s) is fixed.

## Scientific Protocol & Constraints
- **Representation**: Precomputed 128-dimensional spatial embeddings from the frozen CNN + Spatial GNN backbone ($\\theta=0.30$ graph, 52,497 parameters).
- **Temporal Input**: $8 \\times 128$ sequences (22.5 seconds causal span).
- **Patient Partitions**:
  - TRAIN (16 patients): chb04, chb09, chb11–chb24 (901,391 windows)
  - VALIDATION (4 patients): chb06, chb07, chb08, chb10 (293,410 windows)
  - TEST (4 patients): chb01, chb02, chb03, chb05 (219,909 windows, 22 seizures) — LOCKED until model selection finalized.
- **Training Protocol**:
  - Batch size: 4
  - Epochs: 3
  - Optimizer: AdamW ($lr=1e-3, weight\\_decay=1e-4$)
  - Scheduler: CosineAnnealingLR ($T_{\\max}=3, \\eta_{\\min}=1e-5$)
  - Loss: Binary Focal Loss ($\\gamma=2.0, \\alpha=0.25$)
  - Negative Subsampling: 10:1 ratio with deterministic seed $42 + \\text{epoch}$
  - Hardware: Apple Silicon M4 (PyTorch MPS)
  - Memory Management: Memory-mapped embeddings and bounded chunk evaluation (4096-window chunks), strictly enforcing Peak RSS $< 4.0$ GB.

## Summary Results Table

| Metric | Reference GRU | Causal LSTM | Causal TCN | Model C (Frozen Reference) |
|---|---|---|---|---|
| **Trainable Parameters** | **39,361** | 51,777 | 60,417 | **39,361** |
| **Total Parameters** | **91,858** | 104,274 | 112,914 | **91,858** |
| **Val AUPRC (Selection)** | **0.39365** | 0.35615 | 0.27008 | 0.42002 |
| **Val AUROC** | 0.89046 | **0.91902** | 0.90238 | 0.90414 |
| **Test AUROC** | 0.98340 | **0.98641** | 0.98067 | **0.98970** |
| **Test AUPRC** | **0.75450** | 0.74638 | 0.71582 | **0.80681** |
| **Test Sensitivity** | 76.77% | 82.10% | **87.60%** | 83.83% |
| **Test Specificity** | **99.88%** | 99.73% | 99.64% | 99.82% |
| **Test Precision** | **64.17%** | 46.53% | 41.39% | 57.24% |
| **Test F1 Score** | **0.69907** | 0.59398 | 0.56222 | 0.68025 |
| **Test Balanced Accuracy** | 88.32% | 90.92% | **93.62%** | 91.82% |
| **Event Sensitivity** | 21/22 (95.45%) | 21/22 (95.45%) | **22/22 (100.0%)** | 21/22 (95.45%) |
| **Missed Seizures** | 1 | 1 | **0** | 1 |
| **Mean Detection Delay** | 11.88s | **10.45s** | 10.68s | 10.57s |
| **Median Detection Delay** | 9.00s | **8.00s** | **8.00s** | 9.00s |
| **False Alarms Total** | **273** | 601 | 790 | 399 |
| **False Alarms / 24 Hours** | **42.87** | 94.38 | 124.07 | 62.66 |
| **Peak RSS** | 2677.6 MB | 2676.8 MB | 2688.0 MB | 1420.0 MB |
| **Inference Latency** | 0.462 ms | **0.153 ms** | 0.465 ms | 0.010 ms |
| **Training Time** | 151.2s | **134.3s** | 159.7s | 145.0s |

## Key Findings
1. **Precision & False Alarm Control**: The GRU achieved the highest test precision (64.17%) and lowest false alarm rate (42.87 FA/24h vs 94.38 for LSTM and 124.07 for TCN), making it the most clinically viable temporal model under clinical false alarm constraints.
2. **Sensitivity Trade-off**: Causal TCN achieved 100% event sensitivity (0 missed seizures) and 87.60% window sensitivity, but at the cost of almost $3\\times$ higher false alarms (124.07/24h) and lowest test AUPRC (0.71582).
3. **Model Complexity**: GRU operates with the smallest parameter footprint (39,361 trainable params), compared to 51,777 for LSTM (+31.5%) and 60,417 for TCN (+53.5%).
4. **Validation-Driven Selection Alignment**: Validation AUPRC correctly ranked GRU > LSTM > TCN, matching the final test AUPRC ordering.
