# NeuroAegis — Channel-Independent Attention Pooling Training Module

## Architecture Overview

```
┌────────────────────────────────────────────────────────────────────┐
│              AttentionSeizureDetector (end-to-end)                 │
│                                                                    │
│  Stage 1: ChannelEncoder (shared weights)                         │
│  ┌──────────┐  ┌──────────┐       ┌──────────┐                   │
│  │ Ch0 [57] │  │ Ch1 [57] │  ...  │ ChN [57] │   ← per-channel  │
│  │ Linear→  │  │ Linear→  │       │ Linear→  │     57-dim feats  │
│  │ ReLU→    │  │ ReLU→    │       │ ReLU→    │                   │
│  │ Drop→    │  │ Drop→    │       │ Drop→    │                   │
│  │ Linear   │  │ Linear   │       │ Linear   │                   │
│  └────┬─────┘  └────┬─────┘       └────┬─────┘                   │
│       │e_0 [32]      │e_1 [32]         │e_N [32]                 │
│       └──────────────┼─────────────────┘                          │
│                      │                                             │
│  Stage 2: AttentionPooling (permutation-invariant)                │
│  ┌───────────────────┴────────────────────┐                       │
│  │  Scoring MLP: Linear→Tanh→Linear → a_i│                       │
│  │  Masked Softmax → α_i                 │                       │
│  │  z = Σ α_i · e_i                      │                       │
│  └───────────────────┬────────────────────┘                       │
│                      │z [32]                                       │
│  Stage 3: SeizureClassifier                                       │
│  ┌───────────────────┴────────────────────┐                       │
│  │  Linear→ReLU→Drop→Linear→Sigmoid → p  │                       │
│  └────────────────────────────────────────┘                       │
│                      │                                             │
│                 p ∈ [0, 1] (seizure probability)                  │
└────────────────────────────────────────────────────────────────────┘
```

**Key properties:**
- **Channel-count agnostic**: works with 1, 8, or 23 channels via padding + masking.
- **Weight sharing**: identical encoder applied independently to every channel.
- **Lightweight**: ~2,500 trainable parameters (encoder + pooling + classifier).
- **Interpretable**: attention weights α reveal which channels drive the prediction.

## Directory Structure

```
apps/api/training/
├── __init__.py          # Package init
├── model.py             # ChannelEncoder, AttentionPooling, SeizureClassifier, AttentionSeizureDetector
├── data.py              # MultiChannelEEGDataset, collate_fn, get_patient_splits, FeatureNormalizer
├── train.py             # LOPO-CV training loop with entropy regularization
├── evaluate.py          # Metrics, ablation baselines, attention visualization
├── test_model.py        # Unit tests for architecture shapes and invariants
└── README.md            # This file

scripts/
└── extract_multichannel.py  # Downloads CHB-MIT EDF, extracts 23-channel features
```

## Prerequisites

1. **Python environment** with the project's venv activated:
   ```bash
   source .venv/bin/activate
   ```

2. **Install PyTorch** (CPU-only is sufficient):
   ```bash
   pip install torch matplotlib
   ```

3. **Existing dependencies** (already in `requirements.txt`):
   - numpy, pandas, scipy, scikit-learn, mne, pywt, lightgbm, shap, joblib

## Step 1: Extract Multi-Channel Features

Download CHB-MIT EDF files from PhysioNet and extract 57 features × 23 channels:

```bash
# From the repository root
python scripts/extract_multichannel.py --patients chb01 chb02 chb03 chb04 chb05
```

This produces:
- `data/chbmit_multichannel.parquet` — multi-channel feature matrix
- `data/bonn_features.csv` — Bonn single-channel features

**Note**: Downloads ~500 MB of EDF files (deleted after extraction to save space).
Extraction takes ~15–30 minutes depending on network and CPU.

## Step 2: Train with LOPO Cross-Validation

```bash
cd apps/api

# Full attention model (default settings)
python -m training.train \
    --data ../../data/chbmit_multichannel.parquet \
    --output models/chbmit/attention_pooling/ \
    --epochs 100 \
    --batch-size 32 \
    --lr 1e-3 \
    --lambda-entropy 0.01 \
    --patience 15

# With focal loss (for severe class imbalance)
python -m training.train \
    --data ../../data/chbmit_multichannel.parquet \
    --output models/chbmit/attention_pooling_focal/ \
    --focal

# Ablation: no entropy regularization
python -m training.train \
    --data ../../data/chbmit_multichannel.parquet \
    --output models/chbmit/attention_no_entropy/ \
    --lambda-entropy 0.0
```

Output per fold:
- `model_fold_{i}.pt` — best model checkpoint
- `history_fold_{i}.json` — per-epoch training logs
- `cv_summary.json` — aggregate LOPO-CV results

## Step 3: Evaluate & Run Ablations

```bash
cd apps/api

# (a) Average voting baseline
python -m training.evaluate \
    --data ../../data/chbmit_multichannel.parquet \
    --model-dir models/chbmit/attention_pooling/ \
    --output results/voting/ \
    --mode voting \
    --lgbm-model models/chbmit/lightgbm_patient_wise.pkl

# (b) Best single channel
python -m training.evaluate \
    --data ../../data/chbmit_multichannel.parquet \
    --model-dir models/chbmit/attention_pooling/ \
    --output results/best_channel/ \
    --mode best-channel \
    --lgbm-model models/chbmit/lightgbm_patient_wise.pkl

# (c) Concat baseline (23×57 = 1311 features → MLP)
python -m training.evaluate \
    --data ../../data/chbmit_multichannel.parquet \
    --model-dir models/chbmit/attention_pooling/ \
    --output results/concat/ \
    --mode concat

# (d) Full attention model
python -m training.evaluate \
    --data ../../data/chbmit_multichannel.parquet \
    --model-dir models/chbmit/attention_pooling/ \
    --output results/attention/ \
    --mode attention

# (e) Ablation: no entropy regularization
python -m training.evaluate \
    --data ../../data/chbmit_multichannel.parquet \
    --model-dir models/chbmit/attention_no_entropy/ \
    --output results/no_entropy/ \
    --mode no-entropy

# (f) Ablation: no channel augmentation
python -m training.evaluate \
    --data ../../data/chbmit_multichannel.parquet \
    --model-dir models/chbmit/attention_no_augment/ \
    --output results/no_augment/ \
    --mode no-augment
```

## Step 4: Run Unit Tests

```bash
cd apps/api
python -m pytest training/test_model.py -v
```

## Zero-Shot Bonn → CHB-MIT Experiment

To evaluate domain generalization from Bonn (single-channel, 173.61 Hz) to CHB-MIT (23-channel, 256 Hz):

1. Train on Bonn features using the existing single-channel pipeline.
2. At inference time, the attention model processes each CHB-MIT channel independently through the same feature extraction pipeline.
3. The attention layer automatically discovers which channels are most informative.

This is built into the `evaluate.py --mode voting` baseline — it runs the Bonn-trained LightGBM on every CHB-MIT channel independently and averages the predictions.

## Metrics Reported

| Metric | Description |
|---|---|
| AUROC | Area Under Receiver Operating Characteristic curve |
| AUPRC | Area Under Precision-Recall Curve (more informative under class imbalance) |
| Sensitivity @ 1 FP/hr | Seizure detection rate at ≤1 false alarm per hour |
| Sensitivity @ 0.25 FP/hr | Seizure detection rate at ≤0.25 false alarms per hour |
| Per-patient breakdown | Individual patient metrics (box plots + table) |
| Attention heatmap | Channel-wise α weights for true positive seizure detections |

## Loss Function

Total loss combines classification and regularization:

$$L = L_{BCE} - \lambda \cdot L_{entropy}$$

Where:
- $L_{BCE}$: Binary cross-entropy with class weighting (or focal loss)
- $L_{entropy} = -\sum_i \alpha_i \log \alpha_i$: Attention entropy (higher = more distributed)
- $\lambda$: Entropy regularization weight (default 0.01)

Subtracting entropy encourages the model to **attend to multiple channels** rather than collapsing onto a single dominant one.
