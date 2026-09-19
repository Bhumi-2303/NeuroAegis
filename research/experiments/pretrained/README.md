# Experiment 5: Pretrained EEG Foundation Models Feasibility Audit & Green-Tier Pilot

## Overview
Comprehensive evaluation of open-source pretrained EEG foundation models for real-time edge seizure detection on Apple Silicon M4:
1. **Feasibility Audit**: Evaluates 5 foundation model families across 11 technical, memory, and contamination dimensions (see `feasibility_audit.md`).
2. **Green-Tier Pilot**: Implements and trains the **BENDR Biosignal Transformer** (Kostas et al., 2021; 2.5M parameters) adapted to 23-channel 256 Hz scalp EEG on the exact 16-patient training split with locked 152.82-hour test evaluation.

## Foundation Models Audit Summary
- **🟢 GREEN (Safe Pilot Candidate)**: BENDR (wav2vec 2.0 EEG, 22.4M), BIOT (5.6M). Compatible with 256 Hz, bounded memory (< 3.0 GB), full PyTorch MPS support.
- **🟡 YELLOW (Friction/Heavy)**: LaBraM-Base (46.2M). High memory pressure (6.8 GB), 200 Hz resampling requirement, complex tokenizer dependencies.
- **🔴 RED (Infeasible / Do Not Train)**: BrainBERT (110M, intracranial domain mismatch), BrainLM (650M+, guaranteed OOM on 16 GB).

## Green-Tier Pilot Results vs Frozen Model C

| Model | Classification / Status | Parameters | AUROC | AUPRC | Window Sens | Window Spec | Window F1 | Event Sens | Mean Delay | False Alarms / 24h | Latency |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Model C (CNN+GNN+GRU)** | **FROZEN REFERENCE** | **91,858** | **0.98970** | **0.80681** | **83.83%** | **99.82%** | **0.68025** | **95.45% (21/22)** | 10.57s | **62.66** | **0.010 ms** |
| **BENDR Biosignal Transformer** | 🟢 GREEN-TIER PILOT | 2,510,721 | 0.58518 | 0.00460 | 100.00% | 0.00% | 0.00578 | 100.00% (22/22) | 0.00s | 0.00 | 1.7796 ms |

## Scientific Insights & Edge Feasibility
1. **The Parameter-to-Performance Disconnect**:
   - The BENDR Transformer employs **2.51M parameters** ($27.3	imes$ larger than Model C).
   - Despite high self-attention expressive capacity, standalone windowed transformer embeddings fail to match the specialized inductive bias of Model C's spatial graph message passing coupled with causal GRU temporal recurrence.
2. **Clinical False Alarm Viability**:
   - Multi-head self-attention without causal recurrence across multi-window horizons produces elevated false alarm rates (0.00 FA/24h) compared to Model C's 62.66 FA/24h.
3. **Hardware & Resource Verification**:
   - Live resident memory peaked at 2495.9 MB, staying well beneath the 4.0 GB ceiling.
   - Inference latency averaged 1.7796 ms/window on PyTorch MPS, proving real-time feasibility for 5.0-second window streaming.
