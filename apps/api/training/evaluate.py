import argparse
import json
import os
import pickle
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import roc_auc_score, average_precision_score
import matplotlib.pyplot as plt

# Assuming these are available in PYTHONPATH
import sys
from pathlib import Path

# Add apps/api directory to sys.path so 'training' package is always importable
TRAINING_DIR = Path(__file__).resolve().parent
API_DIR = TRAINING_DIR.parent
if str(API_DIR) not in sys.path:
    sys.path.insert(0, str(API_DIR))

from training.model import AttentionSeizureDetector
from training.data import MultiChannelEEGDataset, collate_fn, get_patient_splits

def calculate_sensitivity_at_fp_rate(labels, scores, target_fp_per_hour=1.0, window_len_sec=60):
    """
    Sweeps thresholds to find sensitivity at a given False Positives per hour rate.
    """
    thresholds = np.sort(np.unique(scores))[::-1]
    
    total_hours = len(labels) * (window_len_sec / 3600.0)
    total_positives = np.sum(labels)
    
    if total_positives == 0:
        return 0.0
        
    best_sens = 0.0
    
    for th in thresholds:
        preds = (scores >= th).astype(int)
        
        tp = np.sum((preds == 1) & (labels == 1))
        fp = np.sum((preds == 1) & (labels == 0))
        
        fp_per_hour = fp / total_hours if total_hours > 0 else 0
        sens = tp / total_positives
        
        if fp_per_hour <= target_fp_per_hour:
            best_sens = sens
        else:
            # FP rate exceeded target, stop sweeping
            break
            
    return best_sens

class ConcatMLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.net(x)

def plot_attention_weights(alpha, channel_names, filepath):
    """Plots attention weights as a bar chart and saves it."""
    plt.figure(figsize=(12, 6))
    plt.bar(channel_names, alpha, color='skyblue')
    plt.xticks(rotation=45, ha='right')
    plt.xlabel('Channel')
    plt.ylabel('Attention Weight (\u03b1)')
    plt.title('Attention Distribution over Channels (True Positive Seizure)')
    plt.tight_layout()
    plt.savefig(filepath)
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Evaluate Seizure Detection Models")
    parser.add_argument("--data", type=str, required=True, help="Path to multi-channel parquet file")
    parser.add_argument("--model-dir", type=str, required=True, help="Directory containing trained model checkpoints")
    parser.add_argument("--output", type=str, required=True, help="Directory for evaluation results")
    parser.add_argument("--mode", type=str, default="attention", 
                        choices=["voting", "best-channel", "concat", "attention", "no-entropy", "no-augment"],
                        help="Evaluation mode")
    parser.add_argument("--lgbm-model", type=str, help="Path to existing LightGBM model .pkl (for voting/best-channel baselines)")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu'))
    
    print(f"Loading data from {args.data}...")
    dataset = MultiChannelEEGDataset(args.data, augment=False, normalize=True)
    splits = get_patient_splits(dataset)
    
    metrics_per_fold = []
    patient_ids_list = dataset.patient_ids
    
    # Pre-load LightGBM if needed
    lgbm_model = None
    if args.mode in ["voting", "best-channel"]:
        if not args.lgbm_model or not os.path.exists(args.lgbm_model):
            raise ValueError(f"--lgbm-model is required and must exist for {args.mode} mode")
        import joblib
        loaded = joblib.load(args.lgbm_model)
        if isinstance(loaded, dict) and 'model' in loaded:
            lgbm_model = loaded['model']
        else:
            lgbm_model = loaded
            
    for fold, (train_idx, val_idx) in enumerate(splits):
        print(f"\nEvaluating Fold {fold}...")
        test_sub = Subset(dataset, val_idx)
        test_loader = DataLoader(test_sub, batch_size=32, shuffle=False, collate_fn=collate_fn)
        
        all_labels = []
        all_scores = []
        
        if args.mode in ["attention", "no-entropy", "no-augment"]:
            model_path = os.path.join(args.model_dir, f"model_fold_{fold}.pt")
            if not os.path.exists(model_path):
                print(f"Warning: Model for fold {fold} not found at {model_path}. Skipping.")
                continue
                
            model = AttentionSeizureDetector(input_dim=dataset.n_features, embed_dim=32).to(device)
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.eval()
            
            plotted_attention = False
            
            with torch.no_grad():
                for batch in test_loader:
                    features = batch['features'].to(device)
                    mask = batch['mask'].to(device)
                    labels = batch['labels'].numpy()
                    
                    preds, alphas = model(features, mask)
                    preds = preds.cpu().numpy()
                    alphas = alphas.cpu().numpy()
                    
                    all_labels.extend(labels.flatten())
                    all_scores.extend(preds.flatten())
                    
                    # Plot attention for the first True Positive found
                    if not plotted_attention:
                        for i in range(len(labels)):
                            if labels[i] == 1 and preds[i] > 0.5:
                                # Found a true positive
                                alpha = alphas[i][:batch['n_channels'][i]]
                                channels = dataset.channel_prefixes[:batch['n_channels'][i]]
                                out_png = os.path.join(args.output, f"attention_fold_{fold}.png")
                                plot_attention_weights(alpha, channels, out_png)
                                plotted_attention = True
                                break
                                
        elif args.mode == "voting":
            # For each sample, run LGBM on each channel and average
            # Features: [N, C, F]
            # Ensure we only pass the features the model was trained on
            lgbm_feats = getattr(lgbm_model, 'feature_name_', None)
            if lgbm_feats is None and hasattr(lgbm_model, 'booster_'):
                lgbm_feats = lgbm_model.booster_.feature_name()
                
            feat_indices = None
            if lgbm_feats:
                # Strip Ch0_ prefix if present
                clean_lgbm_feats = [f.replace('Ch0_', '') for f in lgbm_feats]
                feat_indices = [dataset.feature_names.index(f) for f in clean_lgbm_feats if f in dataset.feature_names]
                
            for idx in val_idx:
                x = dataset.data_array[idx] # [C, F]
                if feat_indices:
                    x = x[:, feat_indices]
                    
                y = dataset.labels[idx]
                
                # predict_proba returns [N_samples, 2]
                probs = lgbm_model.predict_proba(x)[:, 1]
                avg_prob = np.mean(probs)
                
                all_labels.append(y)
                all_scores.append(avg_prob)
                
        elif args.mode == "best-channel":
            # Determine best channel on train_idx
            best_c = 0
            best_auc = 0
            train_labels = dataset.labels[train_idx]
            
            for c in range(dataset.n_channels):
                c_feats = dataset.data_array[train_idx, c, :]
                c_probs = lgbm_model.predict_proba(c_feats)[:, 1]
                try:
                    auc = roc_auc_score(train_labels, c_probs)
                except ValueError:
                    auc = 0
                if auc > best_auc:
                    best_auc = auc
                    best_c = c
            
            # Predict on val
            for idx in val_idx:
                x = dataset.data_array[idx, best_c, :].reshape(1, -1)
                y = dataset.labels[idx]
                prob = lgbm_model.predict_proba(x)[0, 1]
                all_labels.append(y)
                all_scores.append(prob)
                
        elif args.mode == "concat":
            # Train simple MLP inline on concat features
            concat_dim = dataset.n_channels * dataset.n_features
            model = ConcatMLP(concat_dim).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
            criterion = nn.BCELoss()
            
            # Simplified training loop
            model.train()
            train_sub = Subset(dataset, train_idx)
            train_loader = DataLoader(train_sub, batch_size=32, shuffle=True, collate_fn=collate_fn)
            for epoch in range(10): # quick train
                for batch in train_loader:
                    # Flatten channel and feature dims
                    x = batch['features'].reshape(batch['features'].shape[0], -1).to(device)
                    y = batch['labels'].to(device)
                    optimizer.zero_grad()
                    p = model(x)
                    loss = criterion(p, y)
                    loss.backward()
                    optimizer.step()
                    
            # Predict on val
            model.eval()
            with torch.no_grad():
                for batch in test_loader:
                    x = batch['features'].reshape(batch['features'].shape[0], -1).to(device)
                    y = batch['labels'].numpy()
                    p = model(x).cpu().numpy()
                    all_labels.extend(y.flatten())
                    all_scores.extend(p.flatten())
                    
        all_labels = np.array(all_labels)
        all_scores = np.array(all_scores)
        
        if len(np.unique(all_labels)) < 2:
            print(f"Fold {fold} has only one class in validation. Skipping metrics.")
            continue
            
        auroc = roc_auc_score(all_labels, all_scores)
        auprc = average_precision_score(all_labels, all_scores)
        sens_1fp = calculate_sensitivity_at_fp_rate(all_labels, all_scores, target_fp_per_hour=1.0)
        sens_025fp = calculate_sensitivity_at_fp_rate(all_labels, all_scores, target_fp_per_hour=0.25)
        
        patient_id = patient_ids_list[val_idx[0]]
        
        fold_metrics = {
            "fold": fold,
            "patient_id": patient_id,
            "auroc": float(auroc),
            "auprc": float(auprc),
            "sens_1fp_hr": float(sens_1fp),
            "sens_025fp_hr": float(sens_025fp)
        }
        metrics_per_fold.append(fold_metrics)
        
        print(f"Patient {patient_id} | AUROC: {auroc:.4f} | AUPRC: {auprc:.4f} | "
              f"Sens@1FP/h: {sens_1fp:.4f} | Sens@0.25FP/h: {sens_025fp:.4f}")

    if not metrics_per_fold:
        print("No valid folds to summarize.")
        return
        
    print("\n" + "="*80)
    print(f"EVALUATION SUMMARY ({args.mode.upper()})")
    print("="*80)
    print(f"{'Patient':<12} | {'AUROC':<10} | {'AUPRC':<10} | {'Sens@1FP/h':<12} | {'Sens@0.25FP/h':<12}")
    print("-" * 80)
    
    for m in metrics_per_fold:
        print(f"{m['patient_id']:<12} | {m['auroc']:<10.4f} | {m['auprc']:<10.4f} | {m['sens_1fp_hr']:<12.4f} | {m['sens_025fp_hr']:<12.4f}")
        
    print("-" * 80)
    mean_auroc = np.mean([m['auroc'] for m in metrics_per_fold])
    mean_auprc = np.mean([m['auprc'] for m in metrics_per_fold])
    mean_sens_1 = np.mean([m['sens_1fp_hr'] for m in metrics_per_fold])
    mean_sens_025 = np.mean([m['sens_025fp_hr'] for m in metrics_per_fold])
    
    print(f"{'MEAN':<12} | {mean_auroc:<10.4f} | {mean_auprc:<10.4f} | {mean_sens_1:<12.4f} | {mean_sens_025:<12.4f}")
    
    report = {
        "mode": args.mode,
        "mean_metrics": {
            "auroc": float(mean_auroc),
            "auprc": float(mean_auprc),
            "sens_1fp_hr": float(mean_sens_1),
            "sens_025fp_hr": float(mean_sens_025)
        },
        "per_patient": metrics_per_fold
    }
    
    report_path = os.path.join(args.output, f"evaluation_report_{args.mode}.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=4)
        
    print(f"\nEvaluation complete. Results saved to {report_path}")

if __name__ == "__main__":
    main()
