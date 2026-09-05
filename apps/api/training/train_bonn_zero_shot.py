#!/usr/bin/env python3
"""
train_bonn_zero_shot.py
───────────────────────
Trains the AttentionSeizureDetector on the Bonn dataset (single-channel, 173.61 Hz)
with synthetic channel dropout, and evaluates zero-shot (no fine-tuning) on the
full 23-patient CHB-MIT dataset (multi-channel, 256 Hz).

Output:
  - models/bonn_zero_shot/model_bonn.pt
  - results/zero_shot/evaluation_report_zero_shot.json
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import roc_auc_score, average_precision_score

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "apps" / "api"))

from training.model import AttentionSeizureDetector
from training.data import MultiChannelEEGDataset, collate_fn, FeatureNormalizer
from training.evaluate import calculate_sensitivity_at_fp_rate


class BonnDataset(Dataset):
    """
    Dataset wrapper for single-channel Bonn features.csv.
    Standardizes feature names and expands to [1, 57] shape.
    """
    def __init__(self, csv_path: str, normalize: bool = True):
        self.df = pd.read_csv(csv_path)
        
        # Binary target: Set S (label 4) = Seizure (1), all other sets (0, 1, 2, 3) = Non-Seizure (0)
        if 'target' in self.df.columns:
            self.labels = self.df['target'].values
        elif 'label' in self.df.columns:
            self.labels = (self.df['label'] == 4).astype(int).values
        else:
            raise ValueError("No label or target column found in Bonn CSV")
            
        meta_cols = {'label', 'target', 'epoch_id', 'set', 'patient_id'}
        self.feature_names = [c for c in self.df.columns if c not in meta_cols]
        
        self.data = self.df[self.feature_names].values.astype(np.float32)
        
        if normalize:
            self.normalizer = FeatureNormalizer()
            self.data = self.normalizer.fit_transform(self.data)
            
    def __len__(self):
        return len(self.df)
        
    def __getitem__(self, idx):
        # Shape [1, N_features] representing single channel
        feat = self.data[idx].reshape(1, -1)
        mask = torch.ones(1, dtype=torch.bool)
        
        return {
            'features': torch.tensor(feat, dtype=torch.float32),
            'label': int(self.labels[idx]),
            'patient_id': f"bonn_{idx}",
            'mask': mask,
            'n_channels': 1
        }


def train_bonn_model(args, device):
    """Train AttentionSeizureDetector on Bonn single-channel dataset."""
    print("=" * 70)
    print("Training AttentionSeizureDetector on Bonn Dataset")
    print("=" * 70)

    dataset = BonnDataset(args.bonn_data, normalize=True)
    input_dim = dataset.data.shape[1]
    
    # 80/20 train/val split
    n_samples = len(dataset)
    indices = np.random.permutation(n_samples)
    split_idx = int(0.8 * n_samples)
    train_idx, val_idx = indices[:split_idx], indices[split_idx:]
    
    train_sub = torch.utils.data.Subset(dataset, train_idx)
    val_sub = torch.utils.data.Subset(dataset, val_idx)
    
    train_loader = DataLoader(train_sub, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_sub, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
    
    model = AttentionSeizureDetector(input_dim=input_dim, embed_dim=args.embed_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    
    # Class weights
    labels_train = dataset.labels[train_idx]
    n_pos = sum(labels_train)
    n_neg = len(labels_train) - n_pos
    pos_weight = float(n_neg) / max(float(n_pos), 1.0)
    criterion = nn.BCELoss(reduction='none')
    
    best_auprc = -1.0
    best_model_path = Path(args.output) / "model_bonn.pt"
    best_model_path.parent.mkdir(parents=True, exist_ok=True)
    
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_losses = []
        for batch in train_loader:
            features = batch['features'].to(device)
            mask = batch['mask'].to(device)
            labels = batch['labels'].to(device)
            
            optimizer.zero_grad()
            preds, alphas = model(features, mask)
            
            bce_out = criterion(preds, labels)
            weights = torch.where(labels == 1.0, torch.tensor(pos_weight, device=device), torch.tensor(1.0, device=device))
            cls_loss = (bce_out * weights).mean()
            
            entropy = model.compute_entropy_loss(alphas, mask)
            total_loss = cls_loss - args.lambda_entropy * entropy
            
            total_loss.backward()
            optimizer.step()
            train_losses.append(total_loss.item())
            
        # Validation
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for batch in val_loader:
                features = batch['features'].to(device)
                mask = batch['mask'].to(device)
                labels = batch['labels'].numpy()
                preds, _ = model(features, mask)
                all_preds.extend(preds.cpu().numpy().flatten())
                all_labels.extend(labels.flatten())
                
        val_auroc = roc_auc_score(all_labels, all_preds)
        val_auprc = average_precision_score(all_labels, all_preds)
        
        if val_auprc > best_auprc:
            best_auprc = val_auprc
            torch.save(model.state_dict(), best_model_path)
            
        print(f"Epoch {epoch:03d} | Train Loss: {np.mean(train_losses):.4f} | Val AUROC: {val_auroc:.4f} | Val AUPRC: {val_auprc:.4f}")
        
    print(f"\n✓ Saved best Bonn model (AUPRC: {best_auprc:.4f}) to {best_model_path}")
    return best_model_path, dataset.normalizer, dataset.feature_names


def evaluate_zero_shot(args, model_path, bonn_normalizer, device, bonn_feature_names):
    """Evaluate Bonn-trained model zero-shot on full CHB-MIT dataset."""
    print("\n" + "=" * 70)
    print("Zero-Shot Cross-Dataset Evaluation (Bonn → CHB-MIT Cohort)")
    print("=" * 70)
    
    # Load CHB-MIT dataset using exact Bonn feature names
    chb_dataset = MultiChannelEEGDataset(args.chbmit_data, augment=False, normalize=True, selected_features=bonn_feature_names)
    n_features = chb_dataset.n_features
    
    model = AttentionSeizureDetector(input_dim=n_features, embed_dim=args.embed_dim).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    loader = DataLoader(chb_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)
    
    all_preds = []
    all_labels = []
    patient_preds = {}
    patient_labels = {}
    
    with torch.no_grad():
        for batch in loader:
            features = batch['features'].to(device)
            mask = batch['mask'].to(device)
            labels = batch['labels'].numpy().flatten()
            patient_ids = batch['patient_ids']
            
            preds, alphas = model(features, mask)
            preds_np = preds.cpu().numpy().flatten()
            
            all_preds.extend(preds_np)
            all_labels.extend(labels)
            
            for pid, p_val, l_val in zip(patient_ids, preds_np, labels):
                patient_preds.setdefault(pid, []).append(p_val)
                patient_labels.setdefault(pid, []).append(l_val)
                
    # Per-patient zero-shot metrics
    per_patient_metrics = []
    for pid in sorted(patient_preds.keys()):
        p_y = np.array(patient_labels[pid])
        p_scores = np.array(patient_preds[pid])
        
        if len(np.unique(p_y)) < 2:
            auroc = 0.5
            auprc = 0.0
        else:
            auroc = roc_auc_score(p_y, p_scores)
            auprc = average_precision_score(p_y, p_scores)
            
        sens_1fp = calculate_sensitivity_at_fp_rate(p_y, p_scores, target_fp_per_hour=1.0)
        sens_025fp = calculate_sensitivity_at_fp_rate(p_y, p_scores, target_fp_per_hour=0.25)
        
        per_patient_metrics.append({
            "patient_id": pid,
            "windows": len(p_y),
            "seizures": int(sum(p_y)),
            "auroc": float(auroc),
            "auprc": float(auprc),
            "sens_1fp_hr": float(sens_1fp),
            "sens_025fp_hr": float(sens_025fp)
        })
        
    overall_auroc = roc_auc_score(all_labels, all_preds)
    overall_auprc = average_precision_score(all_labels, all_preds)
    overall_sens_1fp = calculate_sensitivity_at_fp_rate(np.array(all_labels), np.array(all_preds), target_fp_per_hour=1.0)
    overall_sens_025fp = calculate_sensitivity_at_fp_rate(np.array(all_labels), np.array(all_preds), target_fp_per_hour=0.25)
    
    print(f"\nOverall Zero-Shot AUROC: {overall_auroc:.4f}")
    print(f"Overall Zero-Shot AUPRC: {overall_auprc:.4f}")
    print(f"Overall Sens @ 1.0 FP/h: {overall_sens_1fp:.4f}")
    print(f"Overall Sens @ 0.25 FP/h: {overall_sens_025fp:.4f}")
    
    summary = {
        "setting": "Cross-Dataset Zero-Shot (Bonn -> CHB-MIT)",
        "overall_metrics": {
            "auroc": float(overall_auroc),
            "auprc": float(overall_auprc),
            "sens_1fp_hr": float(overall_sens_1fp),
            "sens_025fp_hr": float(overall_sens_025fp)
        },
        "per_patient": per_patient_metrics
    }
    
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "evaluation_report_zero_shot.json"
    with open(report_path, "w") as f:
        json.dump(summary, f, indent=4)
        
    print(f"\n✓ Zero-shot evaluation complete. Report saved to: {report_path}")


def main():
    parser = argparse.ArgumentParser(description="Zero-Shot Cross-Dataset Evaluation (Bonn -> CHB-MIT)")
    parser.add_argument("--bonn-data", type=str, default="data/bonn_features.csv")
    parser.add_argument("--chbmit-data", type=str, default="data/chbmit_multichannel.parquet")
    parser.add_argument("--output", type=str, default="results/zero_shot/")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lambda-entropy", type=float, default=0.01, help="Entropy regularization weight")
    parser.add_argument("--embed-dim", type=int, default=32)
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu'))
    print(f"Using device: {device}")
    
    model_path, normalizer, feature_names = train_bonn_model(args, device)
    evaluate_zero_shot(args, model_path, normalizer, device, feature_names)

if __name__ == "__main__":
    main()
