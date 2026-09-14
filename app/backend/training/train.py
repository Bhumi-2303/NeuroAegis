import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import roc_auc_score, average_precision_score

import sys
from pathlib import Path

TRAINING_DIR = Path(__file__).resolve().parent
API_DIR = TRAINING_DIR.parent
if str(API_DIR) not in sys.path:
    sys.path.insert(0, str(API_DIR))

from training.model import AttentionSeizureDetector
from training.data import MultiChannelEEGDataset, collate_fn, get_patient_splits

def set_seed(seed: int):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

class FocalLoss(nn.Module):
    """Focal Loss for imbalanced binary classification (assumes inputs are probabilities)."""
    def __init__(self, alpha: float = None, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.bce = nn.BCELoss(reduction='none')

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce_loss = self.bce(inputs, targets)
        pt = torch.exp(-bce_loss)  # pt is the probability of the true class
        focal_loss = ((1 - pt) ** self.gamma) * bce_loss
        
        if self.alpha is not None:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            focal_loss = alpha_t * focal_loss
            
        return focal_loss.mean()

def train_one_fold(args, fold: int, train_idx, val_idx, train_dataset, val_dataset, device):
    """Trains the model on a single LOPO fold."""
    print(f"\n{'='*40}")
    print(f"Starting Fold {fold}")
    print(f"Train samples: {len(train_idx)}, Val samples: {len(val_idx)}")
    print(f"{'='*40}")

    # Create Subsets
    train_sub = Subset(train_dataset, train_idx)
    val_sub = Subset(val_dataset, val_idx)

    # DataLoaders
    train_loader = DataLoader(train_sub, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn, num_workers=0, pin_memory=False)
    val_loader = DataLoader(val_sub, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn, num_workers=0, pin_memory=False)

    # Calculate class weights for this fold
    train_labels = [train_dataset.labels[i] for i in train_idx]
    num_pos = sum(train_labels)
    num_neg = len(train_labels) - num_pos
    pos_weight = float(num_neg) / max(float(num_pos), 1.0)
    print(f"Computed pos_weight for BCE: {pos_weight:.2f}")

    # Model, Optimizer, Scheduler
    # The dataset features shape is [N_channels, 57]
    input_dim = train_dataset.n_features
    model = AttentionSeizureDetector(input_dim=input_dim, embed_dim=args.embed_dim).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)

    if args.focal:
        criterion = FocalLoss(gamma=2.0)
    else:
        criterion = nn.BCELoss(reduction='none')

    best_val_auprc = -1.0
    best_metrics = {}
    epochs_no_improve = 0
    best_model_path = os.path.join(args.output, f"model_fold_{fold}.pt")
    
    fold_history = []

    for epoch in range(1, args.epochs + 1):
        # Training Phase
        model.train()
        train_losses = []
        
        for batch in train_loader:
            features = batch['features'].to(device)
            mask = batch['mask'].to(device)
            labels = batch['labels'].to(device)
            
            optimizer.zero_grad()
            
            preds, alphas = model(features, mask)
            
            # Base classification loss
            if args.focal:
                cls_loss = criterion(preds, labels)
            else:
                bce_out = criterion(preds, labels)
                weights = torch.where(labels == 1.0, torch.tensor(pos_weight, device=device), torch.tensor(1.0, device=device))
                cls_loss = (bce_out * weights).mean()
                
            # Entropy regularization (model.compute_entropy_loss returns positive entropy, 
            # we subtract lambda * entropy in total loss, meaning we *add* -lambda * entropy?
            # Wait, maximizing entropy means loss = cls_loss - lambda * entropy. 
            # BUT the prompt says: Total: L = L_BCE - lambda * L_entropy.
            entropy = model.compute_entropy_loss(alphas, mask)
            total_loss = cls_loss - args.lambda_entropy * entropy
            
            total_loss.backward()
            optimizer.step()
            
            train_losses.append(total_loss.item())
            
        avg_train_loss = np.mean(train_losses)

        # Validation Phase
        model.eval()
        val_losses = []
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in val_loader:
                features = batch['features'].to(device)
                mask = batch['mask'].to(device)
                labels = batch['labels'].to(device)
                
                preds, alphas = model(features, mask)
                
                if args.focal:
                    cls_loss = criterion(preds, labels)
                else:
                    bce_out = criterion(preds, labels)
                    weights = torch.where(labels == 1.0, torch.tensor(pos_weight, device=device), torch.tensor(1.0, device=device))
                    cls_loss = (bce_out * weights).mean()
                    
                entropy = model.compute_entropy_loss(alphas, mask)
                total_loss = cls_loss - args.lambda_entropy * entropy
                
                val_losses.append(total_loss.item())
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
        avg_val_loss = np.mean(val_losses)
        all_preds = np.array(all_preds).flatten()
        all_labels = np.array(all_labels).flatten()
        
        # Calculate Metrics
        try:
            val_auroc = roc_auc_score(all_labels, all_preds)
            val_auprc = average_precision_score(all_labels, all_preds)
        except ValueError:
            # In case validation set only has one class
            val_auroc = 0.0
            val_auprc = 0.0
            
        print(f"Epoch {epoch:03d} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | "
              f"Val AUROC: {val_auroc:.4f} | Val AUPRC: {val_auprc:.4f}")
              
        fold_history.append({
            'epoch': epoch,
            'train_loss': float(avg_train_loss),
            'val_loss': float(avg_val_loss),
            'val_auroc': float(val_auroc),
            'val_auprc': float(val_auprc)
        })
        
        scheduler.step(val_auprc)

        # Early stopping & Best model saving
        if val_auprc > best_val_auprc:
            best_val_auprc = val_auprc
            epochs_no_improve = 0
            best_metrics = fold_history[-1]
            torch.save(model.state_dict(), best_model_path)
            print(f"  [*] New best model saved (AUPRC: {val_auprc:.4f})")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= args.patience:
                print(f"Early stopping triggered after {epoch} epochs.")
                break
                
    return best_metrics, fold_history

def main():
    parser = argparse.ArgumentParser(description="Train NeuroAegis Attention Pooling Model")
    parser.add_argument("--data", type=str, required=True, help="Path to multi-channel parquet file")
    parser.add_argument("--output", type=str, default="apps/api/models/chbmit/attention_pooling/", help="Directory for model checkpoints")
    parser.add_argument("--epochs", type=int, default=100, help="Maximum epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--lambda-entropy", type=float, default=0.01, help="Entropy regularization weight")
    parser.add_argument("--focal", action="store_true", help="Use focal loss instead of BCE")
    parser.add_argument("--patience", type=int, default=15, help="Early stopping patience")
    parser.add_argument("--embed-dim", type=int, default=32, help="Embedding dimension")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    set_seed(args.seed)
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu'))
    print(f"Using device: {device}")
    
    # Create datasets
    print(f"Loading data from {args.data}...")
    train_dataset = MultiChannelEEGDataset(args.data, augment=True, normalize=True)
    val_dataset = MultiChannelEEGDataset(args.data, augment=False, normalize=True)
    
    # We must ensure normalizers are identical. The simplest way is to copy the fitted normalizer.
    # val_dataset should use the stats computed on the whole dataset, which it does by default when normalize=True.
    
    # Get LOPO splits
    splits = get_patient_splits(val_dataset)
    print(f"Found {len(splits)} patients for LOPO CV.")
    
    all_best_metrics = []
    
    for fold, (train_idx, val_idx) in enumerate(splits):
        best_metrics, history = train_one_fold(
            args=args, 
            fold=fold, 
            train_idx=train_idx, 
            val_idx=val_idx, 
            train_dataset=train_dataset, 
            val_dataset=val_dataset,
            device=device
        )
        
        best_metrics['fold'] = fold
        all_best_metrics.append(best_metrics)
        
        # Save history for this fold
        with open(os.path.join(args.output, f"history_fold_{fold}.json"), "w") as f:
            json.dump(history, f, indent=4)
            
    # Print summary
    print("\n" + "="*50)
    print("TRAINING SUMMARY (Best Metrics per Fold)")
    print("="*50)
    print(f"{'Fold':<6} | {'Epoch':<6} | {'Val AUROC':<12} | {'Val AUPRC':<12}")
    print("-" * 50)
    
    auroc_list = []
    auprc_list = []
    
    for metrics in all_best_metrics:
        print(f"{metrics['fold']:<6} | {metrics['epoch']:<6} | {metrics['val_auroc']:<12.4f} | {metrics['val_auprc']:<12.4f}")
        auroc_list.append(metrics['val_auroc'])
        auprc_list.append(metrics['val_auprc'])
        
    print("-" * 50)
    print(f"{'MEAN':<15} | {np.mean(auroc_list):<12.4f} | {np.mean(auprc_list):<12.4f}")
    print(f"{'STD':<15} | {np.std(auroc_list):<12.4f} | {np.std(auprc_list):<12.4f}")
    
    # Save summary
    summary_path = os.path.join(args.output, "cv_summary.json")
    with open(summary_path, "w") as f:
        json.dump({
            "args": vars(args),
            "folds": all_best_metrics,
            "mean_auroc": float(np.mean(auroc_list)),
            "std_auroc": float(np.std(auroc_list)),
            "mean_auprc": float(np.mean(auprc_list)),
            "std_auprc": float(np.std(auprc_list))
        }, f, indent=4)
    print(f"\nSaved summary to {summary_path}")

if __name__ == "__main__":
    main()
