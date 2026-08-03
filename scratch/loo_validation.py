import pandas as pd
import numpy as np
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, precision_recall_curve
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def get_metrics(y_true, y_pred, y_prob):
    prec = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
    rec = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
    f1 = f1_score(y_true, y_pred, pos_label=1, zero_division=0)
    try:
        auc = roc_auc_score(y_true, y_prob)
    except:
        auc = np.nan
    acc = accuracy_score(y_true, y_pred)
    return acc, auc, prec, rec, f1

def analyze_thresholds(y_true, y_prob):
    # Only analyze if there are actual seizures in the fold to avoid undefined PR curves
    if np.sum(y_true) == 0:
        results = {t: (0.0, 0.0, 0.0) for t in [0.5, 0.3, 0.2, 0.1]}
        results['best'] = (0.5, 0.0, 0.0, 0.0)
        return results, [], [], []
        
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)
    
    # Calculate best F1
    with np.errstate(divide='ignore', invalid='ignore'):
        f1_scores = np.where((precisions + recalls) == 0, 0, 2 * (precisions * recalls) / (precisions + recalls))
    best_idx = np.argmax(f1_scores)
    best_thresh = thresholds[best_idx] if best_idx < len(thresholds) else 1.0
    best_f1 = f1_scores[best_idx]
    
    results = {}
    for t in [0.5, 0.3, 0.2, 0.1]:
        y_pred_t = (y_prob >= t).astype(int)
        p = precision_score(y_true, y_pred_t, zero_division=0)
        r = recall_score(y_true, y_pred_t, zero_division=0)
        f = f1_score(y_true, y_pred_t, zero_division=0)
        results[t] = (r, p, f)
        
    y_pred_best = (y_prob >= best_thresh).astype(int)
    bp = precision_score(y_true, y_pred_best, zero_division=0)
    br = recall_score(y_true, y_pred_best, zero_division=0)
    bf = f1_score(y_true, y_pred_best, zero_division=0)
    results['best'] = (best_thresh, br, bp, bf)
    
    return results, precisions, recalls, thresholds

def main():
    # Setup paths
    Path("scratch").mkdir(exist_ok=True)
    df = pd.read_parquet('chbmit_subset.parquet')
    drop_cols = ['target', 'patient_id', 'record', 'window_idx']
    X = df.drop(columns=[c for c in drop_cols if c in df.columns])
    y = df['target'].values
    groups = df['patient_id'].values
    
    logo = LeaveOneGroupOut()
    
    fold_metrics = []
    
    print("=== Leave-One-Patient-Out Cross-Validation ===")
    fold_idx = 1
    
    fold_probs = {}
    fold_y_true = {}
    
    for train_idx, test_idx in logo.split(X, y, groups):
        test_patient = groups[test_idx][0]
        
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        model = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler()),
            ('rf', RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42, n_jobs=-1))
        ])
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        
        acc, auc, prec, rec, f1 = get_metrics(y_test, y_pred, y_prob)
        
        n_sz_test = np.sum(y_test)
        print(f"Fold {fold_idx}: Held-out: {test_patient}")
        print(f"  Train: {len(y_train)}, Test: {len(y_test)} (Seizures in test: {n_sz_test})")
        print(f"  Acc: {acc:.4f}, AUC: {auc:.4f}, Prec: {prec:.4f}, Rec: {rec:.4f}, F1: {f1:.4f}\n")
        
        fold_metrics.append({
            'fold': fold_idx,
            'patient': test_patient,
            'acc': acc, 'auc': auc, 'prec': prec, 'rec': rec, 'f1': f1,
            'test_seizures': n_sz_test
        })
        
        fold_probs[test_patient] = y_prob
        fold_y_true[test_patient] = y_test
        
        fold_idx += 1
        
    metrics_df = pd.DataFrame(fold_metrics)
    print("=== Average Metrics Across Folds ===")
    print(f"AUC:       {metrics_df['auc'].mean():.4f} +/- {metrics_df['auc'].std():.4f}")
    print(f"Precision: {metrics_df['prec'].mean():.4f} +/- {metrics_df['prec'].std():.4f}")
    print(f"Recall:    {metrics_df['rec'].mean():.4f} +/- {metrics_df['rec'].std():.4f}")
    print(f"F1:        {metrics_df['f1'].mean():.4f} +/- {metrics_df['f1'].std():.4f}\n")
    
    # Worst vs Best Recall Analysis (exclude folds with 0 seizures in test)
    valid_folds = metrics_df[metrics_df['test_seizures'] > 0]
    if len(valid_folds) > 0:
        best_patient = valid_folds.loc[valid_folds['rec'].idxmax()]['patient']
        worst_patient = valid_folds.loc[valid_folds['rec'].idxmin()]['patient']
        
        print(f"=== Threshold Tuning ===")
        # Handle case where worst == best (e.g. if recall is the same)
        patients_to_analyze = [("Worst Recall Fold", worst_patient)]
        if best_patient != worst_patient:
            patients_to_analyze.append(("Best Recall Fold", best_patient))
            
        for name, pt in patients_to_analyze:
            print(f"--- {name} ({pt}) ---")
            res, p, r, t = analyze_thresholds(fold_y_true[pt], fold_probs[pt])
            for thresh in [0.5, 0.3, 0.2, 0.1]:
                print(f"Threshold {thresh}: Rec={res[thresh][0]:.4f}, Prec={res[thresh][1]:.4f}, F1={res[thresh][2]:.4f}")
            b_t, b_r, b_p, b_f = res['best']
            print(f"Best F1 Thr ({b_t:.4f}): Rec={b_r:.4f}, Prec={b_p:.4f}, F1={b_f:.4f}\n")
            
            # Save PR curve
            if len(p) > 0:
                plt.figure()
                plt.plot(r, p, marker='.')
                plt.xlabel('Recall')
                plt.ylabel('Precision')
                plt.title(f'PR Curve for {pt}')
                plt.grid(True)
                plt.savefig(f'scratch/pr_curve_{pt}.png')
                plt.close()

if __name__ == "__main__":
    main()
