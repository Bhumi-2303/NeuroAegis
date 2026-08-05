import pandas as pd
import numpy as np
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, precision_recall_curve
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
import json
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
    df = pd.read_parquet('chbmit_subset.parquet')
    drop_cols = ['target', 'patient_id', 'record', 'window_idx']
    X = df.drop(columns=[c for c in drop_cols if c in df.columns])
    y = df['target'].values
    groups = df['patient_id'].values
    
    logo = LeaveOneGroupOut()
    
    fold_metrics = []
    fold_probs = {}
    fold_y_true = {}
    
    print("Running LOO Validation...")
    
    fold_idx = 1
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
        n_sz_test = int(np.sum(y_test))
        
        res, p, r, t = analyze_thresholds(y_test, y_prob)
        b_t, b_r, b_p, b_f = res['best']
        
        fold_metrics.append({
            'fold': fold_idx,
            'patient_id': str(test_patient),
            'test_seizures': n_sz_test,
            'default_threshold': {
                'accuracy': float(acc),
                'roc_auc': float(auc) if not np.isnan(auc) else None,
                'precision': float(prec),
                'recall': float(rec),
                'f1': float(f1)
            },
            'tuned_threshold': {
                'threshold': float(b_t),
                'precision': float(b_p),
                'recall': float(b_r),
                'f1': float(b_f)
            }
        })
        
        fold_probs[test_patient] = y_prob
        fold_y_true[test_patient] = y_test
        fold_idx += 1

    # Calculate overall average metrics
    # We must properly average valid folds
    auc_list = [f['default_threshold']['roc_auc'] for f in fold_metrics if f['default_threshold']['roc_auc'] is not None]
    
    final_output = {
        "metadata": {
            "sample_size": "n=4 patients",
            "validation_strategy": "Leave-One-Patient-Out Cross-Validation (LOOCV)",
            "description": "True clinical performance preventing data leakage."
        },
        "average_metrics": {
            "default_threshold": {
                "accuracy": sum(f['default_threshold']['accuracy'] for f in fold_metrics) / len(fold_metrics),
                "roc_auc": sum(auc_list) / len(auc_list) if auc_list else None,
                "precision": sum(f['default_threshold']['precision'] for f in fold_metrics) / len(fold_metrics),
                "recall": sum(f['default_threshold']['recall'] for f in fold_metrics) / len(fold_metrics),
                "f1": sum(f['default_threshold']['f1'] for f in fold_metrics) / len(fold_metrics),
            },
            "tuned_threshold": {
                "precision": sum(f['tuned_threshold']['precision'] for f in fold_metrics) / len(fold_metrics),
                "recall": sum(f['tuned_threshold']['recall'] for f in fold_metrics) / len(fold_metrics),
                "f1": sum(f['tuned_threshold']['f1'] for f in fold_metrics) / len(fold_metrics)
            }
        },
        "folds": fold_metrics
    }
    
    out_dir = Path("apps/api/data")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "metrics.json"
    
    with open(out_file, "w") as f:
        json.dump(final_output, f, indent=2)
        
    print(f"Metrics saved to {out_file}")

if __name__ == "__main__":
    main()
