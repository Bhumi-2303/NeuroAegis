import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold, GroupKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

def get_metrics(y_true, y_pred, y_prob):
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "AUC": roc_auc_score(y_true, y_prob)
    }

def print_metrics(title, metrics_list):
    print(f"\n{'='*50}\n{title}\n{'='*50}")
    if isinstance(metrics_list, dict):
        for k, v in metrics_list.items():
            print(f"{k}: {v:.4f}")
    else:
        avg = {k: np.mean([m[k] for m in metrics_list]) for k in metrics_list[0].keys()}
        std = {k: np.std([m[k] for m in metrics_list]) for k in metrics_list[0].keys()}
        for k in avg.keys():
            print(f"{k}: {avg[k]:.4f} \u00B1 {std[k]:.4f}")

def main():
    print("Loading datasets...")
    # Load Bonn
    df_bonn = pd.read_csv("NeuroAegis_bonn_dataset_model/features/features.csv")
    df_bonn['target_binary'] = (df_bonn['label'] == 4).astype(int)
    bonn_cols = set(df_bonn.columns)

    # Load CHB-MIT
    df_chb = pd.read_parquet("chbmit_subset.parquet")
    df_chb['target_binary'] = df_chb['target']
    
    # Rename CHB-MIT columns
    chb_renames = {c: c.replace('Ch0_', '') for c in df_chb.columns if c.startswith('Ch0_')}
    df_chb = df_chb.rename(columns=chb_renames)
    chb_cols = set(df_chb.columns)

    # Common Features
    metadata_cols = {'patient_id', 'epoch_id', 'label', 'target', 'y', 'target_binary', 'record', 'window_idx'}
    common_features = sorted(list(bonn_cols.intersection(chb_cols) - metadata_cols))
    print(f"Found {len(common_features)} common features.")

    X_bonn = df_bonn[common_features].values
    y_bonn = df_bonn['target_binary'].values

    X_chb = df_chb[common_features].values
    y_chb = df_chb['target_binary'].values
    groups_chb = df_chb['patient_id'].values

    lgb_params = {
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt',
        'random_state': 42,
        'verbose': -1,
        'n_estimators': 100,
        'max_depth': 6
    }

    # 1. In-Domain Bonn (5-Fold Stratified)
    bonn_metrics = []
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for train_idx, test_idx in skf.split(X_bonn, y_bonn):
        X_tr, X_te = X_bonn[train_idx], X_bonn[test_idx]
        y_tr, y_te = y_bonn[train_idx], y_bonn[test_idx]
        
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_tr)
        X_te = scaler.transform(X_te)
        
        clf = lgb.LGBMClassifier(**lgb_params)
        clf.fit(X_tr, y_tr)
        y_pred = clf.predict(X_te)
        y_prob = clf.predict_proba(X_te)[:, 1]
        bonn_metrics.append(get_metrics(y_te, y_pred, y_prob))
    
    print_metrics("1. IN-DOMAIN BONN (5-Fold CV)", bonn_metrics)

    # 2. In-Domain CHB-MIT (LOPO-CV)
    chb_metrics = []
    gkf = GroupKFold(n_splits=len(np.unique(groups_chb)))
    for train_idx, test_idx in gkf.split(X_chb, y_chb, groups=groups_chb):
        X_tr, X_te = X_chb[train_idx], X_chb[test_idx]
        y_tr, y_te = y_chb[train_idx], y_chb[test_idx]
        
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_tr)
        X_te = scaler.transform(X_te)
        
        clf = lgb.LGBMClassifier(**lgb_params)
        clf.fit(X_tr, y_tr)
        y_pred = clf.predict(X_te)
        y_prob = clf.predict_proba(X_te)[:, 1]
        chb_metrics.append(get_metrics(y_te, y_pred, y_prob))

    print_metrics("2. IN-DOMAIN CHB-MIT (LOPO-CV)", chb_metrics)

    # 3. Cross-Domain (Train Bonn -> Test CHB-MIT)
    # INDEPENDENT SCALING: Fit a separate scaler for each dataset to eliminate the V vs uV unit mismatch
    scaler_bonn = StandardScaler()
    X_bonn_scaled = scaler_bonn.fit_transform(X_bonn)
    
    scaler_chb = StandardScaler()
    X_chb_scaled = scaler_chb.fit_transform(X_chb)
    
    clf_bonn = lgb.LGBMClassifier(**lgb_params)
    clf_bonn.fit(X_bonn_scaled, y_bonn)
    
    y_pred_cross = clf_bonn.predict(X_chb_scaled)
    y_prob_cross = clf_bonn.predict_proba(X_chb_scaled)[:, 1]
    
    print_metrics("3. CROSS-DOMAIN (Train Bonn -> Test CHB-MIT)", get_metrics(y_chb, y_pred_cross, y_prob_cross))

    # 4. Cross-Domain (Train CHB-MIT -> Test Bonn)
    clf_chb = lgb.LGBMClassifier(**lgb_params)
    clf_chb.fit(X_chb_scaled, y_chb)
    
    y_pred_cross_2 = clf_chb.predict(X_bonn_scaled)
    y_prob_cross_2 = clf_chb.predict_proba(X_bonn_scaled)[:, 1]
    
    print_metrics("4. CROSS-DOMAIN (Train CHB-MIT -> Test Bonn)", get_metrics(y_bonn, y_pred_cross_2, y_prob_cross_2))

if __name__ == "__main__":
    main()
