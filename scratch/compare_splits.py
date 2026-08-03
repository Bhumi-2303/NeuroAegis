import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')

def get_metrics(y_true, y_pred, y_prob):
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision (Seizure)": precision_score(y_true, y_pred, pos_label=1, zero_division=0),
        "Recall (Seizure)": recall_score(y_true, y_pred, pos_label=1, zero_division=0),
        "F1 (Seizure)": f1_score(y_true, y_pred, pos_label=1, zero_division=0),
        "AUC": roc_auc_score(y_true, y_prob)
    }

def print_results(name, y_train, y_test, metrics):
    print(f"--- {name} ---")
    print(f"Train size: {len(y_train)}, Test size: {len(y_test)}")
    print(f"Train Seizures: {sum(y_train)}, Test Seizures: {sum(y_test)}")
    print(f"Accuracy:            {metrics['Accuracy']:.4f}")
    print(f"Precision (Seizure): {metrics['Precision (Seizure)']:.4f}")
    print(f"Recall (Seizure):    {metrics['Recall (Seizure)']:.4f}")
    print(f"F1 (Seizure):        {metrics['F1 (Seizure)']:.4f}")
    print(f"AUC:                 {metrics['AUC']:.4f}\n")


def main():
    df = pd.read_parquet('chbmit_subset.parquet')
    
    # Separate features, target, groups
    # The feature columns are all columns except 'target', 'patient_id', 'record', 'window_idx'
    drop_cols = ['target', 'patient_id', 'record', 'window_idx']
    
    X = df.drop(columns=[c for c in drop_cols if c in df.columns])
    y = df['target'].values
    groups = df['patient_id'].values
    
    model = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('rf', RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42, n_jobs=-1))
    ])

    # 1. Random Split
    X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    model.fit(X_train_r, y_train_r)
    y_pred_r = model.predict(X_test_r)
    y_prob_r = model.predict_proba(X_test_r)[:, 1]
    metrics_r = get_metrics(y_test_r, y_pred_r, y_prob_r)
    
    # 2. Patient-wise Split
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(gss.split(X, y, groups))
    
    X_train_g, X_test_g = X.iloc[train_idx], X.iloc[test_idx]
    y_train_g, y_test_g = y[train_idx], y[test_idx]
    
    # Confirm which patient went to test
    train_patients = df.iloc[train_idx]['patient_id'].unique()
    test_patients = df.iloc[test_idx]['patient_id'].unique()
    print(f"GroupShuffleSplit details:")
    print(f"Train Patients: {train_patients}")
    print(f"Test Patients:  {test_patients}\n")
    
    model.fit(X_train_g, y_train_g)
    y_pred_g = model.predict(X_test_g)
    try:
        y_prob_g = model.predict_proba(X_test_g)[:, 1]
        metrics_g = get_metrics(y_test_g, y_pred_g, y_prob_g)
    except Exception as e:
        # Might fail if AUC is undefined (e.g. 0 seizures in test set)
        print(f"Error computing AUC for Group Split (no seizures in test?): {e}")
        metrics_g = get_metrics(y_test_g, y_pred_g, y_pred_g)
        metrics_g["AUC"] = np.nan

    print_results("Random train_test_split", y_train_r, y_test_r, metrics_r)
    print_results("Patient-wise GroupShuffleSplit", y_train_g, y_test_g, metrics_g)


if __name__ == "__main__":
    main()
