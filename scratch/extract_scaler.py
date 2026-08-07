import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import joblib

df = pd.read_parquet('chbmit_subset.parquet')
DROP_COLS = ["target", "patient_id", "record", "window_idx"]
feature_cols = [c for c in df.columns if c not in DROP_COLS]
X = df[feature_cols].values

imputer = SimpleImputer(strategy="median")
scaler = StandardScaler()

print("Fitting imputer and scaler...")
X_imputed = imputer.fit_transform(X)
scaler.fit(X_imputed)

joblib.dump(imputer, 'apps/api/models/chbmit/imputer.pkl')
joblib.dump(scaler, 'apps/api/models/chbmit/scaler.pkl')

print("Saved imputer.pkl and scaler.pkl to apps/api/models/chbmit/")
