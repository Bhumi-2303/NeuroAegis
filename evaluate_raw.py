import pandas as pd
df = pd.read_csv("research/results/phase_4b/final_test_predictions.csv")
print(f"Total rows: {len(df)}")
print(f"Positive labels: {df['label_50pct_overlap'].sum()}")
print(f"Positive predictions: {(df['predicted_probability'] >= 0.5).sum()}")
