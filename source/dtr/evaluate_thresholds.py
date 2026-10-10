import json
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, f1_score

# Load metrics and predictions
outputs_dir = "../data/outputs"
predictions = pd.read_csv(f"{outputs_dir}/predictions.csv")

print(f"Loaded {len(predictions)} prediction rows for analysis.")

# Threshold tuning loop
y_true = predictions['actual_conflict'] if 'actual_conflict' in predictions.columns else None
y_pred_proba = predictions['conflict_prob'] if 'conflict_prob' in predictions.columns else None

if y_true is not None and y_pred_proba is not None:
    thresholds = np.linspace(0.1, 0.9, 9)
    best_thresh = 0.5
    best_f1 = 0.0

    for t in thresholds:
        preds = (y_pred_proba >= t).astype(int)
        score = f1_score(y_true, preds, zero_division=0)
        if score > best_f1:
            best_f1 = score
            best_thresh = t

    print(f"Optimal threshold: {best_thresh:.2f} with F1-score: {best_f1:.4f}")
else:
    print("Required columns for threshold optimization not found in predictions.csv.")