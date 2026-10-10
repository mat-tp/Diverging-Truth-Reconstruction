from __future__ import annotations
import os
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from .groups import make_dataset
from .ingest import load_case
from .models import LogisticRegressionScratch
from sklearn.ensemble import RandomForestClassifier

def optimize_threshold(data_dir: str, holdout: str = "Magnet_CTF_2022"):
    cases = sorted(d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d)))
    groups = {c: make_dataset([g for g in load_case(os.path.join(data_dir, c), c)][:100], seed=0) for c in cases if c != holdout}
    
    # Run evaluation sweep
    predictions_path = "../data/outputs/predictions.csv"
    if not os.path.exists(predictions_path):
        print("Run pipeline first to generate predictions.csv")
        return

    df = pd.read_csv(predictions_path)
    y_true, y_proba = df["truth"], df["p_rf"]

    thresholds = np.linspace(0.1, 0.9, 81)
    best_t, best_f1 = 0.5, 0.0

    for t in thresholds:
        preds = (y_proba >= t).astype(int)
        score = f1_score(y_true, preds, zero_division=0)
        if score > best_f1:
            best_f1 = score
            best_t = t

    print(f"Optimal Threshold Found: {best_t:.2f} with F1-Score: {best_f1:.4f}")

if __name__ == "__main__":
    optimize_threshold("../data/ADFIR forensic dataset/CTF datasets/Modified EZ tools dataset")
    