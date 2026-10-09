"""Metrics: classification (from scratch) + temporal reconstruction error."""
from __future__ import annotations
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score


def confusion(y, p):
    y, p = np.asarray(y), np.asarray(p)
    return int(((p == 1) & (y == 1)).sum()), int(((p == 1) & (y == 0)).sum()), \
           int(((p == 0) & (y == 0)).sum()), int(((p == 0) & (y == 1)).sum())


def classification_report(y, p, proba=None) -> dict:
    tp, fp, tn, fn = confusion(y, p)
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    out = {"accuracy": (tp + tn) / max(len(y), 1), "precision": prec, "recall": rec,
           "f1": 2 * prec * rec / (prec + rec) if prec + rec else 0.0,
           "tp": tp, "fp": fp, "tn": tn, "fn": fn}
    if proba is not None and len(set(y)) == 2:
        out["roc_auc"] = float(roc_auc_score(y, proba))
        out["pr_auc"] = float(average_precision_score(y, proba))
    return out


def temporal_report(err_seconds) -> dict:
    e = np.abs(np.asarray(err_seconds, float))
    if not len(e):
        return {}
    return {"n": int(len(e)), "mae_s": float(e.mean()), "median_s": float(np.median(e)),
            "within_1min": float((e <= 60).mean()), "within_5min": float((e <= 300).mean()),
            "within_1h": float((e <= 3600).mean())}
