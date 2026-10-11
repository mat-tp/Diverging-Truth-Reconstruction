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


def tune_threshold(y, proba, grid=None) -> float:
    """Threshold maximising F1 on (y, proba). Call it on VALIDATION data only - tuning on the test
    case leaks the test labels into the reported score."""
    grid = np.linspace(0.05, 0.95, 91) if grid is None else grid
    best, best_f1 = 0.5, -1.0
    for t in grid:
        f1 = classification_report(y, (np.asarray(proba) >= t).astype(int))["f1"]
        if f1 > best_f1:
            best, best_f1 = float(t), f1
    return best


def recall_by_offset(offsets, flagged) -> dict:
    """Recall of tampered groups, bucketed by how far the clock was shifted (seconds)."""
    off, fl = np.abs(np.asarray(offsets, float)), np.asarray(flagged, int)
    edges = [(0, 600, "<10min"), (600, 3600, "10min-1h"), (3600, 86400, "1h-1d"), (86400, np.inf, ">1d")]
    out = {}
    for lo, hi, name in edges:
        m = (off >= lo) & (off < hi)
        out[name] = {"n": int(m.sum()), "recall": float(fl[m].mean()) if m.any() else None}
    return out


def effort_reduction_report(raw_count: int, grouped_count: int) -> dict:
    """Raw executable-related events vs evidence groups. This is a COMPRESSION ratio, not a measured
    saving in investigator effort (single-source programs are not in any group)."""
    if raw_count == 0:
        return {"raw_events": 0, "evidence_groups": 0, "compression_percentage": 0.0}
    return {"raw_events": int(raw_count), "evidence_groups": int(grouped_count),
            "compression_percentage": round((raw_count - grouped_count) / raw_count * 100, 2)}



def review_effort(y, scores, target: float = 0.9) -> dict:
    """How many evidence groups must be reviewed to find `target` of the tampered ones?

    Ranked: groups are read in descending conflict probability. Raw: groups are read in an uninformed order
    (expected value target * N). This counts groups, not minutes: no analyst time is measured here.
    """
    y, s = np.asarray(y), np.asarray(scores)
    n, pos = len(y), int(y.sum())
    if pos == 0:
        return {"n_groups": n, "n_tampered": 0, "ranked_reviews": None, "raw_expected_reviews": None, "saving": None}
    order = np.argsort(-s, kind="stable")
    found = np.cumsum(y[order])
    need = int(np.searchsorted(found, int(np.ceil(target * pos))) + 1)
    raw = target * n
    return {"n_groups": n, "n_tampered": pos, "target_recall": target, "ranked_reviews": need,
            "raw_expected_reviews": float(raw), "saving": float(1 - need / raw)}
