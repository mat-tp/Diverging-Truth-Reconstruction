"""Command line: `python -m dtr.cli synth|run`  (run from the `source/` directory)."""
from __future__ import annotations
import argparse, json, os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from .ingest import load_case
from .groups import build_groups, make_dataset, FEATURE_NAMES
from .models import LogisticRegressionScratch
from .evaluate import classification_report, temporal_report
from .reconstruct import reconcile, timeline, outlier_family

def run(data_dir: str, holdout: str, out_dir: str, seed: int = 0, min_families: int = 2) -> dict:
    os.makedirs(out_dir, exist_ok=True)
    cases = sorted(d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d)))
    if holdout not in cases:
        raise SystemExit(f"holdout case {holdout!r} not in {cases}")
    groups = {}
    for c in cases:
        ev = load_case(os.path.join(data_dir, c), c)
        groups[c] = build_groups(ev, min_families)
        print(f"[{c}] events={len(ev):,} groups(>= {min_families} sources)={len(groups[c])}")
    train_g = [g for c in cases if c != holdout for g in groups[c]]
    Xtr, ytr, _ = make_dataset(train_g, seed=seed)
    Xte, yte, gte = make_dataset(groups[holdout], seed=seed + 1)
    if len(set(ytr)) < 2 or len(Xte) == 0:
        raise SystemExit("not enough groups to train/evaluate")

    lr = LogisticRegressionScratch(seed=seed).fit(Xtr, ytr)
    rf = RandomForestClassifier(n_estimators=300, min_samples_leaf=2, random_state=seed, n_jobs=-1).fit(Xtr, ytr)
    p_lr, p_rf = lr.predict_proba(Xte), rf.predict_proba(Xte)[:, 1]
    metrics = {"holdout": holdout, "n_train": int(len(ytr)), "n_test": int(len(yte)),
               "logistic_regression_scratch": classification_report(yte, (p_lr >= .5).astype(int), p_lr),
               "random_forest": classification_report(yte, (p_rf >= .5).astype(int), p_rf)}

    naive, aware, hit, n3 = [], [], 0, 0
    for g, y, p in zip(gte, yte, p_rf):
        if not y:
            continue
        r = reconcile(g, p)
        naive.append(r["naive_when"] - g["true_time"])
        aware.append(r["when"] - g["true_time"])
        if len(g["ts"]) >= 3:
            n3 += 1
            hit += outlier_family(g["ts"]) == g["tampered_family"]
    metrics["temporal_naive"] = temporal_report(naive)
    metrics["temporal_reconciled"] = temporal_report(aware)
    metrics["tampered_source_identified_(>=3 sources)"] = {"n": n3, "accuracy": hit / n3 if n3 else None}

    imp = sorted(zip(FEATURE_NAMES, rf.feature_importances_), key=lambda x: -x[1])[:8]
    metrics["rf_top_features"] = [(n, round(float(v), 4)) for n, v in imp]
    json.dump(metrics, open(os.path.join(out_dir, "metrics.json"), "w"), indent=2)
    pd.DataFrame({"iteration": range(len(lr.loss_history)), "loss": lr.loss_history}).to_csv(
        os.path.join(out_dir, "loss_history.csv"), index=False)
    pd.DataFrame({"key": [g["key"] for g in gte], "truth": yte, "p_lr": p_lr, "p_rf": p_rf}).to_csv(
        os.path.join(out_dir, "predictions.csv"), index=False)
    timeline(gte, p_rf).to_csv(os.path.join(out_dir, "reconstructed_timeline.csv"), index=False)
    return metrics

def main():
    ap = argparse.ArgumentParser(prog="dtr")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("synth", help="generate SYNTHETIC cases in the ADFIR schema")
    s.add_argument("--out", default="../data/synthetic")
    s.add_argument("--programs", type=int, default=300)
    r = sub.add_parser("run", help="train, evaluate, write outputs")
    r.add_argument("--data", required=True)
    r.add_argument("--holdout", default="Magnet_CTF_2022")
    r.add_argument("--out", default="../data/outputs")
    r.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    if a.cmd == "synth":
        from .synth import make_dataset_dir
        make_dataset_dir(a.out, a.programs)
        print("wrote", a.out)
    else:
        print(json.dumps(run(a.data, a.holdout, a.out, a.seed), indent=2))

if __name__ == "__main__":
    main()