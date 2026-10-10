"""Command line (works from any directory):  python -m dtr.main  synth | run     (from source/)

`run` defaults to leave-one-case-out over ALL cases found in the dataset folder: each case is held out
once, the models are trained on the other cases, and results are reported per case and macro-averaged.
"""
from __future__ import annotations
import argparse, json, os
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from .ingest import load_case, family_offset_report
from .groups import build_groups, make_dataset, FEATURE_NAMES
from .models import LogisticRegressionScratch
from .evaluate import (classification_report, effort_reduction_report, recall_by_offset,
                       temporal_report, tune_threshold)
from .reconstruct import reconcile, timeline, outlier_family

REPO = Path(__file__).resolve().parents[2]
_EZ = Path("CTF datasets") / "Modified EZ tools dataset"
# accept either folder spelling ("ADFIR forensic dataset" or "ADFIR_forensic_dataset")
DEFAULT_DATA = next((REPO / "data" / n / _EZ for n in ("ADFIR_forensic_dataset", "ADFIR forensic dataset")
                     if (REPO / "data" / n / _EZ).is_dir()),
                    REPO / "data" / "ADFIR_forensic_dataset" / _EZ)
DEFAULT_OUT = REPO / "data" / "outputs"
MIN_GROUPS = 20
SCORES = ["precision", "recall", "f1", "roc_auc", "pr_auc"]
MODELS = ["logistic_regression_scratch", "random_forest"]


def fit_models(X, y, seed):
    lr = LogisticRegressionScratch(seed=seed).fit(X, y)
    rf = RandomForestClassifier(n_estimators=300, min_samples_leaf=2, random_state=seed, n_jobs=-1).fit(X, y)
    return lr, rf


def validation_thresholds(groups, train_cases, seed):
    """Pick one threshold per model on a validation CASE taken from the training cases (largest one),
    fitting on the remaining training cases. The test case is never touched."""
    val = max(train_cases, key=lambda c: len(groups[c]))
    fit_g = [g for c in train_cases if c != val for g in groups[c]]
    if len(fit_g) < MIN_GROUPS or len(groups[val]) < MIN_GROUPS:
        return {m: 0.5 for m in MODELS}, None
    Xf, yf, _ = make_dataset(fit_g, seed=seed)
    Xv, yv, _ = make_dataset(groups[val], seed=seed + 2)
    lr, rf = fit_models(Xf, yf, seed)
    return {"logistic_regression_scratch": tune_threshold(yv, lr.predict_proba(Xv)),
            "random_forest": tune_threshold(yv, rf.predict_proba(Xv)[:, 1])}, val


def evaluate_fold(groups, holdout, raw_events, seed, threshold, tune):
    train_cases = [c for c in groups if c != holdout]
    train_g = [g for c in train_cases for g in groups[c]]
    thr, val_case = ({m: threshold for m in MODELS}, None)
    if tune:
        thr, val_case = validation_thresholds(groups, train_cases, seed)

    Xtr, ytr, _ = make_dataset(train_g, seed=seed)
    Xte, yte, gte = make_dataset(groups[holdout], seed=seed + 1)
    lr, rf = fit_models(Xtr, ytr, seed)
    p_lr, p_rf = lr.predict_proba(Xte), rf.predict_proba(Xte)[:, 1]
    flag_lr, flag_rf = (p_lr >= thr[MODELS[0]]).astype(int), (p_rf >= thr[MODELS[1]]).astype(int)

    gap_max = Xte[:, FEATURE_NAMES.index("gap_max")]
    gap_mean = Xte[:, FEATURE_NAMES.index("gap_mean")]
    m = {"n_train_groups": int(len(ytr)), "n_test_groups": int(len(yte)),
         "threshold": thr, "threshold_validation_case": val_case,
         "effort_reduction": effort_reduction_report(raw_events, len(gte)),
         MODELS[0]: classification_report(yte, flag_lr, p_lr),
         MODELS[1]: classification_report(yte, flag_rf, p_rf),
         # context for reading the numbers: 0.5 = chance, and no-ML "largest gap" scoring
         "baseline_gap_max_roc_auc": classification_report(yte, flag_rf, gap_max).get("roc_auc"),
         "untampered_median_gap_mean_log1p_s": float(np.median(gap_mean[yte == 0])) if (yte == 0).any() else None,
         "tampered_median_gap_mean_log1p_s": float(np.median(gap_mean[yte == 1])) if (yte == 1).any() else None}

    tam = [(g, p, f) for g, y, p, f in zip(gte, yte, p_rf, flag_rf) if y]
    m["rf_recall_by_clock_shift"] = recall_by_offset([g["tamper_offset"] for g, _, _ in tam], [f for _, _, f in tam])
    naive, aware, hit, n3 = [], [], 0, 0
    for g, p, _ in tam:
        r = reconcile(g, p, thr[MODELS[1]])
        naive.append(r["naive_when"] - g["true_time"])
        aware.append(r["when"] - g["true_time"])
        if len(g["ts"]) >= 3:
            n3 += 1
            hit += outlier_family(g["ts"]) == g["tampered_family"]
    m["temporal_naive"], m["temporal_reconciled"] = temporal_report(naive), temporal_report(aware)
    m["tampered_source_identified_3plus"] = {"n": n3, "accuracy": hit / n3 if n3 else None}
    m["rf_top_features"] = [(n, round(float(v), 4)) for n, v in
                            sorted(zip(FEATURE_NAMES, rf.feature_importances_), key=lambda x: -x[1])[:8]]
    preds = pd.DataFrame({"case": holdout, "key": [g["key"] for g in gte], "truth": yte, "p_lr": p_lr, "p_rf": p_rf})
    return m, preds, timeline(gte, p_rf, threshold=thr[MODELS[1]]), lr.loss_history


def run(data_dir, out_dir, holdout="all", threshold=0.5, tune=False, seed=0, min_families=2) -> dict:
    os.makedirs(out_dir, exist_ok=True)
    cases = sorted(d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d)))
    if not cases:
        raise SystemExit(f"no case folders found in {data_dir}")
    groups, raw_events, offsets = {}, {}, {}
    for c in cases:
        ev = load_case(os.path.join(data_dir, c), c)
        groups[c], raw_events[c] = build_groups(ev, min_families), len(ev)
        offsets[c] = family_offset_report(ev)
        print(f"[{c}] events={len(ev):,} groups(>= {min_families} families)={len(groups[c]):,} "
              f"families={ev['family'].value_counts().to_dict()}")
    usable = [c for c in cases if len(groups[c]) >= MIN_GROUPS]
    folds = usable if holdout == "all" else [holdout]
    if holdout != "all" and holdout not in cases:
        raise SystemExit(f"holdout {holdout!r} not in {cases}")
    for c in set(cases) - set(usable):
        print(f"[warn] {c}: only {len(groups[c])} groups - used for training only, not as a test case")

    per_case, preds_all, tl_all, losses = {}, [], [], {}
    for h in folds:
        if sum(len(groups[c]) for c in cases if c != h) < MIN_GROUPS:
            raise SystemExit("not enough training groups")
        m, preds, tl, loss = evaluate_fold(groups, h, raw_events[h], seed, threshold, tune)
        per_case[h] = m
        preds_all.append(preds); tl_all.append(tl.assign(case=h)); losses[h] = loss
        print(f"[fold {h}] RF f1={m[MODELS[1]]['f1']:.3f} auc={m[MODELS[1]].get('roc_auc', float('nan')):.3f} | "
              f"LR f1={m[MODELS[0]]['f1']:.3f} auc={m[MODELS[0]].get('roc_auc', float('nan')):.3f} | "
              f"gap_max-only auc={m['baseline_gap_max_roc_auc']:.3f}")

    macro = {k: {s: float(np.mean([per_case[c][k][s] for c in folds if s in per_case[c][k]])) for s in SCORES}
             for k in MODELS}
    metrics = {"protocol": "leave-one-case-out" if holdout == "all" else f"holdout={holdout}",
               "threshold_mode": "tuned on a validation case" if tune else f"fixed {threshold}",
               "cases": folds, "macro_average": macro, "per_case": per_case, "family_offsets": offsets}
    json.dump(metrics, open(os.path.join(out_dir, "metrics.json"), "w"), indent=2)
    pd.concat(preds_all).to_csv(os.path.join(out_dir, "predictions.csv"), index=False)
    pd.concat(tl_all).to_csv(os.path.join(out_dir, "reconstructed_timeline.csv"), index=False)
    pd.DataFrame(dict(((c, pd.Series(v)) for c, v in losses.items()))).rename_axis("iteration").reset_index() \
        .to_csv(os.path.join(out_dir, "loss_history.csv"), index=False)
    return metrics


def main():
    ap = argparse.ArgumentParser(prog="dtr")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("synth", help="generate SYNTHETIC cases in the ADFIR schema")
    s.add_argument("--out", default=str(REPO / "data" / "synthetic"))
    s.add_argument("--programs", type=int, default=300)
    r = sub.add_parser("run", help="train, evaluate, write outputs")
    r.add_argument("--data", default=str(DEFAULT_DATA))
    r.add_argument("--holdout", default="all", help="'all' = leave-one-case-out (default) or one case name")
    r.add_argument("--out", default=str(DEFAULT_OUT))
    r.add_argument("--threshold", type=float, default=0.5, help="fixed decision threshold")
    r.add_argument("--tune-threshold", action="store_true", help="tune it on a validation case instead")
    r.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    if a.cmd == "synth":
        from .synth import make_dataset_dir
        make_dataset_dir(a.out, a.programs)
        print("wrote", a.out)
    else:
        res = run(a.data, a.out, a.holdout, a.threshold, a.tune_threshold, a.seed)
        print(json.dumps({k: res[k] for k in ("protocol", "threshold_mode", "macro_average")}, indent=2))


if __name__ == "__main__":
    main()