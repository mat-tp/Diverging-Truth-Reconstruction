"""Candidate matching + tamper injection + feature extraction (one vector per evidence group)."""
from __future__ import annotations
import itertools
import numpy as np
import pandas as pd
from .ingest import FAMILIES

PAIRS = list(itertools.combinations(FAMILIES, 2))
FEATURE_NAMES = (["n_families", "log_n_events", "log_span", "gap_min", "gap_mean", "gap_max",
                  "user_agreement", "n_users"] + [f"gap_{a}__{b}" for a, b in PAIRS])

def build_groups(events: pd.DataFrame, min_families: int = 2) -> list[dict]:
    """Group events by (case, exe name); keep groups corroborated by >= min_families families."""
    groups = []
    for (case, key), g in events.groupby(["case", "key"], sort=False):
        if g["family"].nunique() >= min_families:
            groups.append({"case": case, "key": key,
                           "ts": {f: np.sort(x["ts"].to_numpy()) for f, x in g.groupby("family")},
                           "users": [u for u in g["user"].dropna().astype(str).str.lower()],
                           "attack": int(g["attack"].max())})
    return groups

def pair_gap(a: np.ndarray, b: np.ndarray) -> float:
    """Smallest |ta - tb| over all timestamp pairs (sources hold several timestamps each)."""
    idx = np.searchsorted(b, a)
    lo = np.abs(a - b[np.clip(idx - 1, 0, len(b) - 1)])
    hi = np.abs(a - b[np.clip(idx, 0, len(b) - 1)])
    return float(np.minimum(lo, hi).min())

def consensus_time(ts: dict[str, np.ndarray]) -> tuple[float, float]:
    """Time t* minimising the summed distance to the nearest timestamp of every family."""
    cands = np.unique(np.concatenate(list(ts.values())))
    cost = np.stack([np.abs(cands[:, None] - v[None, :]).min(axis=1) for v in ts.values()], axis=1)
    i = int(np.argmin(cost.sum(axis=1)))
    return float(cands[i]), float(cost[i].max())

def tamper(group: dict, rng: np.random.Generator) -> dict:
    """Shift one family's clock by +-(1 min .. 30 days), log-uniform. Records ground truth."""
    fam = rng.choice(sorted(group["ts"]))
    off = 10 ** rng.uniform(np.log10(60), np.log10(30 * 86400)) * rng.choice([-1, 1])
    g = dict(group)
    g["ts"] = {f: (v + off if f == fam else v) for f, v in group["ts"].items()}
    g["tampered_family"], g["tamper_offset"] = str(fam), float(off)
    g["true_time"] = consensus_time(group["ts"])[0]
    return g

def features(group: dict) -> np.ndarray:
    ts = group["ts"]
    fams = [f for f in FAMILIES if f in ts]
    gaps = {}
    for a, b in PAIRS:
        if a in ts and b in ts:
            gaps[(a, b)] = np.log1p(pair_gap(ts[a], ts[b]))
    vals = np.array(list(gaps.values()))
    allt = np.concatenate(list(ts.values()))
    users = group["users"]
    if users:
        _, counts = np.unique(users, return_counts=True)
        agree, n_users = counts.max() / counts.sum(), len(counts)
    else:
        agree, n_users = 1.0, 0
    fill = vals.mean() if len(vals) > 0 else 0.0
    row = [len(fams), np.log1p(len(allt)), np.log1p(allt.max() - allt.min()),
           vals.min() if len(gaps) else 0.0, 
           vals.mean() if len(gaps) else 0.0, 
           vals.max() if len(gaps) else 0.0, 
           agree, n_users]
    row += [gaps.get(p, fill) for p in PAIRS]
    return np.array(row, dtype=float)

def make_dataset(groups: list[dict], tamper_frac: float = 0.5, seed: int = 0):
    """Return X, y (1 = tampered / conflicting), and the (possibly tampered) groups."""
    rng = np.random.default_rng(seed)
    out, y = [], []
    for g in groups:
        if rng.random() < tamper_frac:
            out.append(tamper(g, rng))
            y.append(1)
        else:
            g2 = dict(g)
            g2["true_time"] = consensus_time(g["ts"])[0]
            out.append(g2)
            y.append(0)
    X = np.stack([features(g) for g in out]) if out else np.zeros((0, len(FEATURE_NAMES)))
    return X, np.array(y, dtype=int), out