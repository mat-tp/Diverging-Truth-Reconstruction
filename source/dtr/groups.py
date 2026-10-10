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

MIN_BASELINE_N = 10
MIN_SCALE = 0.5  # log-seconds; stops a very tight pair from exploding into huge z-scores


def _robust_stats(vals) -> tuple[float, float]:
    v = np.asarray(vals, float)
    if len(v) == 0:
        return 0.0, 1.0
    med = float(np.median(v))
    return med, max(1.4826 * float(np.median(np.abs(v - med))), MIN_SCALE)


def compute_baselines(groups: list[dict]) -> dict:
    """Natural (untampered) gap statistics of ONE case, per family pair and for the group span.

    Different cases have very different natural disagreement between artifacts (e.g. the median
    event-log -> filesystem offset is ~2 s in one case and ~2,300 s in another). A model that sees raw
    gaps mistakes a noisy case for a tampered one, so gap features can be expressed as robust z-scores
    against the case's own baseline. Uses no labels: call it on the groups BEFORE tampering is injected.
    Pairs seen in fewer than MIN_BASELINE_N groups fall back to the pooled statistics of the case.
    """
    per_pair: dict = {p: [] for p in PAIRS}
    spans, pooled = [], []
    for g in groups:
        ts = g["ts"]
        allt = np.concatenate(list(ts.values()))
        spans.append(np.log1p(allt.max() - allt.min()))
        for a, b in PAIRS:
            if a in ts and b in ts:
                v = np.log1p(pair_gap(ts[a], ts[b]))
                per_pair[(a, b)].append(v)
                pooled.append(v)
    fallback = _robust_stats(pooled)
    out = {p: (_robust_stats(v) if len(v) >= MIN_BASELINE_N else fallback) for p, v in per_pair.items()}
    out["span"] = _robust_stats(spans)
    return out


def features(group: dict, baseline: dict | None = None) -> np.ndarray:
    """Feature vector for one group. With `baseline` (from compute_baselines) the gap and span
    features are robust z-scores against the case's natural distribution instead of raw log-seconds."""
    ts = group["ts"]
    fams = [f for f in FAMILIES if f in ts]
    gaps = {}
    for a, b in PAIRS:
        if a in ts and b in ts:
            v = np.log1p(pair_gap(ts[a], ts[b]))
            if baseline is not None:
                med, scale = baseline[(a, b)]
                v = (v - med) / scale
            gaps[(a, b)] = v
    vals = np.array(list(gaps.values()))
    allt = np.concatenate(list(ts.values()))
    users = group["users"]
    if users:
        _, counts = np.unique(users, return_counts=True)
        agree, n_users = counts.max() / counts.sum(), len(counts)
    else:
        agree, n_users = 1.0, 0
    fill = vals.mean() if len(vals) > 0 else 0.0
    span = np.log1p(allt.max() - allt.min())
    if baseline is not None:
        span = (span - baseline["span"][0]) / baseline["span"][1]
    row = [len(fams), np.log1p(len(allt)), span,
           vals.min() if len(gaps) else 0.0, 
           vals.mean() if len(gaps) else 0.0, 
           vals.max() if len(gaps) else 0.0, 
           agree, n_users]
    row += [gaps.get(p, fill) for p in PAIRS]
    return np.array(row, dtype=float)

def make_dataset(groups: list[dict], tamper_frac: float = 0.5, seed: int = 0, normalise: bool = False):
    """Return X, y (1 = tampered / conflicting), and the (possibly tampered) groups.

    normalise=True expresses gap/span features relative to each case's own untampered baseline
    (computed per case, before any tampering is injected, so it uses no labels).
    """
    baselines = {}
    if normalise:
        by_case: dict = {}
        for g in groups:
            by_case.setdefault(g["case"], []).append(g)
        baselines = {c: compute_baselines(gs) for c, gs in by_case.items()}
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
    X = (np.stack([features(g, baselines.get(g["case"])) for g in out])
         if out else np.zeros((0, len(FEATURE_NAMES))))
    return X, np.array(y, dtype=int), out
