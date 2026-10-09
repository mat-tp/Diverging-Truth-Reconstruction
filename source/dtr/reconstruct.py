"""Who / what / when reconstruction with conflict-aware reconciliation."""
from __future__ import annotations
import numpy as np
import pandas as pd
from .groups import consensus_time, pair_gap

def outlier_family(ts: dict) -> str | None:
    """Family whose evidence sits furthest from all the others (needs >= 3 families)."""
    if len(ts) < 3:
        return None
    score = {f: np.mean([pair_gap(ts[f], ts[g]) for g in ts if g != f]) for f in ts}
    return max(score, key=score.get)

def reconcile(group: dict, conflict_prob: float, threshold: float = 0.5) -> dict:
    """If the model flags a conflict, set aside the most divergent family and re-estimate 'when'."""
    ts = group["ts"]
    t_naive, spread_naive = consensus_time(ts)
    dropped = None
    if conflict_prob >= threshold:
        dropped = outlier_family(ts)
    kept = {f: v for f, v in ts.items() if f != dropped}
    
    if not kept:
        kept = ts 
        dropped = None
        
    t, spread = consensus_time(kept)
    return {"when": t, "spread_s": spread, "naive_when": t_naive, "dropped_family": dropped,
            "sources": sorted(kept)}

def timeline(groups: list[dict], conflict_probs, threshold: float = 0.5) -> pd.DataFrame:
    rows = []
    for g, p in zip(groups, conflict_probs):
        r = reconcile(g, p, threshold)
        users = pd.Series(g["users"]).mode()
        rows.append({
            "case": g["case"],
            "who": users.iloc[0] if len(users) else "unknown",
            "what": f"executed/touched {g['key']}",
            "when_utc": pd.to_datetime(r["when"], unit="s", utc=True).strftime("%Y-%m-%d %H:%M:%S"),
            "sources": "+".join(r["sources"]), "n_sources": len(r["sources"]),
            "spread_s": round(r["spread_s"], 1), "conflict_prob": round(float(p), 4),
            "confidence": round(float(1 - p), 4),
            "dropped_source": r["dropped_family"] or "",
        })
    return pd.DataFrame(rows).sort_values("when_utc").reset_index(drop=True)