"""Compare the reconstructed REAL timeline with independently documented events (published case write-ups).

The documented file is supplied by the investigator/researcher (it must come from the write-up, not from this
tool). Columns: case, event, key (executable or file name as it appears in the evidence), time_utc.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

TEMPLATE_COLUMNS = ["case", "event", "key", "time_utc"]
TOLERANCES = {"within_1min": 60, "within_5min": 300, "within_1h": 3600}


def match_documented(timeline: pd.DataFrame, documented: pd.DataFrame, threshold: float = 0.5) -> pd.DataFrame:
    """One row per documented event: the matching reconstructed row, the time error and the conflict flag."""
    need = set(TEMPLATE_COLUMNS) - set(documented.columns)
    if need:
        raise ValueError(f"documented events file is missing columns: {sorted(need)}")
    tl = timeline.copy()
    tl["key"] = tl["what"].str.replace("executed/touched ", "", regex=False).str.lower()
    tl["t"] = pd.to_datetime(tl["when_utc"], utc=True)
    rows = []
    for _, d in documented.iterrows():
        doc_t = pd.to_datetime(d["time_utc"], utc=True)
        hit = tl[(tl["case"] == d["case"]) & (tl["key"] == str(d["key"]).strip().lower())]
        if hit.empty:
            rows.append({**d.to_dict(), "found": False, "reconstructed_utc": None, "error_s": np.nan,
                         "conflict_prob": np.nan, "flagged_conflicting": None, "sources": ""})
            continue
        i = (hit["t"] - doc_t).abs().idxmin()
        r = hit.loc[i]
        rows.append({**d.to_dict(), "found": True, "reconstructed_utc": r["when_utc"],
                     "error_s": float((r["t"] - doc_t).total_seconds()),
                     "conflict_prob": float(r["conflict_prob"]),
                     "flagged_conflicting": bool(r["conflict_prob"] >= threshold), "sources": r["sources"]})
    return pd.DataFrame(rows)


def summarise(matches: pd.DataFrame) -> dict:
    found = matches[matches["found"]]
    out = {"documented": int(len(matches)), "found": int(len(found))}
    if len(found):
        a = found["error_s"].abs()
        out.update({k: float((a <= v).mean()) for k, v in TOLERANCES.items()})
        out["median_abs_error_s"] = float(a.median())
        out["share_flagged_conflicting"] = float(found["flagged_conflicting"].mean())
    return out
