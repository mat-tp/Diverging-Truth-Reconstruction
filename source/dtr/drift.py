"""Anchor-event clock-drift detection and correction (research question 1).

Anchor events = the same program (exe name) seen by two independent source families. If one source's clock
is wrong by d seconds, the signed gaps to the nearest anchor in a reference family move by d, so the median
of those gaps estimates d robustly (a few tampered or unrelated events do not move a median).

Caveat reported by the experiment: families also differ by a *natural* offset (e.g. a file's creation time
is not its execution time), so the estimate is only meaningful relative to that baseline. The experiment
therefore injects a KNOWN offset and measures how well the change in the estimate recovers it.
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd
from .ingest import load_case

REFERENCE = "execution"      # execution <-> eventlog agree to < 1 s in every case (family_offset_report)
MAX_GAP = 43200.0            # anchors are accepted within +-12 h
MIN_ANCHORS = 20


def index_events(events: pd.DataFrame) -> dict:
    """Sorted timestamps per (case, program, family): built once, reused for every estimate."""
    idx: dict = {}
    for (case, key, fam), g in events.groupby(["case", "key", "family"], sort=False):
        idx.setdefault((case, key), {})[fam] = np.sort(g["ts"].to_numpy())
    return idx


def _gaps(idx: dict, fam: str, ref: str, max_gap: float, shift: float = 0.0) -> np.ndarray:
    out = []
    for fams in idx.values():
        if fam not in fams or ref not in fams:
            continue
        a, b = fams[fam] + shift, fams[ref]
        j = np.clip(np.searchsorted(b, a), 1, len(b) - 1) if len(b) > 1 else np.zeros(len(a), int)
        cand = np.stack([b[np.clip(j - 1, 0, len(b) - 1)], b[j]])
        d = cand[np.abs(cand - a).argmin(axis=0), np.arange(len(a))] - a
        out.append(d[np.abs(d) <= max_gap])
    return np.concatenate(out) if out else np.array([])


def estimate_offsets(events: pd.DataFrame, ref: str = REFERENCE, max_gap: float = MAX_GAP,
                     min_anchors: int = MIN_ANCHORS, idx: dict | None = None) -> dict:
    """Offset to ADD to each family's timestamps to line it up with `ref`, plus anchor count and spread."""
    idx = idx if idx is not None else index_events(events)
    res = {}
    for fam in sorted(set(events["family"]) - {ref}):
        d = _gaps(idx, fam, ref, max_gap)
        if len(d) >= min_anchors:
            res[fam] = {"offset_s": float(np.median(d)), "n_anchors": int(len(d)),
                        "mad_s": float(np.median(np.abs(d - np.median(d))))}
    return res


def apply_offsets(events: pd.DataFrame, offsets: dict) -> pd.DataFrame:
    out = events.copy()
    for fam, o in offsets.items():
        out.loc[out["family"] == fam, "ts"] += o["offset_s"] if isinstance(o, dict) else o
    return out


def recover(idx: dict, fam: str, injected_s: float, ref: str = REFERENCE, max_gap: float = MAX_GAP,
            min_anchors: int = MIN_ANCHORS) -> dict | None:
    """Inject a clock drift into `fam`, estimate it from anchors, correct it, and report the error.

    The family's clock reads `injected_s` too late (ts + injected_s). The estimate of how much to add to align
    it with the reference moves by -injected_s, so the detected drift is (before - after).
    """
    d0 = _gaps(idx, fam, ref, max_gap)
    d1 = _gaps(idx, fam, ref, max_gap, shift=injected_s)
    if len(d0) < min_anchors or len(d1) < min_anchors:
        return None
    before, after = float(np.median(d0)), float(np.median(d1))
    detected = before - after
    d2 = _gaps(idx, fam, ref, max_gap, shift=injected_s - detected)      # apply the correction
    resid = float(np.median(d2)) - before if len(d2) >= min_anchors else float("nan")
    return {"family": fam, "injected_s": injected_s, "detected_s": detected,
            "abs_error_s": abs(detected - injected_s), "residual_after_correction_s": resid,
            "n_anchors": int(len(d1)), "baseline_offset_s": before}


def run_drift_experiment(data_dir: str, out_dir: str, offsets=(-21600, -3600, -600, -60, 60, 600, 3600, 21600),
                         families=("eventlog", "filesystem")) -> pd.DataFrame:
    os.makedirs(out_dir, exist_ok=True)
    rows = []
    for c in sorted(d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))):
        idx = index_events(load_case(os.path.join(data_dir, c), c))
        for fam in families:
            for off in offsets:
                r = recover(idx, fam, float(off))
                if r:
                    rows.append({"case": c, **r})
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out_dir, "drift_experiment.csv"), index=False)
    return df
