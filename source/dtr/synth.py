"""SYNTHETIC case generator in the ADFIR / EZ Tools CSV schema.

For development and tests only (the real ADFIR files are too large to ship). Each program has one
true execution time; every artifact reports it with source-specific jitter, and ~15% of programs
get a large *benign* divergence (e.g. a later re-run) so that natural disagreement exists.
Results on this data validate the pipeline, NOT the research claims.
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd

CASES = ["Magnet_CTF_2019", "Magnet_CTF_2022", "NIST_Data_Leakage_Case",
         "Stolen_Szechuan_Sauce_DC", "Stolen_Szechuan_Sauce_Desktop"]
_T0 = pd.Timestamp("2022-01-10", tz="UTC").timestamp()
_JIT = {"userassist": 4, "bam": 6, "prefetch": 2, "amcache": 40, "appcompat": 90,
        "evtx": 3, "mft": 15, "lnk": 25, "recycle": 20}

def _fmt(t):
    return pd.to_datetime(t, unit="s", utc=True).strftime("%Y-%m-%d %H:%M:%S.%f")

def make_case(path: str, n_programs: int = 300, seed: int = 0, users=("alice", "bob")):
    rng = np.random.default_rng(seed)
    rows = {k: [] for k in _JIT}
    names = [f"prog{i:03d}.exe" for i in range(n_programs)]
    for name in names:
        t0 = _T0 + rng.uniform(0, 40 * 86400)
        user = rng.choice(users)
        present = [k for k in _JIT if rng.random() < 0.55]
        if len(present) < 2:
            present = list(rng.choice(list(_JIT), 2, replace=False))
        benign = rng.choice(present) if rng.random() < 0.15 else None
        for k in present:
            t = t0 + rng.normal(0, _JIT[k])
            if k == benign:
                t += rng.lognormal(8, 1.2)
            rows[k].append((name, t, user))
    d = lambda sub: os.makedirs(os.path.join(path, sub), exist_ok=True)
    for sub in ["Registry/x", "ProgramExecution", "EventLogs", "FileSystem", "FileFolderAccess", "FileDeletion"]:
        d(sub)
    W = lambda sub, fname, df: df.assign(is_attack_related=0).to_csv(os.path.join(path, sub, fname), index=False)
    for u in users:
        sel = [(n, t) for n, t, uu in rows["userassist"] if uu == u]
        W("Registry/x", f"20250101_UserAssist__D_Users_{u}_NTUSER.DAT.csv",
          pd.DataFrame({"ProgramName": [f"C:\\Apps\\{n}" for n, _ in sel], "LastExecuted": [_fmt(t) for _, t in sel]}))
    W("Registry/x", "20250101_BamDam__D_Windows_System32_config_SYSTEM.csv",
      pd.DataFrame({"Program": [f"\\Device\\HarddiskVolume3\\Apps\\{n}" for n, _, _ in rows["bam"]],
                    "ExecutionTime": [_fmt(t) for _, t, _ in rows["bam"]]}))
    pf = pd.DataFrame({"ExecutableName": [n.upper() for n, _, _ in rows["prefetch"]],
                       "LastRun": [_fmt(t) for _, t, _ in rows["prefetch"]]})
    W("ProgramExecution", "20250101_PECmd_Output.csv", pf)
    W("ProgramExecution", "20250101_Amcache_UnassociatedFileEntries.csv",
      pd.DataFrame({"FullPath": [f"c:\\apps\\{n}" for n, _, _ in rows["amcache"]],
                    "FileKeyLastWriteTimestamp": [_fmt(t) for _, t, _ in rows["amcache"]]}))
    W("ProgramExecution", "20250101_Win10_SYSTEM_AppCompatCache.csv",
      pd.DataFrame({"Path": [f"C:\\Apps\\{n}" for n, _, _ in rows["appcompat"]],
                    "LastModifiedTimeUTC": [_fmt(t) for _, t, _ in rows["appcompat"]]}))
    W("EventLogs", "20250101_EvtxECmd_Output.csv",
      pd.DataFrame({"TimeCreated": [_fmt(t) for _, t, _ in rows["evtx"]], "UserName": [u for _, _, u in rows["evtx"]],
                    "ExecutableInfo": [f"C:\\Apps\\{n} /run" for n, _, _ in rows["evtx"]]}))
    W("FileSystem", "20250101_MFTECmd_$MFT_Output.csv",
      pd.DataFrame({"FileName": [n for n, _, _ in rows["mft"]], "Extension": ".exe",
                    "Created0x10": [_fmt(t) for _, t, _ in rows["mft"]]}))
    W("FileFolderAccess", "20250101_LECmd_Output.csv",
      pd.DataFrame({"LocalPath": [f"C:\\Apps\\{n}" for n, _, _ in rows["lnk"]],
                    "TargetAccessed": [_fmt(t) for _, t, _ in rows["lnk"]]}))
    W("FileDeletion", "20250101_RBCmd_Output.csv",
      pd.DataFrame({"FileName": [f"C:\\Apps\\{n}" for n, _, _ in rows["recycle"]],
                    "DeletedOn": [_fmt(t) for _, t, _ in rows["recycle"]]}))

def make_dataset_dir(root: str, n_programs: int = 300, seed: int = 0):
    for i, c in enumerate(CASES):
        make_case(os.path.join(root, c), n_programs, seed + i)