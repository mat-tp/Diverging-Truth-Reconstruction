"""Evidence ingestion: ADFIR 'Modified EZ tools dataset' CSVs -> one normalised event table.

Normalised columns: family, artifact, ts (UTC epoch seconds), user, key (lower-case exe name), attack.
Only executable-related evidence (key ends with .exe) is kept, since the claim being
corroborated is "program X was run/touched at time T".
"""
from __future__ import annotations
import glob, os, re
import pandas as pd

FAMILIES = ["registry", "execution", "eventlog", "filesystem", "filefolder", "deletion"]
COLS = ["family", "artifact", "ts", "user", "key", "attack"]
_USER_FROM_PATH = re.compile(r"_Users_([^_]+?)_NTUSER", re.I)
_EXE_IN_TEXT = re.compile(r"([^\\/\s\"'|;,]+\.exe)", re.I)

def to_epoch(s: pd.Series) -> pd.Series:
    dt = pd.to_datetime(s, utc=True, errors="coerce")
    return (dt - pd.Timestamp("1970-01-01", tz="UTC")).dt.total_seconds()

def exe_key(s: pd.Series) -> pd.Series:
    base = s.astype("string").str.replace("\\", "/", regex=False).str.rsplit("/", n=1).str[-1]
    base = base.str.lower().str.strip()
    return base.where(base.str.endswith(".exe", na=False))

def _attack(df: pd.DataFrame) -> pd.Series:
    if "is_attack_related" in df:
        return pd.to_numeric(df["is_attack_related"], errors="coerce").fillna(0).astype(int)
    return pd.Series(0, index=df.index)

def _frame(df, family, artifact, ts, key, user=None):
    out = pd.DataFrame({
        "family": family, "artifact": artifact, "ts": ts, "key": key,
        "user": user if user is not None else pd.NA, "attack": _attack(df),
    })
    return out.dropna(subset=["ts", "key"])[COLS]

def _multi(df, family, artifact, key, ts_cols, user=None):
    parts = []
    for c in ts_cols:
        if c in df:
            parts.append(_frame(df, family, artifact, to_epoch(df[c]), key, user))
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=COLS)

def _first(df, names):
    for n in names:
        if n in df:
            return n
    return None

def a_userassist(df, path):
    m = _USER_FROM_PATH.search(os.path.basename(path))
    user = m.group(1) if m else None
    return _frame(df, "registry", "UserAssist", to_epoch(df["LastExecuted"]), exe_key(df["ProgramName"]), user)

def a_bam(df, path):
    return _frame(df, "registry", "BAM", to_epoch(df["ExecutionTime"]), exe_key(df["Program"]))

def a_pecmd(df, path):
    ts_cols = ["RunTime", "LastRun"] + [f"PreviousRun{i}" for i in range(7)]
    return _multi(df, "execution", "Prefetch", exe_key(df["ExecutableName"]), ts_cols)

def a_amcache(df, path):
    c = _first(df, ["FileKeyLastWriteTimestamp", "FileIDLastWriteTimestamp"])
    return _frame(df, "execution", "Amcache", to_epoch(df[c]), exe_key(df["FullPath"])) if c else pd.DataFrame(columns=COLS)

def a_appcompat(df, path):
    return _frame(df, "execution", "AppCompatCache", to_epoch(df["LastModifiedTimeUTC"]), exe_key(df["Path"]))

def a_evtx(df, path):
    txt = pd.Series("", index=df.index)
    for c in ["ExecutableInfo", "Payload"] + [f"PayloadData{i}" for i in range(1, 7)]:
        if c in df:
            txt = txt + " " + df[c].fillna("").astype(str)
    key = txt.str.extract(_EXE_IN_TEXT, expand=False).str.lower()
    user = df["UserName"] if "UserName" in df else None
    return _frame(df, "eventlog", "EventLog", to_epoch(df["TimeCreated"]), key, user)

def a_mft(df, path):
    if "Extension" in df:
        df = df[df["Extension"].astype(str).str.lower() == ".exe"]
    return _multi(df, "filesystem", "MFT", exe_key(df["FileName"]), ["Created0x10", "LastModified0x10", "LastAccess0x10"])

def a_usnj(df, path):
    if "Extension" in df:
        df = df[df["Extension"].astype(str).str.lower() == ".exe"]
    return _frame(df, "filesystem", "UsnJrnl", to_epoch(df["UpdateTimestamp"]), exe_key(df["Name"]))

def a_lnk(df, path):
    return _multi(df, "filefolder", "LNK", exe_key(df["LocalPath"]), ["TargetAccessed", "TargetModified"])

def a_recycle(df, path):
    return _frame(df, "deletion", "RecycleBin", to_epoch(df["DeletedOn"]), exe_key(df["FileName"]))

ADAPTERS = [
    ("*UserAssist__*.csv", a_userassist), ("*BamDam__*.csv", a_bam),
    ("*PECmd_Output.csv", a_pecmd), ("*Amcache_*FileEntries.csv", a_amcache),
    ("*AppCompatCache.csv", a_appcompat), ("*EvtxECmd_Output.csv", a_evtx),
    ("*MFTECmd_$MFT_Output.csv", a_mft), ("*MFTECmd_$J_Output.csv", a_usnj),
    ("*LECmd_Output.csv", a_lnk), ("*RBCmd_Output.csv", a_recycle),
]

def load_case(case_dir: str, case_name: str | None = None) -> pd.DataFrame:
    frames = []
    for pat, fn in ADAPTERS:
        for f in sorted(glob.glob(os.path.join(case_dir, "**", pat), recursive=True)):
            try:
                df = pd.read_csv(f, low_memory=False)
                part = fn(df, f)
            except (KeyError, ValueError, pd.errors.EmptyDataError) as e:
                print(f"[ingest] skipped {os.path.basename(f)}: {e!r}")
                continue
            if len(part):
                frames.append(part)
    out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=COLS)
    out["case"] = case_name or os.path.basename(os.path.normpath(case_dir))
    return out