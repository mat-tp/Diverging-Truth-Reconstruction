"""EXPERIMENTAL / NOT WIRED IN - your clock-drift anchor ideas (from todos.txt), moved out of ingest.py.

Why it is parked: `detect_robust_clock_drift` calls `calculate_prefetch_drift`, which was never written;
`detect_case_clock_drift` always returns 0.0; the BAM/UserAssist branch reads columns (`ValueName`,
`LastExecuted` on BAM) that do not exist in the EZ Tools output; and nothing calls any of it.
Before reviving it, run the offset diagnostic (`ingest.family_offset_report`, printed by `main run`) -
if the per-family-pair median offsets are ~0, there is no clock drift to correct in this dataset.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from pathlib import Path

def detect_case_clock_drift(case_path: str) -> float:
    """
    Identifies shared anchor events between Windows Event Logs and filesystem artefacts
    to calculate the median baseline clock skew in seconds.
    """
    path = Path(case_path)
    evtx_files = list(path.rglob("*EvtxECmd_Output.csv"))
    prefetch_files = list(path.rglob("*PECmd_Output.csv"))
    
    if not evtx_files or not prefetch_files:
        return 0.0
        
    try:
        evtx_df = pd.read_csv(evtx_files[0], low_memory=False)
        pf_df = pd.read_csv(prefetch_files[0], low_memory=False)
        
        evtx_df["TimeGenerated"] = pd.to_datetime(evtx_df["TimeGenerated"], errors="coerce")
        pf_df["SourceCreated"] = pd.to_datetime(pf_df["SourceCreated"], errors="coerce")
        
        # Isolate Event ID 4688 (Process Creation) for anchor matching
        process_events = evtx_df[evtx_df["EventId"] == 4688].dropna(subset=["TimeGenerated"])
        
        # In a complete implementation, you will merge these dataframes on the executable name
        # to find intersecting timestamps and calculate the difference in seconds.
        # offsets = (pf_df["SourceCreated"] - process_events["TimeGenerated"]).dt.total_seconds()
        # return float(np.median(offsets))
        
        return 0.0
    except Exception as e:
        print(f"Error calculating drift for {case_path}: {e}")
        return 0.0

def apply_temporal_standardisation(events: list, offset_seconds: float) -> list:
    """Applies the calculated median offset to normalise timelines."""
    for event in events:
        event["timestamp_corrected"] = event["timestamp"] + offset_seconds
    return events

def calculate_registry_execution_drift(case_path: str) -> float:
    """
    Extracts UserAssist and BAM execution timestamps to calculate 
    median clock skew against Prefetch creation times.
    """
    path = Path(case_path)
    
    # Locate the necessary CSV files
    bam_files = list(path.rglob("*BAM*.csv"))
    ua_files = list(path.rglob("*UserAssist*.csv"))
    prefetch_files = list(path.rglob("*PECmd_Output.csv"))
    
    if not prefetch_files or (not bam_files and not ua_files):
        return None
        
    registry_records = []
    
    # 1. Parse BAM Records
    if bam_files:
        try:
            bam_df = pd.read_csv(bam_files[0], low_memory=False)
            bam_df = bam_df.dropna(subset=["ValueName", "LastExecuted"])
            bam_df["reg_timestamp"] = pd.to_datetime(bam_df["LastExecuted"], errors="coerce")
            # Extract just the executable name (e.g., "cmd.exe") and lowercase it
            bam_df["exe_name"] = bam_df["ValueName"].apply(lambda x: str(x).split("\\")[-1].lower())
            registry_records.append(bam_df[["reg_timestamp", "exe_name"]])
        except Exception as e:
            print(f"BAM parsing error: {e}")

    # 2. Parse UserAssist Records
    if ua_files:
        try:
            ua_df = pd.read_csv(ua_files[0], low_memory=False)
            # EZ Tools automatically decodes ROT13 UserAssist values
            ua_df = ua_df.dropna(subset=["ValueName", "LastExecuted"])
            ua_df["reg_timestamp"] = pd.to_datetime(ua_df["LastExecuted"], errors="coerce")
            ua_df["exe_name"] = ua_df["ValueName"].apply(lambda x: str(x).split("\\")[-1].lower())
            registry_records.append(ua_df[["reg_timestamp", "exe_name"]])
        except Exception as e:
            print(f"UserAssist parsing error: {e}")
            
    if not registry_records:
        return None
        
    # Combine all extracted registry execution anchors
    reg_df = pd.concat(registry_records, ignore_index=True).dropna()
    
    # 3. Parse Prefetch Records (Filesystem Anchor)
    try:
        pf_df = pd.read_csv(prefetch_files[0], low_memory=False)
        pf_df = pf_df.dropna(subset=["ExecutableName", "SourceCreated"])
        pf_df["pf_timestamp"] = pd.to_datetime(pf_df["SourceCreated"], errors="coerce")
        pf_df["exe_name"] = pf_df["ExecutableName"].str.lower()
        
        # 4. Merge DataFrames on the executable name
        # We merge to find instances where the registry says an app ran, 
        # and the filesystem shows a Prefetch file was created for it.
        merged = pd.merge(reg_df, pf_df[["exe_name", "pf_timestamp"]], on="exe_name", how="inner")
        
        # Filter for timestamps that are highly correlated (e.g., within 24 hours) 
        # to prevent matching a BAM record from today with a Prefetch file from last month.
        merged["offset_seconds"] = (merged["pf_timestamp"] - merged["reg_timestamp"]).dt.total_seconds()
        valid_anchors = merged[abs(merged["offset_seconds"]) < 86400] 
        
        if valid_anchors.empty:
            return None
            
        # Return the median offset across all matched registry/prefetch pairs
        return float(np.median(valid_anchors["offset_seconds"]))
        
    except Exception as e:
        print(f"Prefetch/Merge error: {e}")
        return None

def detect_robust_clock_drift(case_path: str) -> dict:
    """
    Attempts multiple tiers of anchor detection to establish clock skew.
    Returns the offset in seconds and a confidence boolean.
    """
    # Tier 1: Process Creation vs Prefetch
    primary_offset = calculate_prefetch_drift(case_path)
    if primary_offset is not None:
        return {"offset": primary_offset, "confidence": True, "method": "Prefetch"}
        
    # Tier 2: Registry Execution (UserAssist/BAM) vs MFT
    secondary_offset = calculate_registry_execution_drift(case_path)
    if secondary_offset is not None:
        return {"offset": secondary_offset, "confidence": True, "method": "Registry"}
        
    # Tier 3: Event Burst Alignment (Relative chronologies)
    burst_offset = calculate_event_density_alignment(case_path)
    if burst_offset is not None:
        return {"offset": burst_offset, "confidence": False, "method": "Burst Alignment"}
        
    # Fallback: Zero-drift assumption
    return {"offset": 0.0, "confidence": False, "method": "None - Raw Timestamps"}

def calculate_event_density_alignment(case_path: str, bin_freq: str = '1min') -> float:
    """
    Calculates the temporal shift between filesystem and event log activity 
    bursts using signal cross-correlation.
    """
    path = Path(case_path)
    
    mft_files = list(path.rglob("*MFTECmd_$MFT_Output.csv"))
    evtx_files = list(path.rglob("*EvtxECmd_Output.csv"))
    
    if not mft_files or not evtx_files:
        return None
        
    try:
        # 1. Load and parse the datasets
        mft_df = pd.read_csv(mft_files[0], low_memory=False, usecols=["Created0x10"])
        evtx_df = pd.read_csv(evtx_files[0], low_memory=False, usecols=["TimeGenerated"])
        
        mft_df["timestamp"] = pd.to_datetime(mft_df["Created0x10"], errors="coerce")
        evtx_df["timestamp"] = pd.to_datetime(evtx_df["TimeGenerated"], errors="coerce")
        
        mft_df = mft_df.dropna(subset=["timestamp"])
        evtx_df = evtx_df.dropna(subset=["timestamp"])
        
        if mft_df.empty or evtx_df.empty:
            return None

        # 2. Floor timestamps to the chosen frequency (default: 1 minute)
        mft_df["bin"] = mft_df["timestamp"].dt.floor(bin_freq)
        evtx_df["bin"] = evtx_df["timestamp"].dt.floor(bin_freq)
        
        # Count the number of events in each time bucket
        mft_counts = mft_df.groupby("bin").size()
        evtx_counts = evtx_df.groupby("bin").size()
        
        # 3. Standardise the index range so both signals match in length
        min_time = min(mft_counts.index.min(), evtx_counts.index.min())
        max_time = max(mft_counts.index.max(), evtx_counts.index.max())
        full_range = pd.date_range(min_time, max_time, freq=bin_freq)
        
        # Reindex fills empty minutes with 0
        mft_sig = mft_counts.reindex(full_range, fill_value=0).values
        evtx_sig = evtx_counts.reindex(full_range, fill_value=0).values
        
        # 4. Perform cross-correlation to find the optimal lag
        correlation = np.correlate(mft_sig, evtx_sig, mode="full")
        
        # The lags array maps the correlation output to specific bin shifts
        lags = np.arange(-len(evtx_sig) + 1, len(mft_sig))
        best_lag_bins = lags[np.argmax(correlation)]
        
        # Convert the bin lag back into seconds (1 minute bin = 60 seconds)
        offset_seconds = float(best_lag_bins * pd.Timedelta(bin_freq).total_seconds())
        
        return offset_seconds
        
    except Exception as e:
        print(f"Density alignment error: {e}")
        return None
    
