# Diverging Truths: Evidence Correlation and Temporal Reconstruction for Digital Forensics

Author: Tshephang Matlala \
Module: Computer Forensics \
Theme: who did what, when

Multi-source Windows forensic evidence from the ADFIR dataset ("Modified EZ tools dataset": CSV exports of
EZ Tools parsers) is grouped by the executable program it describes. A supervised model estimates, for each
group, the probability that one source's timestamps disagree with the others (a *conflict*); a reconciliation
step then sets aside the most divergent source and produces a who / what / when timeline with a confidence value.

The five CTF cases are used: Magnet CTF 2019, Magnet CTF 2022, NIST Data Leakage Case, Stolen Szechuan Sauce DC
and Desktop. They are evaluated **leave-one-case-out**: each case is held out once, models are trained on the
other four, and results are reported per case and macro-averaged. The Simulated Attackers' Techniques part of
ADFIR (a SQL Server backup) is not used: it is not CSV and needs far more memory.

Dataset: ADFIR forensic dataset, Sokol et al., <https://data.mendeley.com/datasets/65g9gm8zrd/1>
(DOI 10.17632/65g9gm8zrd.1).

## Pipeline

```
EZ Tools CSVs -> ingest (10 adapters, 6 source families) -> group by (case, exe name)
  -> tamper injection (ground-truth labels) -> features -> LR (fully Implemented) & Random Forest
  -> conflict probability -> reconcile (drop most divergent source) -> who / what / when timeline
```

Source families: registry (UserAssist, BAM), execution (Prefetch, Amcache, AppCompatCache),
eventlog (EvtxECmd), filesystem ($MFT, $J), filefolder (LNK), deletion (Recycle Bin).

## Project Layout

```
source/dtr/     ingest.py groups.py models.py evaluate.py reconstruct.py main.py vis.py synth.py
                drift_experimental.py   (parked clock-drift ideas, not used - see its docstring)
tests/          pytest suite (synthetic fixtures)
data/           ADFIR forensic dataset/ (git-ignored), processed/, outputs/
docs/           manuscript/ proposals/ references/ notes/
archive/        earlier Java prototype; scratch/ = parked threshold scripts
```

## Run

```bash
pip install -r requirements.txt
python -m pytest -q
cd source
python -m dtr.main run                      # all 5 cases, leave-one-case-out, default data path
python -m dtr.main run --tune-threshold     # decision threshold tuned on a validation case (not the test case)
python -m dtr.main run --holdout Magnet_CTF_2022
python -m dtr.vis                           # plots from data/outputs
python -m dtr.main synth --out ../data/synthetic   # SYNTHETIC data: validates the code, not the research claims
```

Outputs in `data/outputs/`: `metrics.json` (per case + macro average, incl. per-family-pair clock offsets),
`predictions.csv`, `reconstructed_timeline.csv`, `loss_history.csv`, plots.

## Method limitations

- **Labels are synthetic.** ADFIR has no "these records are the same event" ground truth. Positives are made by
  shifting one source family's timestamps by 1 min to 30 days (log-uniform); negatives are untouched real groups,
  so natural disagreement in real data counts as "consistent". Read results with this in mind, and compare against
  the `baseline_gap_max_roc_auc` and `rf_recall_by_clock_shift` fields in `metrics.json`.
- Six source families (registry, execution, eventlog, filesystem, filefolder, deletion); a group needs >= 2.
  Reconciliation can only decide *which* source to distrust when a group has >= 3 families.
- The reconciliation rule drops the family whose timestamps are furthest from the others. It does not favour any
  source type by design.
- `is_attack_related` is carried through ingestion but not used as a label (relevance is not consistency).
- Embeddings are not used. Clock drift is not corrected: run `main run` and inspect `family_offsets` first -
  medians near 0 mean there is nothing to correct.
- "who" is only known for sources that carry a user (UserAssist hive path, EventLog UserName).
- AppCompatCache time is a file modification time, not an execution time (weaker evidence).
- `effort_reduction` is a compression ratio (raw events vs groups), not a measured saving in investigator time.

## References

1. Sokol et al., ADFIR forensic dataset, Data in Brief / Mendeley Data. <https://www.sciencedirect.com/science/article/pii/S2352340924006607>

## Per-case normalisation and multiple seeds

Natural gaps between artifacts differ strongly between cases (e.g. eventlog→filesystem median offset ≈2 s in the DC image vs ≈2334 s in Magnet 2022), which breaks raw-gap features under leave-one-case-out. `--normalise` converts gap/span features to robust z-scores using each case's own untampered data (no labels). `--seeds N` repeats tamper injection and reports mean ± sd.

    python -m dtr.main run                       # raw features
    python -m dtr.main run --normalise --seeds 5

Notes: on real data only three families (eventlog, filesystem, execution) carry substantial volume; execution↔eventlog agree to under 1 s in every case, so clock-drift correction appears unnecessary. Random Forest vs Logistic Regression results are mixed. Normalisation is validated on a controlled synthetic case (AUC 0.51 → 0.94) but not yet on real data.
