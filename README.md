# Diverging Truths: Evidence Correlation and Temporal Reconstruction for Digital Forensics

Author: Tshephang Matlala
Module: Computer Forensics
Theme: who did what, when

Multi-source Windows forensic evidence from the ADFIR Mendeley dataset is processed to automatically detect cross-source temporal disagreement. The pipeline ingests the pre-processed Plaso CSV exports, focusing specifically on event logs and file system metadata to establish a reliable sequence of actions. These disparate data sources are grouped by the executable program they describe and evaluated by a supervised machine learning model. This model calculates a conflict probability for each event group and reconciles them into a unified, confidence-scored timeline.

The project relies on the ADFIR forensic dataset compiled by Sokol et al., which provides harmonised feature extracts from scenarios including Magnet CTF 2019, Magnet CTF 2022, the NIST Data Leakage Case, and the Stolen Szechuan Sauce environments. The system ingests the CSV outputs directly through two dedicated adapters and applies adversarial tampering injections to generate synthetic ground-truth labels for training. The engineered features are then fed into Logistic Regression and Random Forest classifiers. A conflict probability is generated, and the reconciliation engine attempts to drop the most divergent source, which is typically the highly manipulated file system artefact, to finalise the timeline.

You can configure and run the pipeline entirely from the command line interface provided in the source directory. Installing the dependencies via the requirements file ensures the environment is prepared for execution. You must place the downloaded Mendeley case folders into the raw data directory before initiating the primary run command, which targets a specific holdout case for evaluation. A synthetic data generation command is also included to validate the codebase architecture locally without requiring the massive raw dataset downloads upfront.

The methodology operates under specific structural constraints due to the nature of the harmonised dataset. Labels are purely synthetic because the original corpus lacks explicit ground truth connecting different records to the exact same execution event. The engine produces positive conflict examples by artificially shifting one source's clock by a log-uniform margin between one minute and thirty days. Any natural disagreement inherent in the underlying raw data is currently counted as consistent behaviour. The system is also limited by having only two primary artefact families, meaning traditional statistical consensus voting is impossible. The reconciliation script therefore relies on a strict heuristic that favours event logs over file system metadata when attempting to resolve a detected conflict.

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
source/dtr/     ingest.py groups.py models.py evaluate.py reconstruct.py synth.py cli.py viz.py
tests/          pytest suite + 3-row real fixtures from Magnet_CTF_2022
data/           raw/ processed/ outputs/ (git-ignored contents)
docs/           manuscript/ proposals/ references/ notes/
archive/        earlier Java logistic-regression prototype (kept for reference)
```

## Run

```bash
pip install -r requirements.txt
python -m pytest -q
cd source
# put the ADFIR "Modified EZ tools dataset" case folders in ../data/raw/ , then:
python -m dtr.cli run --data ../data/raw --holdout Magnet_CTF_2022 --out ../data/outputs
python -m dtr.viz ../data/outputs
# no data yet? SYNTHETIC smoke test (validates the code, not the research claims):
python -m dtr.cli synth --out ../data/synthetic && python -m dtr.cli run --data ../data/synthetic
```

Outputs: `metrics.json`, `loss_history.csv`, `predictions.csv`, `reconstructed_timeline.csv`, plots.

## Method limitations

- **Labels are synthetic.** ADFIR has no "these records are the same event" ground truth. Positives
  are produced by shifting one source's clock by 1 min to 30 days (log-uniform); negatives are
  untouched real groups. Natural, unlabelled disagreement in real data is therefore counted as
  "consistent".
- Train on four cases, evaluate on the held-out case (Magnet_CTF_2022, mirroring ADFIR's train/val split).
- `is_attack_related` is carried through ingestion but is not used as a label (relevance != consistency).
- Embeddings are not wired in yet (only a 3-line sample was available). Add embedding cosine
  similarity as an extra feature in `groups.features` to run the structured-vs-embedding experiment.
- Reconciliation needs >= 3 sources to decide which one to distrust.
- "who" is only known for sources that carry a user (UserAssist hive path, EventLog UserName).
- AppCompatCache time is the file's modification time, not an execution time; treat it as weaker evidence.
- `Simulated attackers' techniques dataset` part of the ADFIR forensic dataset is not used through the implementation as it makes the problem too complex , requires larger memory (The dataset is stored in a database format rather than using .csv files). And am happy to expand in the future.

## References

1. (https://www.sciencedirect.com/science/article/pii/S2352340924006607)[Data Preparation]
