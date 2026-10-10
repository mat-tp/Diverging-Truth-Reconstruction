"""Plots from files written by main.run (never trains)."""
import json, sys
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

DEFAULT = Path(__file__).resolve().parents[2] / "data" / "outputs"


def main(out_dir=DEFAULT):
    d = Path(out_dir)
    
    ax = pd.read_csv(d / "loss_history.csv").set_index("iteration").plot(title="Logistic regression training loss (one curve per fold)")
    ax.set_ylabel("loss")
    plt.savefig(d / "loss_curve.png", dpi=150, bbox_inches="tight")
    plt.close()

    m = json.load(open(d / "metrics.json"))
    names = ["logistic_regression_scratch", "random_forest"]
    for score in ("f1", "roc_auc"):
        df = pd.DataFrame({n: {c: m["per_case"][c][n].get(score) for c in m["cases"]} for n in names})
        ax = df.plot.bar(title=f"{score.upper()} per held-out case ({m['protocol']})", rot=20)
        if score == "roc_auc":
            ax.axhline(0.5, color="grey", ls="--", lw=1)  # chance level
        plt.ylim(0, 1.05)
        plt.savefig(d / f"{score}_per_case.png", dpi=150, bbox_inches="tight")
        plt.close()

    preds = pd.read_csv(d / "predictions.csv")
    plt.figure(figsize=(8, 5))
    preds[preds["truth"] == 0]["p_rf"].plot.hist(alpha=0.6, bins=25, label="Authentic", color="steelblue")
    preds[preds["truth"] == 1]["p_rf"].plot.hist(alpha=0.6, bins=25, label="Tampered", color="darkorange")
    plt.title("Conflict Probability Distribution")
    plt.xlabel("Predicted Probability of Tampering")
    plt.ylabel("Frequency")
    plt.legend()
    plt.savefig(d / "probability_distribution.png", dpi=150, bbox_inches="tight")
    plt.close()

    timeline = pd.read_csv(d / "reconstructed_timeline.csv")
    
    dropped = timeline[timeline["dropped_source"].notna()]["dropped_source"].value_counts()
    if not dropped.empty:
        plt.figure(figsize=(8, 5))
        dropped.plot.bar(color="crimson", alpha=0.8)
        plt.title("Tampered Evidence by Source Type")
        plt.xlabel("Forensic Artefact Source")
        plt.ylabel("Frequency of Manipulation")
        plt.xticks(rotation=45)
        plt.savefig(d / "tampered_sources_frequency.png", dpi=150, bbox_inches="tight")
        plt.close()

    timeline["when_utc"] = pd.to_datetime(timeline["when_utc"], errors="coerce")
    timeline = timeline.dropna(subset=["when_utc"]).sort_values("when_utc")
    plt.figure(figsize=(10, 5))
    clean_events = timeline[timeline["conflict_prob"] < 0.5]
    tampered_events = timeline[timeline["conflict_prob"] >= 0.5]
    plt.scatter(clean_events["when_utc"], clean_events["confidence"], color="steelblue", alpha=0.5, label="Aligned Evidence", s=10)
    plt.scatter(tampered_events["when_utc"], tampered_events["confidence"], color="crimson", alpha=0.8, label="Conflicting Evidence", s=20)
    plt.title("Chronological Evidence Review")
    plt.xlabel("Timeline (UTC)")
    plt.ylabel("Reconstruction Confidence")
    plt.legend()
    plt.xticks(rotation=45)
    plt.savefig(d / "chronological_evidence_review.png", dpi=150, bbox_inches="tight")
    plt.close()

    conflict_data = timeline.dropna(subset=["dropped_source"]).copy()
    if not conflict_data.empty:
        plt.figure(figsize=(10, 6))
        sns.boxplot(data=conflict_data, x="dropped_source", y="spread_s", hue="dropped_source", palette="Set2", legend=False)
        plt.yscale("symlog")
        plt.title("Temporal Gap Magnitude by Manipulated Artefact")
        plt.xlabel("Dropped Forensic Source")
        plt.ylabel("Time Discrepancy Spread (seconds, log scale)")
        plt.xticks(rotation=45)
        plt.savefig(d / "gap_magnitude_by_artefact.png", dpi=150, bbox_inches="tight")
        plt.close()

        if "case" in conflict_data.columns:
            plt.figure(figsize=(12, 6))
            case_conflicts = conflict_data.groupby(["case", "dropped_source"]).size().unstack(fill_value=0)
            case_conflicts.plot(kind="bar", stacked=True, colormap="tab10", figsize=(12, 6))
            plt.title("Artefact Manipulation Profiles Across ADFIR Cases")
            plt.xlabel("Forensic Case / Investigation")
            plt.ylabel("Number of Dropped Sources")
            plt.legend(title="Manipulated Source")
            plt.xticks(rotation=45)
            plt.savefig(d / "case_manipulation_profiles.png", dpi=150, bbox_inches="tight")
            plt.close()


    # To be considered if I get more time (Time Permitting)
    # 1. An interactive HTML timelines using like Plotly library to zoom into specific timeframes.
    # 2. Adding graph network visualisations linking manipulated artefacts to specific user accounts or process IDs.
    # 3. Another valuable addition would be automated anomaly highlights that compare normal after-hours activity against the timeline discrepancies.
    # 4. Finally, building an export function to generate CSVs formatted specifically for ingest into commercial tools like Autopsy or Magnet AXIOM would significantly improve practical utilisation.

if __name__ == "__main__":
    main(*sys.argv[1:])