"""Plots from files written by cli.run (never trains)."""
import json, sys
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def main(out_dir="../data/outputs"):
    d = Path(out_dir)
    ax = pd.read_csv(d / "loss_history.csv").plot(x="iteration", y="loss", legend=False, title="Logistic regression training loss")
    ax.set_ylabel("loss")
    plt.savefig(d / "loss_curve.png", dpi=150, bbox_inches="tight")
    plt.close()
    
    m = json.load(open(d / "metrics.json"))
    names = ["logistic_regression_scratch", "random_forest"]
    df = pd.DataFrame({n: {k: m[n][k] for k in ["precision", "recall", "f1", "roc_auc", "pr_auc"]} for n in names})
    df.plot.bar(title=f"Conflict detection on held-out {m['holdout']}", rot=0)
    plt.ylim(0, 1.05)
    plt.savefig(d / "model_comparison.png", dpi=150, bbox_inches="tight")

if __name__ == "__main__":
    main(*sys.argv[1:])