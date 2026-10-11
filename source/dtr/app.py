"""Streamlit presentation layer.   Run from the repo's source/ folder:  streamlit run dtr/app.py

It only READS the files written by `python -m dtr.main run` (metrics.json, predictions.csv, real_timeline.csv,
group_evidence.csv, ...) and `python -m dtr.main drift` (drift_experiment.csv). It does no modelling itself.
"""
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dtr.validate import TEMPLATE_COLUMNS, match_documented, summarise  # noqa: E402

DEFAULT_OUT = Path(__file__).resolve().parents[2] / "data" / "outputs"
MODEL_LABEL = {"logistic_regression_scratch": "Logistic Regression (from scratch)", "random_forest": "Random Forest"}
FAM_COLOURS = {"eventlog": "#1f77b4", "execution": "#2ca02c", "filesystem": "#7f7f7f", "registry": "#17becf",
               "filefolder": "#8c564b", "deletion": "#d62728"}

st.set_page_config(page_title="Diverging Truths", layout="wide")


@st.cache_data(show_spinner=False)
def load(out_dir: str):
    p = Path(out_dir)
    def csv(n):
        f = p / n
        return pd.read_csv(f) if f.exists() else None
    m = p / "metrics.json"
    return {"metrics": json.loads(m.read_text()) if m.exists() else None,
            "pred": csv("predictions.csv"), "real": csv("real_timeline.csv"), "evidence": csv("group_evidence.csv"),
            "drift": csv("drift_experiment.csv"), "synth_tl": csv("reconstructed_timeline.csv")}


st.sidebar.title("Diverging Truths")
st.sidebar.caption("Who did what, when: reconciling conflicting forensic timelines")
out_dir = st.sidebar.text_input("Results folder", str(DEFAULT_OUT))
page = st.sidebar.radio("View", ["Overview", "Evidence explorer", "Clock drift", "Write-up validation", "Method and limits"])
D = load(out_dir)
if D["metrics"] is None:
    st.error(f"No metrics.json in {out_dir}. Run `python -m dtr.main run` first, or fix the folder path.")
    st.stop()
M = D["metrics"]


def headline(model):
    s = M.get("macro_over_seeds", {}).get(model, {})
    return {k: (v["mean"], v["sd"]) for k, v in s.items()}


# ---------------------------------------------------------------- Overview
if page == "Overview":
    st.title("Does the evidence agree with itself?")
    st.write(f"Leave-one-case-out over **{len(M['cases'])} real cases**, {M['n_seeds']} seed(s), "
             f"features: *{M['features']}*, threshold: *{M['threshold_mode']}*.")
    st.warning("Labels are synthetic: tampering is injected as a clock shift into real evidence groups. "
               "The scores measure detection of injected shifts, not conflicts found in the wild.")
    cols = st.columns(2)
    for col, model in zip(cols, MODEL_LABEL):
        h = headline(model)
        with col:
            st.subheader(MODEL_LABEL[model])
            a, b, c = st.columns(3)
            a.metric("F1", f"{h['f1'][0]:.3f}", f"± {h['f1'][1]:.3f}", delta_color="off")
            b.metric("ROC-AUC", f"{h['roc_auc'][0]:.3f}", f"± {h['roc_auc'][1]:.3f}", delta_color="off")
            c.metric("Recall", f"{h['recall'][0]:.3f}", f"± {h['recall'][1]:.3f}", delta_color="off")
    rows = []
    for case, pc in M["per_case"].items():
        for model in MODEL_LABEL:
            rows.append({"case": case, "model": MODEL_LABEL[model], "F1": pc[model]["f1"],
                         "ROC-AUC": pc[model].get("roc_auc")})
    df = pd.DataFrame(rows)
    metric = st.radio("Per held-out case", ["F1", "ROC-AUC"], horizontal=True)
    fig = px.bar(df, x="case", y=metric, color="model", barmode="group", range_y=[0, 1.05],
                 color_discrete_sequence=["#1f77b4", "#ff7f0e"])
    if metric == "ROC-AUC":
        fig.add_hline(y=0.5, line_dash="dash", line_color="grey", annotation_text="chance")
    fig.update_layout(xaxis_title=None, legend_title=None, height=380)
    st.plotly_chart(fig, width="stretch")
    st.caption("Per-case values are from the first seed; the headline cards are the mean ± sd over seeds "
               "(sd reflects tamper randomness only, not variation across cases).")

    st.subheader("How much less does an investigator have to read?")
    eff = pd.DataFrame([{"case": c, "raw events": pc["effort_reduction"]["raw_events"],
                         "evidence groups": pc["effort_reduction"]["evidence_groups"],
                         **({"groups to review for 90% of tampered": pc["review_effort_rf"]["ranked_reviews"],
                             "unranked expectation": round(pc["review_effort_rf"]["raw_expected_reviews"], 1),
                             "reduction vs unranked": f"{pc['review_effort_rf']['saving']:.0%}"}
                            if "review_effort_rf" in pc and pc["review_effort_rf"]["ranked_reviews"] else {})}
                        for c, pc in M["per_case"].items()])
    st.dataframe(eff, hide_index=True, width="stretch")
    st.caption("Counts of items to read, not minutes saved. Ranked = review groups in descending conflict "
               "probability; unranked = expected reviews if read in arbitrary order.")

    if D["pred"] is not None:
        st.subheader("Conflict probability")
        model = st.selectbox("Model", list(MODEL_LABEL), format_func=MODEL_LABEL.get)
        col = "p_lr" if model.startswith("logistic") else "p_rf"
        d = D["pred"].assign(label=lambda x: x["truth"].map({0: "Authentic", 1: "Tampered"}))
        st.plotly_chart(px.histogram(d, x=col, color="label", nbins=25, barmode="overlay", opacity=0.65,
                                     color_discrete_map={"Authentic": "#4682b4", "Tampered": "#ff9f1c"})
                        .update_layout(height=340, xaxis_title="Predicted probability of tampering"),
                        width="stretch")

# ---------------------------------------------------------------- Evidence explorer
elif page == "Evidence explorer":
    st.title("Evidence explorer")
    if D["real"] is None or D["evidence"] is None:
        st.info("real_timeline.csv / group_evidence.csv not found. Re-run `python -m dtr.main run`.")
        st.stop()
    tl, ev = D["real"], D["evidence"]
    c1, c2, c3 = st.columns([2, 2, 3])
    case = c1.selectbox("Case", sorted(tl["case"].unique()))
    only = c2.checkbox("Only groups the model flags as conflicting", value=True)
    q = c3.text_input("Search program / file name", "")
    t = tl[tl["case"] == case].copy()
    if only:
        t = t[t["conflict_prob"] >= 0.5]
    if q:
        t = t[t["what"].str.contains(q, case=False)]
    t = t.sort_values("conflict_prob", ascending=False)
    st.caption(f"{len(t):,} groups shown. This is the model applied to the real, untampered evidence.")
    st.dataframe(t[["when_utc", "who", "what", "sources", "spread_s", "conflict_prob", "confidence", "dropped_source"]]
                 .head(500), hide_index=True, width="stretch", height=260)
    if len(t):
        pick = st.selectbox("Inspect a group", t["what"].str.replace("executed/touched ", "", regex=False).head(500))
        g = ev[(ev["case"] == case) & (ev["key"] == pick)].copy()
        row = t[t["what"].str.endswith(pick)].iloc[0]
        g["time"] = pd.to_datetime(g["ts"], unit="s", utc=True)
        fig = go.Figure()
        for fam, x in g.groupby("family"):
            fig.add_trace(go.Scatter(x=x["time"], y=[fam] * len(x), mode="markers", name=fam,
                                     marker=dict(size=11, color=FAM_COLOURS.get(fam, "#444"),
                                                 line=dict(width=1, color="white"),
                                                 symbol="x" if fam == row["dropped_source"] else "circle")))
        fig.add_vline(x=pd.to_datetime(row["when_utc"], utc=True).timestamp() * 1000, line_dash="dash",
                      line_color="crimson", annotation_text="reconciled time")
        fig.update_layout(height=320, showlegend=False, xaxis_title="Time (UTC)", yaxis_title=None,
                          title=f"{pick}: one row per source, dashed line = reconciled time")
        st.plotly_chart(fig, width="stretch")
        a, b, c = st.columns(3)
        a.metric("Conflict probability", f"{row['conflict_prob']:.2f}")
        b.metric("Confidence", f"{row['confidence']:.2f}")
        c.metric("Set aside", row["dropped_source"] or "none")
        st.caption("A cross marks the source the tool set aside as the most divergent. 'Who' is the most common "
                   "user attached to the group's records, or unknown when the artifacts carry no user.")

# ---------------------------------------------------------------- Clock drift
elif page == "Clock drift":
    st.title("Anchor-event clock-drift correction")
    st.write("Anchors are programs seen by two independent sources. The median signed gap between them estimates "
             "a source's clock error. To test it, a known drift is injected into one source and the tool must "
             "recover it from the anchors alone.")
    if D["drift"] is None:
        st.info("drift_experiment.csv not found. Run `python -m dtr.main drift` on the VM.")
    else:
        df = D["drift"]
        fam = st.multiselect("Source", sorted(df["family"].unique()), default=sorted(df["family"].unique()))
        d = df[df["family"].isin(fam)]
        lim = float(np.abs(d["injected_s"]).max())
        fig = px.scatter(d, x="injected_s", y="detected_s", color="case", symbol="family",
                         labels={"injected_s": "Injected drift (s)", "detected_s": "Recovered from anchors (s)"})
        fig.add_trace(go.Scatter(x=[-lim, lim], y=[-lim, lim], mode="lines", name="perfect",
                                 line=dict(dash="dash", color="grey")))
        st.plotly_chart(fig.update_layout(height=400), width="stretch")
        s = d.groupby(["case", "family"]).agg(median_abs_error_s=("abs_error_s", "median"),
                                              worst_abs_error_s=("abs_error_s", "max"),
                                              median_residual_s=("residual_after_correction_s", lambda x: x.abs().median()),
                                              anchors=("n_anchors", "min"),
                                              natural_offset_s=("baseline_offset_s", "first")).round(2)
        st.dataframe(s, width="stretch")
        st.caption("Natural offset = the median gap before any injection. Large natural offsets (hours) mean the "
                   "two sources describe different moments of the same program, which is why correction is "
                   "applied only to a *change* in the estimate, not to the raw median.")
    st.subheader("Natural offsets between sources (no injection)")
    rows = [{"case": c, "pair": p, "n": v["n"], "median_offset_s": round(v["median_offset_s"], 2)}
            for c, d in M.get("family_offsets", {}).items() for p, v in d.items()]
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")

# ---------------------------------------------------------------- Write-up validation
elif page == "Write-up validation":
    st.title("Comparison with documented events")
    st.write("Upload a CSV of events taken from a published case write-up. Each is matched to the reconstructed "
             "real timeline and the time error is measured. The documented times must come from the write-up, "
             "not from this tool.")
    st.download_button("Download CSV template", pd.DataFrame(
        [["Stolen_Szechuan_Sauce_DC", "example description", "program.exe", "2020-09-19 03:21:00"]],
        columns=TEMPLATE_COLUMNS).to_csv(index=False), "documented_events_template.csv")
    up = st.file_uploader("Documented events (CSV)", type="csv")
    if up is not None and D["real"] is not None:
        try:
            m = match_documented(D["real"], pd.read_csv(up))
        except ValueError as e:
            st.error(str(e)); st.stop()
        s = summarise(m)
        a, b, c, d_ = st.columns(4)
        a.metric("Found in timeline", f"{s['found']} / {s['documented']}")
        if s["found"]:
            b.metric("Within 1 min", f"{s['within_1min']:.0%}")
            c.metric("Within 5 min", f"{s['within_5min']:.0%}")
            d_.metric("Median |error|", f"{s['median_abs_error_s']:.0f} s")
        st.dataframe(m, hide_index=True, width="stretch")
        st.download_button("Download matches", m.to_csv(index=False), "write_up_matches.csv")
    elif up is not None:
        st.info("real_timeline.csv not found. Re-run `python -m dtr.main run`.")

# ---------------------------------------------------------------- Method and limits
else:
    st.title("Method and limits")
    st.markdown("""
**Pipeline.** EZ-Tools CSV exports are read into six source families (registry, execution, event log, file system,
file/folder access, deletion). Events are grouped per program and case. Each group becomes a feature vector of
timestamp gaps between families. A classifier outputs a conflict probability; the most divergent source is set
aside and the *when* is re-estimated from the rest.

**What the numbers mean.**
- Training and test labels are synthetic: one source's timestamps are shifted by 1 minute to 30 days.
- Evaluation is leave-one-case-out, so every score is on a case the model has not seen.
- Only three families (event log, file system, execution) hold substantial volume; the others have tens of events.
- Execution and file-system times are not fully independent evidence, since Windows derives some execution
  records from file-system times.
- A single gap feature (`gap_max`) is a strong baseline, so the models should be read against it.
""")
    st.json({k: M[k] for k in ("protocol", "features", "threshold_mode", "n_seeds")})
