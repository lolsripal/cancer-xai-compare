"""Stability study: repeat the experiment on 10 different patient-grouped splits.

    python -m cxai.stability            # both tasks, seeds 0-9 (~15-20 min)
    python -m cxai.stability --demo

For each split we retrain all models and recompute SHAP + native importance
(permutation importance and ablation are skipped for time). This answers:

* How much do the test metrics vary between splits? (mean ± sd)
* Is a given model's explanation stable when the training data changes?
* Is disagreement *between* models larger than a model's own split-to-split noise?
* Which genes are in the consensus top-50 in (almost) every split? -> stable core
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from . import compare, explain
from .data import TASKS
from .models import PRETTY, evaluate, fit_all
from .pipeline import ROOT, get_data, load_config, log


def jaccard(a, b) -> float:
    a, b = set(a), set(b)
    return len(a & b) / len(a | b)


def run(tasks=TASKS, demo: bool = False, config_path: Path = ROOT / "config.yaml",
        out: Path | None = None) -> dict:
    cfg = load_config(config_path, demo)
    st = cfg["stability"]
    k = cfg["explain"]["top_k"]
    exp_cfg = dict(cfg["explain"], shap_samples=st["shap_samples"], mlp_shap_samples=st["mlp_shap_samples"])
    out = Path(out or (ROOT / "results" / ("demo" if demo else "") / "stability"))
    (out / "figures").mkdir(parents=True, exist_ok=True)
    summary = {}
    all_metrics, all_within, all_cross, all_freq = [], [], [], []

    for task in tasks:
        metrics, tops, cons_tops = [], {}, {}
        for seed in st["seeds"]:
            log(f"[{task}] split seed={seed}")
            data, _ = get_data(task, cfg, seed, demo)
            models = fit_all(data.X_train, data.y_train, seed, fast=demo)
            for n, m in models.items():
                metrics.append({"task": task, "seed": seed, "model": n, **evaluate(m, data.X_test, data.y_test)})
            imp, _ = explain.explain_all(models, data, exp_cfg, seed, log=lambda *_: None, permutation=False)
            for c in imp.columns:
                tops[(c, seed)] = list(imp[c].sort_values(ascending=False).index[:k])
            cons_tops[seed] = list(compare.consensus(imp).index[:k])
            # cross-model agreement within this split, per explainer
            for c1, c2 in itertools.combinations(imp.columns, 2):
                (m1, e1), (m2, e2) = c1.split(":"), c2.split(":")
                if e1 == e2:
                    all_cross.append({"task": task, "seed": seed, "a": c1, "b": c2,
                                      "jaccard": jaccard(tops[(c1, seed)], tops[(c2, seed)])})
        mdf = pd.DataFrame(metrics)
        all_metrics.append(mdf)

        cols = sorted({c for c, _ in tops})
        for c in cols:
            seeds = [s for s in st["seeds"] if (c, s) in tops]
            vals = [jaccard(tops[(c, s1)], tops[(c, s2)]) for s1, s2 in itertools.combinations(seeds, 2)]
            all_within.append({"task": task, "ranking": c, "mean_jaccard_across_splits": np.mean(vals),
                               "sd": np.std(vals)})
        cons_vals = [jaccard(cons_tops[a], cons_tops[b]) for a, b in itertools.combinations(cons_tops, 2)]
        all_within.append({"task": task, "ranking": "consensus", "mean_jaccard_across_splits": np.mean(cons_vals),
                           "sd": np.std(cons_vals)})

        freq = pd.Series([g for t in cons_tops.values() for g in t]).value_counts() / len(cons_tops)
        fdf = freq.rename("fraction_of_splits").reset_index().rename(columns={"index": "gene"})
        fdf.insert(0, "task", task)
        all_freq.append(fdf)

        agg = mdf.drop(columns=["task", "seed"]).groupby("model").agg(["mean", "std"])
        cross = pd.DataFrame([r for r in all_cross if r["task"] == task])
        within = pd.DataFrame([r for r in all_within if r["task"] == task])
        summary[task] = {
            "n_splits": len(st["seeds"]),
            "roc_auc_mean_sd": {m: [round(agg.loc[m, ("roc_auc", "mean")], 4),
                                    round(agg.loc[m, ("roc_auc", "std")], 4)] for m in agg.index},
            "balanced_accuracy_mean_sd": {m: [round(agg.loc[m, ("balanced_accuracy", "mean")], 4),
                                              round(agg.loc[m, ("balanced_accuracy", "std")], 4)] for m in agg.index},
            "within_method_stability_mean": round(float(within[within.ranking != "consensus"]
                                                        .mean_jaccard_across_splits.mean()), 3),
            "consensus_stability": round(float(np.mean(cons_vals)), 3),
            "cross_model_agreement_mean": round(float(cross.jaccard.mean()), 3),
            "cross_model_agreement_sd_over_splits": round(float(cross.groupby("seed").jaccard.mean().std()), 3),
            "stable_core_genes_ge_80pct": list(freq[freq >= 0.8].index),
        }

    pd.concat(all_metrics).to_csv(out / "metrics_by_split.csv", index=False, float_format="%.4f")
    within_df = pd.DataFrame(all_within)
    within_df.to_csv(out / "ranking_stability.csv", index=False, float_format="%.3f")
    cross_df = pd.DataFrame(all_cross)
    cross_df.to_csv(out / "cross_model_agreement_by_split.csv", index=False, float_format="%.3f")
    freq_df = pd.concat(all_freq)
    freq_df.to_csv(out / "gene_selection_frequency.csv", index=False, float_format="%.2f")
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    _plots(pd.concat(all_metrics), within_df, cross_df, freq_df, out / "figures")
    log(f"Done. Results in {out}")
    return summary


def _plots(metrics, within, cross, freq, fig_dir: Path):
    sns.set_theme(style="whitegrid")
    tasks = list(dict.fromkeys(metrics.task))

    fig, axes = plt.subplots(1, len(tasks), figsize=(6 * len(tasks), 3.8), squeeze=False)
    for ax, t in zip(axes.ravel(), tasks):
        d = metrics[metrics.task == t].melt(id_vars=["task", "seed", "model"],
                                            value_vars=["roc_auc", "balanced_accuracy"])
        d["model"] = d.model.map(PRETTY)
        sns.boxplot(d, x="variable", y="value", hue="model", ax=ax, fliersize=2)
        ax.set_title(f"{t}: test metrics over {metrics.seed.nunique()} splits")
        ax.set_xlabel("")
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(fig_dir / "metrics_over_splits.png", dpi=150)
    plt.close(fig)

    rows = []
    for t in tasks:
        w = within[(within.task == t) & (within.ranking != "consensus")]
        for _, r in w.iterrows():
            rows.append({"task": t, "kind": "same method, different split", "jaccard": r.mean_jaccard_across_splits})
        for _, r in cross[cross.task == t].groupby(["a", "b"]).jaccard.mean().reset_index().iterrows():
            rows.append({"task": t, "kind": "different model, same split", "jaccard": r.jaccard})
    d = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.boxplot(d, x="task", y="jaccard", hue="kind", ax=ax)
    sns.stripplot(d, x="task", y="jaccard", hue="kind", dodge=True, ax=ax, color="k", size=3, legend=False)
    ax.set_ylabel("Jaccard overlap of top-50 genes")
    ax.set_title("Split-to-split noise vs. model-to-model disagreement")
    fig.tight_layout()
    fig.savefig(fig_dir / "stability_vs_disagreement.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, len(tasks), figsize=(6 * len(tasks), 6), squeeze=False)
    for ax, t in zip(axes.ravel(), tasks):
        f = freq[freq.task == t].head(25)
        sns.barplot(f, y="gene", x="fraction_of_splits", ax=ax, color="#4C72B0")
        ax.axvline(0.8, ls="--", c="k", lw=1)
        ax.set_title(f"{t}: share of splits with gene in consensus top-50")
    fig.tight_layout()
    fig.savefig(fig_dir / "gene_selection_frequency.png", dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", default="all", choices=TASKS + ["all"])
    ap.add_argument("--demo", action="store_true")
    args = ap.parse_args()
    run(TASKS if args.task == "all" else [args.task], demo=args.demo)


if __name__ == "__main__":
    main()
