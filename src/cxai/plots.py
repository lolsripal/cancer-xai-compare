"""Figures for the results section."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402

from .models import PRETTY  # noqa: E402

sns.set_theme(style="whitegrid", context="notebook")
PALETTE = {"logreg": "#4C72B0", "random_forest": "#55A868", "xgboost": "#C44E52", "mlp": "#8172B3",
           "small_logreg": "#4C72B0", "tree_depth3": "#DD8452"}


def _save(fig, path: Path):
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def performance(metrics: pd.DataFrame, path: Path):
    m = metrics[["roc_auc", "pr_auc_normal", "balanced_accuracy", "mcc"]].copy()
    m.index = [PRETTY.get(i, i) for i in m.index]
    long = m.reset_index(names="model").melt(id_vars="model", var_name="metric", value_name="score")
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.barplot(long, x="metric", y="score", hue="model", ax=ax)
    ax.set_ylim(max(0, long.score.min() - 0.05), 1.0)
    ax.set_title("Held-out test performance (tumor vs. normal)")
    ax.legend(loc="lower left", fontsize=8)
    _save(fig, path)


def agreement_heatmap(mat: pd.DataFrame, title: str, path: Path):
    fig, ax = plt.subplots(figsize=(9, 7.5))
    sns.heatmap(mat, annot=True, fmt=".2f", cmap="viridis", vmin=0, vmax=1, square=True,
                cbar_kws={"shrink": 0.7}, annot_kws={"size": 7}, ax=ax)
    ax.set_title(title)
    _save(fig, path)


def consensus_heatmap(rank_df: pd.DataFrame, genes: list[str], path: Path):
    pct = 1 - (rank_df.loc[genes] - 1) / len(rank_df)
    fig, ax = plt.subplots(figsize=(10, 0.28 * len(genes) + 2))
    sns.heatmap(pct, cmap="rocket_r", vmin=0.9, vmax=1, cbar_kws={"label": "rank percentile"}, ax=ax)
    ax.set_title("Top consensus genes: rank percentile under each model:explainer")
    _save(fig, path)


def enrichment_bar(df: pd.DataFrame, path: Path, n: int = 12):
    d = df.head(n).copy()
    d["-log10 FDR"] = -np.log10(d["fdr"].clip(lower=1e-300))
    d["pathway"] = d["pathway"].str.replace("HALLMARK_", "")
    fig, ax = plt.subplots(figsize=(8, 0.4 * len(d) + 1.5))
    sns.barplot(d, y="pathway", x="-log10 FDR", color="#C44E52", ax=ax)
    ax.axvline(-np.log10(0.05), ls="--", c="k", lw=1, label="FDR = 0.05")
    for i, (o, s) in enumerate(zip(d["overlap"], d["list_size"])):
        ax.text(0.05, i, f"{o}/{s}", va="center", color="white", fontsize=8)
    ax.set_title("Hallmark pathway enrichment of consensus top genes")
    ax.legend()
    _save(fig, path)


def ablation_curves(summary: pd.DataFrame, metric: str, path: Path):
    models = summary["model"].unique()
    fig, axes = plt.subplots(1, len(models), figsize=(4 * len(models), 3.6), sharey=True)
    styles = {"own_top": ("-o", "own top genes"), "consensus_top": ("-s", "consensus top genes"),
              "random": ("--^", "random genes (mean ± sd)")}
    for ax, name in zip(np.atleast_1d(axes), models):
        d = summary[summary.model == name]
        base = d[d.removed == "none"][metric].iloc[0]
        for removed, (st, label) in styles.items():
            s = d[d.removed == removed].sort_values("k")
            ax.plot(s.k, s[metric], st, label=label, ms=4)
            sd = f"{metric}_sd"
            if removed == "random" and sd in s:
                ax.fill_between(s.k, s[metric] - s[sd].fillna(0), s[metric] + s[sd].fillna(0), alpha=0.2)
        ax.axhline(base, c="grey", lw=1, ls=":", label="no genes removed")
        ax.set_xscale("log")
        ax.set_title(PRETTY.get(name, name))
        ax.set_xlabel("# genes removed (log scale)")
    np.atleast_1d(axes)[0].set_ylabel(metric)
    np.atleast_1d(axes)[0].legend(fontsize=7)
    fig.suptitle(f"Ablation: retrain without the top-k genes ({metric}, test set)")
    _save(fig, path)


def sparse_curve(sparse_df: pd.DataFrame, full: pd.DataFrame, path: Path):
    fig, ax = plt.subplots(figsize=(7, 4))
    for name, d in sparse_df.groupby("model"):
        ax.plot(d.k, d.roc_auc, "-o", label=f"{name} (top-k consensus genes)",
                color=PALETTE.get(name))
    for name, row in full.iterrows():
        ax.axhline(row.roc_auc, ls=":", lw=1, color=PALETTE.get(name),
                   label=f"{PRETTY.get(name, name)}, all genes")
    ax.set_xscale("log")
    ax.set_xticks(sorted(sparse_df.k.unique()))
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_xlabel("# genes in the small model")
    ax.set_ylabel("test ROC-AUC")
    ax.set_title("Small interpretable models vs. full black-box models")
    ax.legend(fontsize=7, loc="lower right")
    _save(fig, path)


def pdp_grid(pdp: pd.DataFrame, path: Path):
    genes = list(dict.fromkeys(pdp.gene))
    cols = 3
    rows = int(np.ceil(len(genes) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3 * rows), sharey=True)
    for ax, g in zip(axes.ravel(), genes):
        for name, d in pdp[pdp.gene == g].groupby("model"):
            ax.plot(d.z_expression, d.p_tumor, label=PRETTY.get(name, name), color=PALETTE.get(name))
        ax.set_title(g)
        ax.set_xlabel("expression (z-score)")
    for ax in axes.ravel()[len(genes):]:
        ax.axis("off")
    axes.ravel()[0].set_ylabel("P(tumor), partial dependence")
    axes.ravel()[0].legend(fontsize=7)
    fig.suptitle("Partial dependence of each model on the top consensus genes")
    _save(fig, path)


def shap_beeswarm(X: pd.DataFrame, values: np.ndarray, title: str, path: Path, n: int = 20):
    import shap

    shap.summary_plot(values, X, max_display=n, show=False, plot_size=(8, 0.35 * n + 1.5))
    fig = plt.gcf()
    fig.suptitle(title)
    _save(fig, path)
