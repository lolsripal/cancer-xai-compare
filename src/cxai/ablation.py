"""Does removing a model's most important genes break it? (vs. removing random genes)"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .models import evaluate, refit


def ablate(
    models: dict,
    rankings: dict[str, list[str]],
    data,
    k_values: list[int],
    n_random: int,
    seed: int,
    log=print,
) -> pd.DataFrame:
    """Retrain each model without its top-k genes, and without k random genes.

    ``rankings[name]`` is that model's own gene ranking (from its SHAP values);
    ``rankings["consensus"]`` is the cross-method consensus ranking and is
    applied to every model.
    """
    rng = np.random.default_rng(seed)
    genes = data.genes
    rows = []
    for name, model in models.items():
        base = evaluate(model, data.X_test, data.y_test)
        rows.append({"model": name, "removed": "none", "k": 0, "rep": 0, **base})
        for k in k_values:
            if k >= len(genes):
                continue
            strategies = {"own_top": rankings[name][:k], "consensus_top": rankings["consensus"][:k]}
            for r in range(n_random):
                strategies[f"random_{r}"] = list(rng.choice(genes, size=k, replace=False))
            for label, drop in strategies.items():
                keep = [g for g in genes if g not in set(drop)]
                m = refit(model, data.X_train[keep], data.y_train)
                res = evaluate(m, data.X_test[keep], data.y_test)
                removed = "random" if label.startswith("random") else label
                rep = int(label.split("_")[1]) if removed == "random" else 0
                rows.append({"model": name, "removed": removed, "k": k, "rep": rep, **res})
        log(f"  [{name}] ablation done")
    return pd.DataFrame(rows)


def summarise(df: pd.DataFrame, metric: str = "roc_auc") -> pd.DataFrame:
    g = df.groupby(["model", "removed", "k"])[metric]
    return g.agg(["mean", "std"]).reset_index().rename(columns={"mean": metric, "std": f"{metric}_sd"})


def ablate_gene_set(models: dict, drop: list[str], data, label: str, n_random: int, seed: int) -> pd.DataFrame:
    """Retrain without a named gene set (e.g. all PAM50 genes) vs. the same number of random genes."""
    rng = np.random.default_rng(seed)
    drop = [g for g in drop if g in set(data.genes)]
    rows = []
    for name, model in models.items():
        strategies = {label: drop}
        for r in range(n_random):
            strategies[f"random_{r}"] = list(rng.choice(data.genes, size=len(drop), replace=False))
        for lab, d in strategies.items():
            keep = [g for g in data.genes if g not in set(d)]
            res = evaluate(refit(model, data.X_train[keep], data.y_train), data.X_test[keep], data.y_test)
            rows.append({"model": name, "removed": "random" if lab.startswith("random") else lab,
                         "k": len(d), **res})
    return pd.DataFrame(rows)
