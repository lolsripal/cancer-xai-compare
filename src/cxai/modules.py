"""Gene-module explanations: explain groups of co-expressed genes, not single genes.

Round 1 found that equally accurate models pick different top genes, and that
removing top genes barely hurts them, which suggests many interchangeable,
correlated genes. Here we cluster genes into co-expression modules (on the
training split only) and ask whether models agree on *modules* even when they
disagree on genes.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from sklearn.metrics import log_loss

from . import biology
from .models import evaluate, refit


def cluster_genes(X_train: pd.DataFrame, min_abs_corr: float = 0.6) -> pd.Series:
    """Average-linkage clustering on 1 - |Pearson r|; returns gene -> module id (1..M)."""
    r = np.corrcoef(X_train.values.T)
    d = np.clip(1 - np.abs(r), 0, None)
    np.fill_diagonal(d, 0)
    Z = linkage(squareform(d, checks=False), method="average")
    labels = fcluster(Z, t=1 - min_abs_corr, criterion="distance")
    # Renumber so module 1 is the largest.
    order = pd.Series(labels).value_counts().index
    remap = {old: new for new, old in enumerate(order, start=1)}
    return pd.Series([remap[l] for l in labels], index=X_train.columns, name="module")


def describe_modules(modules: pd.Series, X_train: pd.DataFrame, gene_sets: dict,
                     background: list[str]) -> pd.DataFrame:
    """Size, hub gene (highest mean |r| to the rest) and best Hallmark annotation."""
    rows = []
    for m, genes in modules.groupby(modules).groups.items():
        genes = list(genes)
        if len(genes) > 1:
            r = np.abs(np.corrcoef(X_train[genes].values.T))
            hub = genes[int((r.sum(axis=0) - 1).argmax())]
        else:
            hub = genes[0]
        best, best_p, best_ov = "", 1.0, 0
        if len(genes) >= 3:
            enr = biology.enrichment(genes, background, gene_sets)
            if not enr.empty and enr.loc[0, "p_value"] < 0.01 and enr.loc[0, "overlap"] >= 2:
                best = enr.loc[0, "pathway"].replace("HALLMARK_", "")
                best_p, best_ov = float(enr.loc[0, "p_value"]), int(enr.loc[0, "overlap"])
        rows.append({"module": int(m), "label": f"M{m}:{hub}", "size": len(genes), "hub": hub,
                     "hallmark": best, "hallmark_p": best_p, "hallmark_overlap": best_ov,
                     "genes": ",".join(sorted(genes))})
    return pd.DataFrame(rows).set_index("module").sort_index()


def module_shap(values: np.ndarray, genes: list[str], modules: pd.Series) -> pd.Series:
    """|sum of SHAP within a module| per sample & class, then summed over classes and averaged.

    SHAP values are additive, so the within-module sum is the module's joint
    contribution. Opposite-signed genes in a module can cancel, which is the point.
    """
    mod = modules.reindex(genes).values
    ids = np.unique(mod)
    out = {}
    for m in ids:
        cols = np.where(mod == m)[0]
        out[m] = np.abs(values[:, cols, :].sum(axis=1)).sum(axis=1).mean()
    return pd.Series(out)


def _grouped_perm_one(model, X: np.ndarray, y, cols, base, repeats, seed, labels):
    rng = np.random.default_rng(seed)
    losses = []
    for _ in range(repeats):
        Xp = X.copy()
        perm = rng.permutation(len(X))
        Xp[:, cols] = X[perm][:, cols]  # shuffle the whole module jointly
        losses.append(log_loss(y, model.predict_proba(Xp), labels=labels))
    return float(np.mean(losses) - base)


def module_permutation(model, X: pd.DataFrame, y: pd.Series, modules: pd.Series,
                       repeats: int = 3, seed: int = 0, n_jobs: int = -1) -> pd.Series:
    labels = list(range(len(model.classes_)))
    base = log_loss(y, model.predict_proba(X), labels=labels)
    mod = modules.reindex(X.columns).values
    ids = np.unique(mod)
    Xv = X.values
    # Wrap predict_proba so sklearn doesn't warn about missing feature names.
    cols_names = list(X.columns)

    class _Wrap:
        classes_ = model.classes_

        def predict_proba(self, A):
            return model.predict_proba(pd.DataFrame(A, columns=cols_names))

    w = _Wrap()
    scores = Parallel(n_jobs=n_jobs, prefer="threads")(
        delayed(_grouped_perm_one)(w, Xv, y.values, np.where(mod == m)[0], base, repeats, seed + int(m), labels)
        for m in ids
    )
    return pd.Series(scores, index=ids)


def explain_modules(models: dict, shap_mats: dict, data, modules: pd.Series, repeats: int,
                    seed: int, log=print) -> pd.DataFrame:
    cols = {}
    for name, model in models.items():
        _, sv = shap_mats[name]
        cols[f"{name}:shap"] = module_shap(sv, data.genes, modules)
        log(f"  [{name}] grouped permutation over {modules.nunique()} modules")
        cols[f"{name}:permutation"] = module_permutation(model, data.X_val, data.y_val, modules,
                                                         repeats, seed)
    df = pd.DataFrame(cols)
    df.index.name = "module"
    return df


def ablate_modules(models: dict, ranking: list[int], modules: pd.Series, data,
                   m_values: list[int], n_random: int, seed: int, log=print) -> pd.DataFrame:
    """Remove the top-m consensus modules (all their genes) vs. the same number of random genes."""
    rng = np.random.default_rng(seed)
    genes = data.genes
    rows = []
    for name, model in models.items():
        for m in m_values:
            drop = list(modules.index[modules.isin(ranking[:m])])
            if len(drop) >= len(genes) - 5:
                continue
            strategies = {"top_modules": drop}
            for r in range(n_random):
                strategies[f"random_{r}"] = list(rng.choice(genes, size=len(drop), replace=False))
            for label, d in strategies.items():
                keep = [g for g in genes if g not in set(d)]
                res = evaluate(refit(model, data.X_train[keep], data.y_train), data.X_test[keep], data.y_test)
                rows.append({"model": name, "removed": "random" if label.startswith("random") else label,
                             "modules": m, "genes_removed": len(d), **res})
        log(f"  [{name}] module ablation done")
    return pd.DataFrame(rows)
