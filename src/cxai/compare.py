"""Agreement between gene rankings produced by different (model, explainer) pairs."""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def ranks(importance: pd.DataFrame) -> pd.DataFrame:
    """Rank genes within each column (1 = most important)."""
    return importance.rank(ascending=False, method="average")


def top_k(importance: pd.DataFrame, k: int) -> dict[str, list[str]]:
    return {c: list(importance[c].sort_values(ascending=False).index[:k]) for c in importance}


def jaccard_matrix(importance: pd.DataFrame, k: int) -> pd.DataFrame:
    tops = {c: set(v) for c, v in top_k(importance, k).items()}
    cols = list(importance.columns)
    m = pd.DataFrame(np.eye(len(cols)), index=cols, columns=cols)
    for a, b in itertools.combinations(cols, 2):
        j = len(tops[a] & tops[b]) / len(tops[a] | tops[b])
        m.loc[a, b] = m.loc[b, a] = j
    return m


def spearman_matrix(importance: pd.DataFrame, k: int) -> pd.DataFrame:
    """Spearman correlation of ranks, restricted to the union of every column's top-k.

    Correlating all genes is dominated by thousands of near-zero, arbitrarily
    ordered tail genes; the union of top lists is where agreement matters.
    """
    union = sorted(set().union(*top_k(importance, k).values()))
    r = ranks(importance).loc[union]
    rho = spearmanr(r.values).statistic
    rho = np.atleast_2d(rho)
    return pd.DataFrame(rho, index=importance.columns, columns=importance.columns)


def consensus(importance: pd.DataFrame) -> pd.DataFrame:
    """Average rank-percentile across all columns (1.0 = top everywhere)."""
    pct = 1 - (ranks(importance) - 1) / len(importance)
    out = pd.DataFrame({
        "consensus_score": pct.mean(axis=1),
        "n_methods_top50": (ranks(importance) <= 50).sum(axis=1),
    })
    return out.sort_values("consensus_score", ascending=False)


def pairwise_summary(jac: pd.DataFrame, rho: pd.DataFrame) -> pd.DataFrame:
    """Long table of every pair with labels describing what kind of comparison it is."""
    rows = []
    for a, b in itertools.combinations(jac.columns, 2):
        ma, ea = a.split(":")
        mb, eb = b.split(":")
        kind = "same model, different explainer" if ma == mb else (
            "same explainer, different model" if ea == eb else "different model and explainer"
        )
        rows.append({"a": a, "b": b, "comparison": kind,
                     "jaccard_top_k": jac.loc[a, b], "spearman_union_top_k": rho.loc[a, b]})
    return pd.DataFrame(rows)
