"""Sanity checks for the explanations themselves.

1. Chance baseline: how much would two *random* top-k lists overlap? Agreement
   numbers only mean something relative to this.
2. Label randomization (Adebayo et al., NeurIPS 2018, "Sanity Checks for
   Saliency Maps"): retrain every model on shuffled labels. A trustworthy
   explainer must then produce different gene rankings from the real model's;
   if it doesn't, it reflects the data, not what the model learned.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import explain
from .models import evaluate, fit_all


def chance_jaccard(n: int, k: int, sims: int = 20000, seed: int = 0) -> dict:
    """Mean and 95th percentile of Jaccard between two random k-subsets of n items."""
    rng = np.random.default_rng(seed)
    k = min(k, n)
    # Overlap of two random k-subsets is hypergeometric(n, k, k).
    ov = rng.hypergeometric(k, n - k, k, size=sims)
    j = ov / (2 * k - ov)
    return {"n": n, "k": k, "mean": float(j.mean()), "p95": float(np.quantile(j, 0.95))}


def label_randomization(data, real_importance: pd.DataFrame, cfg: dict, k: int, seed: int,
                        fast: bool = False, log=print) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    shuffled = data.y_train.copy()
    shuffled[:] = rng.permutation(shuffled.values)
    models = fit_all(data.X_train, shuffled, seed, fast=fast)
    rows = []
    small = dict(cfg, shap_samples=min(cfg["shap_samples"], 60), mlp_shap_samples=min(cfg["mlp_shap_samples"], 15))
    imp, _ = explain.explain_all(models, data, small, seed, log=lambda *_: None, permutation=False)
    for name, model in models.items():
        res = evaluate(model, data.X_test, data.y_test)
        col = f"{name}:shap"
        real_top = set(real_importance[col].sort_values(ascending=False).index[:k])
        rand_top = set(imp[col].sort_values(ascending=False).index[:k])
        rows.append({"model": name, "test_roc_auc_random_labels": res["roc_auc"],
                     "jaccard_vs_real_model": len(real_top & rand_top) / len(real_top | rand_top)})
    log("  label randomization done")
    return pd.DataFrame(rows)
