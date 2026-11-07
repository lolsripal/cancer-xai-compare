"""Global gene-importance scores from three families of explanation method.

For every fitted model we compute, on the validation split:

* ``shap``        mean |SHAP value| per gene (TreeSHAP, LinearSHAP, or
                  model-agnostic permutation SHAP for the MLP)
* ``permutation`` drop in log-loss when a gene's values are shuffled
* ``native``      the model's built-in importance (|coef|, impurity, gain);
                  the MLP has none

Each returns a pandas Series indexed by gene, higher = more important.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import shap
from sklearn.inspection import partial_dependence, permutation_importance

METHODS = ["shap", "permutation", "native"]


def _positive_class(values) -> np.ndarray:
    """Normalise the different shapes SHAP returns to (n_samples, n_features)."""
    if isinstance(values, list):
        values = values[1]
    values = np.asarray(values)
    if values.ndim == 3:
        values = values[..., 1]
    return values


def _logit_fn(model):
    def f(x):
        p = np.clip(model.predict_proba(x)[:, 1], 1e-6, 1 - 1e-6)
        return np.log(p / (1 - p))

    return f


def shap_values(
    name: str,
    model,
    X_background: pd.DataFrame,
    X_explain: pd.DataFrame,
    seed: int = 0,
) -> np.ndarray:
    """Per-sample SHAP values (log-odds scale where available) for the tumor class."""
    if name == "logreg":
        explainer = shap.LinearExplainer(model, shap.maskers.Independent(X_background, 200))
        return _positive_class(explainer.shap_values(X_explain))
    if name in ("random_forest", "xgboost"):
        explainer = shap.TreeExplainer(model)
        return _positive_class(explainer.shap_values(X_explain, check_additivity=False))
    # Model-agnostic: permutation SHAP on the logit of the predicted probability.
    bg = shap.sample(X_background, 20, random_state=seed)
    explainer = shap.PermutationExplainer(_logit_fn(model), shap.maskers.Independent(bg))
    out = explainer(X_explain.values, max_evals=2 * X_explain.shape[1] + 1, silent=True)
    return out.values


def shap_importance(values: np.ndarray, genes: list[str]) -> pd.Series:
    return pd.Series(np.abs(values).mean(axis=0), index=genes)


def permutation_scores(model, X: pd.DataFrame, y: pd.Series, repeats: int, seed: int) -> pd.Series:
    r = permutation_importance(
        model, X, y, scoring="neg_log_loss", n_repeats=repeats, random_state=seed, n_jobs=-1
    )
    return pd.Series(r.importances_mean, index=X.columns)


def native_importance(name: str, model, genes: list[str]) -> pd.Series | None:
    if name == "logreg":
        return pd.Series(np.abs(model.coef_.ravel()), index=genes)
    if name in ("random_forest", "xgboost"):
        return pd.Series(model.feature_importances_, index=genes)
    return None


def explain_all(models: dict, data, cfg: dict, seed: int, log=print):
    """Return (importance table genes x 'model:method', dict of SHAP matrices)."""
    rng = np.random.default_rng(seed)
    cols, shap_mats = {}, {}
    for name, model in models.items():
        n = cfg["mlp_shap_samples"] if name == "mlp" else cfg["shap_samples"]
        idx = rng.choice(len(data.X_val), size=min(n, len(data.X_val)), replace=False)
        X_exp = data.X_val.iloc[np.sort(idx)]
        log(f"  [{name}] SHAP on {len(X_exp)} validation samples")
        sv = shap_values(name, model, data.X_train, X_exp, seed)
        shap_mats[name] = (X_exp, sv)
        cols[f"{name}:shap"] = shap_importance(sv, data.genes)

        log(f"  [{name}] permutation importance")
        cols[f"{name}:permutation"] = permutation_scores(
            model, data.X_val, data.y_val, cfg["perm_repeats"], seed
        )
        nat = native_importance(name, model, data.genes)
        if nat is not None:
            cols[f"{name}:native"] = nat
    return pd.DataFrame(cols), shap_mats


def pdp_curves(models: dict, X: pd.DataFrame, genes: list[str], grid: int = 30) -> pd.DataFrame:
    """Partial dependence (probability of tumor) of each model on each gene."""
    rows = []
    for name, model in models.items():
        for g in genes:
            pd_res = partial_dependence(
                model, X, [X.columns.get_loc(g)], grid_resolution=grid, kind="average"
            )
            grid_vals = pd_res["grid_values"][0]
            for v, a in zip(grid_vals, pd_res["average"][0]):
                rows.append({"model": name, "gene": g, "z_expression": v, "p_tumor": a})
    return pd.DataFrame(rows)
