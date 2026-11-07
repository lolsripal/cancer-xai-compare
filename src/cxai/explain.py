"""Global gene-importance scores from three families of explanation method.

For every fitted model we compute, on the validation split:

* ``shap``        mean |SHAP value| per gene (TreeSHAP, LinearSHAP, or
                  model-agnostic permutation SHAP for the MLP). For the
                  multi-class task, |SHAP| is summed over classes.
* ``permutation`` increase in log-loss when a gene's values are shuffled
* ``native``      the model's built-in importance (|coef|, impurity, gain);
                  the MLP has none

SHAP matrices are always returned as (samples, genes, classes); the binary
task keeps only the positive (tumor) class, so it has one class column.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import shap
from sklearn.inspection import partial_dependence, permutation_importance

METHODS = ["shap", "permutation", "native"]


def _to_3d(values, n_samples: int, n_features: int, n_classes: int) -> np.ndarray:
    """Normalise SHAP's many output shapes to (n, f, C'), C' = 1 for binary."""
    if isinstance(values, list):
        values = np.stack(values, axis=-1)
    values = np.asarray(values)
    if values.ndim == 2:
        values = values[..., None]
    if values.shape[0] != n_samples and values.shape[1] == n_samples:  # (C, n, f) layout
        values = np.moveaxis(values, 0, -1)
    if n_classes == 2 and values.shape[-1] == 2:
        values = values[..., 1:]
    assert values.shape[:2] == (n_samples, n_features), values.shape
    return values


def _log_proba_fn(model, binary: bool):
    def f(x):
        p = np.clip(model.predict_proba(x), 1e-6, 1 - 1e-6)
        if binary:
            return np.log(p[:, 1] / (1 - p[:, 1]))  # logit of P(tumor)
        return np.log(p)  # per-class log-probability

    return f


def shap_values(name: str, model, X_background: pd.DataFrame, X_explain: pd.DataFrame,
                seed: int = 0) -> np.ndarray:
    """Per-sample SHAP values, shape (n_samples, n_genes, n_outputs)."""
    n_classes = len(model.classes_)
    n, f = X_explain.shape
    if name == "logreg":
        explainer = shap.LinearExplainer(model, shap.maskers.Independent(X_background, 200))
        raw = explainer.shap_values(X_explain)
    elif name in ("random_forest", "xgboost"):
        raw = shap.TreeExplainer(model).shap_values(X_explain, check_additivity=False)
    else:
        bg = shap.sample(X_background, 20, random_state=seed)
        explainer = shap.PermutationExplainer(_log_proba_fn(model, n_classes == 2),
                                              shap.maskers.Independent(bg))
        raw = explainer(X_explain.values, max_evals=2 * f + 1, silent=True).values
    return _to_3d(raw, n, f, n_classes)


def shap_importance(values: np.ndarray, genes: list[str]) -> pd.Series:
    return pd.Series(np.abs(values).sum(axis=2).mean(axis=0), index=genes)


def per_class_importance(values: np.ndarray, genes: list[str], class_names: list[str]) -> pd.DataFrame:
    """Mean |SHAP| per gene for each class output (multi-class only)."""
    return pd.DataFrame(np.abs(values).mean(axis=0), index=genes, columns=class_names[: values.shape[2]])


def permutation_scores(model, X: pd.DataFrame, y: pd.Series, repeats: int, seed: int) -> pd.Series:
    r = permutation_importance(
        model, X, y, scoring="neg_log_loss", n_repeats=repeats, random_state=seed, n_jobs=-1
    )
    return pd.Series(r.importances_mean, index=X.columns)


def native_importance(name: str, model, genes: list[str]) -> pd.Series | None:
    if name == "logreg":
        return pd.Series(np.abs(model.coef_).sum(axis=0), index=genes)
    if name in ("random_forest", "xgboost"):
        return pd.Series(model.feature_importances_, index=genes)
    return None


def explain_all(models: dict, data, cfg: dict, seed: int, log=print, permutation: bool = True):
    """Return (importance table genes x 'model:method', {model: (X_explained, shap 3-D)})."""
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
        if permutation:
            log(f"  [{name}] permutation importance")
            cols[f"{name}:permutation"] = permutation_scores(
                model, data.X_val, data.y_val, cfg["perm_repeats"], seed
            )
        nat = native_importance(name, model, data.genes)
        if nat is not None:
            cols[f"{name}:native"] = nat
    return pd.DataFrame(cols), shap_mats


def pdp_curves(models: dict, X: pd.DataFrame, genes: list[str], class_names: list[str],
               grid: int = 30) -> pd.DataFrame:
    """Partial dependence of predicted class probabilities on each gene."""
    rows = []
    for name, model in models.items():
        for g in genes:
            res = partial_dependence(model, X, [X.columns.get_loc(g)], grid_resolution=grid, kind="average")
            avg = res["average"]  # (1, grid) for binary (positive class), (C, grid) for multi-class
            targets = [class_names[1]] if avg.shape[0] == 1 else class_names
            for c, curve in zip(targets, avg):
                for v, a in zip(res["grid_values"][0], curve):
                    rows.append({"model": name, "gene": g, "class": c, "z_expression": v, "probability": a})
    return pd.DataFrame(rows)
