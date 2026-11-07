"""The four classifiers compared in the study and their evaluation metrics.

All models handle both the binary (tumor vs. normal) and the 4-class (PAM50
subtype) task. Classes are balanced with sample weights so that rare classes
(normal tissue, Her2) count as much as common ones.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    log_loss,
    matthews_corrcoef,
    roc_auc_score,
)
from sklearn.neural_network import MLPClassifier
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

# scikit-learn 1.8 deprecated `penalty=` in favour of `l1_ratio=`.
_SK_VERSION = tuple(int(x) for x in sklearn.__version__.split(".")[:2])
L1 = {"l1_ratio": 1.0} if _SK_VERSION >= (1, 8) else {"penalty": "l1"}

MODEL_NAMES = ["logreg", "random_forest", "xgboost", "mlp"]
PRETTY = {
    "logreg": "L1 Logistic Regression",
    "random_forest": "Random Forest",
    "xgboost": "XGBoost",
    "mlp": "Neural Network (MLP)",
}


def build_models(seed: int = 42, n_classes: int = 2, fast: bool = False) -> dict:
    """Return unfitted estimators. ``fast`` shrinks them for tests / demo runs."""
    n_trees = 60 if fast else 300
    if n_classes == 2:
        logreg = LogisticRegression(**L1, C=0.1, solver="liblinear", max_iter=2000, random_state=seed)
    else:
        logreg = LogisticRegression(**L1, C=0.1, solver="saga", max_iter=2000, tol=1e-3, random_state=seed)
    return {
        "logreg": logreg,
        "random_forest": RandomForestClassifier(
            n_estimators=n_trees, max_features="sqrt", min_samples_leaf=2,
            n_jobs=-1, random_state=seed,
        ),
        "xgboost": XGBClassifier(
            n_estimators=n_trees, max_depth=4, learning_rate=0.05, subsample=0.8,
            colsample_bytree=0.5, eval_metric="mlogloss" if n_classes > 2 else "logloss",
            n_jobs=4, random_state=seed, verbosity=0,
        ),
        "mlp": MLPClassifier(
            hidden_layer_sizes=(64, 32), alpha=1e-2, learning_rate_init=1e-3,
            max_iter=100 if fast else 300, early_stopping=True, validation_fraction=0.15,
            n_iter_no_change=15, random_state=seed,
        ),
    }


def fit_model(model, X: pd.DataFrame, y: pd.Series):
    """Fit with balanced sample weights (MLP early-stopping ignores weights on its holdout)."""
    w = compute_sample_weight("balanced", y)
    return model.fit(X, y, sample_weight=w)


def fit_all(X: pd.DataFrame, y: pd.Series, seed: int = 42, fast: bool = False) -> dict:
    models = build_models(seed, int(y.nunique()), fast)
    for m in models.values():
        fit_model(m, X, y)
    return models


def refit(model, X: pd.DataFrame, y: pd.Series):
    return fit_model(clone(model), X, y)


def evaluate(model, X: pd.DataFrame, y: pd.Series) -> dict:
    P = np.clip(model.predict_proba(X), 1e-7, 1 - 1e-7)
    P = P / P.sum(axis=1, keepdims=True)
    pred = P.argmax(axis=1)
    labels = list(range(P.shape[1]))
    if P.shape[1] == 2:
        p = P[:, 1]
        return {
            "roc_auc": roc_auc_score(y, p),
            "pr_auc_normal": average_precision_score(1 - y, 1 - p),  # the rare class
            "balanced_accuracy": balanced_accuracy_score(y, pred),
            "f1": f1_score(y, pred),
            "mcc": matthews_corrcoef(y, pred),
            "log_loss": log_loss(y, P, labels=labels),
        }
    return {
        "roc_auc": roc_auc_score(y, P, multi_class="ovr", average="macro", labels=labels),
        "balanced_accuracy": balanced_accuracy_score(y, pred),
        "f1": f1_score(y, pred, average="macro"),
        "mcc": matthews_corrcoef(y, pred),
        "log_loss": log_loss(y, P, labels=labels),
    }
