"""The four classifiers compared in the study and their evaluation metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd
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
from xgboost import XGBClassifier

import sklearn

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


def build_models(seed: int = 42, pos_weight: float = 1.0, fast: bool = False) -> dict:
    """Return unfitted estimators. ``fast`` shrinks them for tests / demo runs."""
    n_trees = 60 if fast else 300
    return {
        "logreg": LogisticRegression(
            **L1, C=0.1, solver="liblinear", class_weight="balanced",
            max_iter=2000, random_state=seed,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=n_trees, max_features="sqrt", min_samples_leaf=2,
            class_weight="balanced", n_jobs=-1, random_state=seed,
        ),
        "xgboost": XGBClassifier(
            n_estimators=n_trees, max_depth=4, learning_rate=0.05, subsample=0.8,
            colsample_bytree=0.5, scale_pos_weight=pos_weight, eval_metric="logloss",
            n_jobs=-1, random_state=seed, verbosity=0,
        ),
        "mlp": MLPClassifier(
            hidden_layer_sizes=(64, 32), alpha=1e-2, learning_rate_init=1e-3,
            max_iter=100 if fast else 300, early_stopping=True, validation_fraction=0.15,
            n_iter_no_change=15, random_state=seed,
        ),
    }


def fit_all(X: pd.DataFrame, y: pd.Series, seed: int = 42, fast: bool = False) -> dict:
    # scale_pos_weight < 1 down-weights the (majority) tumor class for XGBoost,
    # mirroring class_weight="balanced" in the scikit-learn models.
    pos_weight = float((y == 0).sum() / max((y == 1).sum(), 1))
    models = build_models(seed, pos_weight, fast)
    for name, m in models.items():
        m.fit(X, y)
    return models


def refit(model, X: pd.DataFrame, y: pd.Series):
    return clone(model).fit(X, y)


def evaluate(model, X: pd.DataFrame, y: pd.Series) -> dict:
    p = np.clip(model.predict_proba(X)[:, 1], 1e-7, 1 - 1e-7)
    pred = (p >= 0.5).astype(int)
    return {
        "roc_auc": roc_auc_score(y, p),
        "pr_auc_normal": average_precision_score(1 - y, 1 - p),  # rare class is the hard one
        "balanced_accuracy": balanced_accuracy_score(y, pred),
        "f1": f1_score(y, pred),
        "mcc": matthews_corrcoef(y, pred),
        "log_loss": log_loss(y, p, labels=[0, 1]),
    }
