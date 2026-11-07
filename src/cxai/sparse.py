"""Can a tiny, human-readable model on a handful of genes match the black boxes?"""

from __future__ import annotations

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, export_text

from .models import evaluate


def sparse_models(depth: int, seed: int) -> dict:
    return {
        "small_logreg": LogisticRegression(class_weight="balanced", max_iter=2000, random_state=seed),
        f"tree_depth{depth}": DecisionTreeClassifier(
            max_depth=depth, class_weight="balanced", random_state=seed
        ),
    }


def run(ranking: list[str], data, k_values: list[int], depth: int, seed: int):
    """Train sparse models on the top-k consensus genes (ranked on validation data)."""
    rows, artifacts = [], {}
    for k in k_values:
        genes = ranking[:k]
        for name, m in sparse_models(depth, seed).items():
            m.fit(data.X_train[genes], data.y_train)
            rows.append({"model": name, "k": k, "genes": ",".join(genes),
                         **evaluate(m, data.X_test[genes], data.y_test)})
            artifacts[(name, k)] = m
    return pd.DataFrame(rows), artifacts


def describe_tree(tree: DecisionTreeClassifier, genes: list[str], scaler_means, scaler_scales) -> str:
    """Human-readable rules, with thresholds converted back to log2 expression units."""
    text = export_text(tree, feature_names=genes, decimals=3)
    lines = []
    for line in text.splitlines():
        for i, g in enumerate(genes):
            for op in ("<=", "> "):
                token = f"{g} {op}"
                if token in line:
                    z = float(line.split(op)[-1])
                    raw = z * scaler_scales[i] + scaler_means[i]
                    line = line.split(token)[0] + f"{g} {op.strip()} {raw:.2f} (log2 expr; z={z:+.2f})"
        line = line.replace("class: 1", "class: TUMOR").replace("class: 0", "class: NORMAL")
        lines.append(line)
    return "\n".join(lines)
