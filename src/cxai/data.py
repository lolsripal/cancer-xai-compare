"""Loading, labelling and splitting TCGA-BRCA gene-expression data.

The UCSC Xena ``HiSeqV2`` matrix is genes x samples with values
log2(RSEM normalized count + 1). Two prediction tasks are supported:

* ``tumor_normal``: primary tumor (barcode ``-01``) vs. solid tissue normal
  (``-11``). Many normals have a matched tumor from the same patient, so all
  splits are grouped by patient to stop the model "recognising the person".
* ``subtype``: PAM50 intrinsic subtype (LumA / LumB / Her2 / Basal) of primary
  tumors, from the ``PAM50Call_RNAseq`` column of the Xena clinical matrix.
  Those calls were made *from* the 50 PAM50 genes, so this task has a known
  answer key and acts as a positive control for the explainers.
"""

from __future__ import annotations

import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

TUMOR_CODE = "01"
NORMAL_CODE = "11"
TASKS = ["tumor_normal", "subtype"]
SUBTYPES = ["LumA", "LumB", "Her2", "Basal"]


@dataclass
class Dataset:
    """Train / validation / test splits in a shared, standardized gene space.

    * ``train`` fits the models.
    * ``val`` is where explanations (SHAP, permutation importance) are computed.
    * ``test`` is untouched until evaluation, ablation and sparse-model checks,
      so gene rankings never leak into the numbers we report.
    """

    X_train: pd.DataFrame
    X_val: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_val: pd.Series
    y_test: pd.Series
    genes: list[str]
    scaler: StandardScaler
    class_names: list[str]
    raw_train: pd.DataFrame = field(repr=False)  # unscaled log2 expression
    raw_test: pd.DataFrame = field(repr=False)

    @property
    def n_classes(self) -> int:
        return len(self.class_names)

    @property
    def binary(self) -> bool:
        return self.n_classes == 2

    def summary(self) -> dict:
        def counts(y):
            vc = y.value_counts()
            return {"n": int(len(y)), **{c: int(vc.get(i, 0)) for i, c in enumerate(self.class_names)}}

        return {
            "n_genes": len(self.genes),
            "classes": self.class_names,
            "train": counts(self.y_train),
            "val": counts(self.y_val),
            "test": counts(self.y_test),
        }


def download(url: str, path: str | Path) -> Path:
    path = Path(path)
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {url} -> {path}")
    urllib.request.urlretrieve(url, path)
    return path


def load_xena_matrix(path: str | Path) -> pd.DataFrame:
    """Return a samples x genes DataFrame from a Xena genes x samples matrix."""
    df = pd.read_csv(path, sep="\t", index_col=0)
    df = df[~df.index.duplicated(keep="first")]
    return df.T.astype(np.float32)


def sample_codes(index: pd.Index) -> pd.Index:
    return index.str.split("-").str[3].str[:2]


def label_samples(expr: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Tumor-vs-normal labels: primary tumors = 1, solid tissue normals = 0."""
    codes = sample_codes(expr.index)
    keep = codes.isin([TUMOR_CODE, NORMAL_CODE])
    X = expr.loc[keep]
    y = pd.Series((codes[keep] == TUMOR_CODE).astype(int), index=X.index, name="label")
    patients = pd.Series(X.index.str[:12], index=X.index, name="patient")
    return X, y, patients


def label_subtypes(expr: pd.DataFrame, clinical: pd.DataFrame,
                   column: str = "PAM50Call_RNAseq") -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """PAM50 subtype labels (index into ``SUBTYPES``) for primary tumors."""
    calls = clinical[column].reindex(expr.index)
    keep = (sample_codes(expr.index) == TUMOR_CODE) & calls.isin(SUBTYPES).values
    X = expr.loc[keep]
    y = pd.Series(calls[keep].map({s: i for i, s in enumerate(SUBTYPES)}).astype(int).values,
                  index=X.index, name="label")
    patients = pd.Series(X.index.str[:12], index=X.index, name="patient")
    return X, y, patients


def grouped_split(y: pd.Series, groups: pd.Series, n_splits: int, seed: int):
    """One stratified, patient-grouped split: returns (rest_index, holdout_index)."""
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    rest, hold = next(sgkf.split(np.zeros(len(y)), y, groups))
    return y.index[rest], y.index[hold]


def prepare(
    X: pd.DataFrame,
    y: pd.Series,
    patients: pd.Series,
    n_genes: int = 2000,
    min_mean_expr: float = 1.0,
    seed: int = 42,
    class_names: list[str] | None = None,
) -> Dataset:
    """Split by patient (64/16/20), select genes on train only, standardize."""
    class_names = class_names or ["normal", "tumor"]
    trainval_idx, test_idx = grouped_split(y, patients, n_splits=5, seed=seed)
    train_idx, val_idx = grouped_split(
        y.loc[trainval_idx], patients.loc[trainval_idx], n_splits=5, seed=seed
    )

    raw_train = X.loc[train_idx]
    expressed = raw_train.columns[raw_train.mean() >= min_mean_expr]
    variances = raw_train[expressed].var().sort_values(ascending=False)
    genes = list(variances.index[:n_genes])

    scaler = StandardScaler().fit(raw_train[genes])

    def scale(idx):
        return pd.DataFrame(scaler.transform(X.loc[idx, genes]), index=idx,
                            columns=genes).astype(np.float64)

    return Dataset(
        X_train=scale(train_idx), X_val=scale(val_idx), X_test=scale(test_idx),
        y_train=y.loc[train_idx], y_val=y.loc[val_idx], y_test=y.loc[test_idx],
        genes=genes, scaler=scaler, class_names=class_names,
        raw_train=X.loc[train_idx, genes], raw_test=X.loc[test_idx, genes],
    )


_EXPR_CACHE: dict[str, pd.DataFrame] = {}


def load_task_frames(task: str, cfg: dict):
    """Return (X, y, patients, class_names) for a TCGA task, downloading if needed."""
    path = str(download(cfg["url"], cfg["path"]))
    if path not in _EXPR_CACHE:
        _EXPR_CACHE[path] = load_xena_matrix(path)
    expr = _EXPR_CACHE[path]
    if task == "tumor_normal":
        return (*label_samples(expr), ["normal", "tumor"])
    if task == "subtype":
        clin_path = download(cfg["clinical_url"], cfg["clinical_path"])
        clinical = pd.read_csv(clin_path, sep="\t", index_col=0, low_memory=False)
        return (*label_subtypes(expr, clinical), list(SUBTYPES))
    raise ValueError(f"unknown task {task!r}; choose from {TASKS}")


def load_task(task: str, cfg: dict, seed: int) -> Dataset:
    X, y, patients, classes = load_task_frames(task, cfg)
    return prepare(X, y, patients, cfg["n_genes"], cfg["min_mean_expr"], seed, classes)


# ---------------------------------------------------------------------------
# Synthetic cohorts for tests and ``--demo``
# ---------------------------------------------------------------------------

def _gene_list(gene_pool, signal, n_genes, rng):
    flat = [g for s in signal for g in s] if signal and isinstance(signal[0], list) else list(signal)
    others = [g for g in gene_pool if g not in set(flat)]
    rng.shuffle(others)
    return list(dict.fromkeys(flat)) + others[: n_genes - len(set(flat))]


def make_synthetic(
    gene_pool: list[str],
    signal_genes: list[str],
    n_patients: int = 160,
    n_genes: int = 300,
    seed: int = 0,
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Fake tumor-vs-normal cohort: ``signal_genes`` are shifted up in tumors.

    Each patient contributes a tumor sample; ~30% also contribute a matched normal.
    """
    rng = np.random.default_rng(seed)
    genes = _gene_list(gene_pool, signal_genes, n_genes, rng)
    rows, labels, names = [], [], []
    for p in range(n_patients):
        patient_effect = rng.normal(0, 0.5, len(genes))
        pid = f"TCGA-SY-{p:04d}"
        samples = [(TUMOR_CODE, 1)] + ([(NORMAL_CODE, 0)] if rng.random() < 0.3 else [])
        for code, lab in samples:
            x = 6 + patient_effect + rng.normal(0, 1, len(genes))
            if lab == 1:
                x[: len(signal_genes)] += rng.normal(1.5, 0.5, len(signal_genes))
            rows.append(x)
            labels.append(lab)
            names.append(f"{pid}-{code}")
    X = pd.DataFrame(np.array(rows, dtype=np.float32), index=names, columns=genes)
    y = pd.Series(labels, index=names, name="label")
    patients = pd.Series(X.index.str[:12], index=names, name="patient")
    return X, y, patients


def make_synthetic_subtypes(
    gene_pool: list[str],
    class_signals: list[list[str]],
    n_patients: int = 240,
    n_genes: int = 300,
    seed: int = 0,
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Fake multi-class cohort: class ``c`` has ``class_signals[c]`` shifted up."""
    rng = np.random.default_rng(seed)
    genes = _gene_list(gene_pool, class_signals, n_genes, rng)
    col = {g: i for i, g in enumerate(genes)}
    rows, labels, names = [], [], []
    for p in range(n_patients):
        c = int(rng.integers(len(class_signals)))
        x = 6 + rng.normal(0, 1, len(genes))
        idx = [col[g] for g in class_signals[c]]
        x[idx] += rng.normal(1.5, 0.5, len(idx))
        rows.append(x)
        labels.append(c)
        names.append(f"TCGA-SY-{p:04d}-01")
    X = pd.DataFrame(np.array(rows, dtype=np.float32), index=names, columns=genes)
    y = pd.Series(labels, index=names, name="label")
    patients = pd.Series(X.index.str[:12], index=names, name="patient")
    return X, y, patients
